"""
resume_rag.py  --  Part A: RAG system setup for resume screening
=================================================================

Pipeline
--------
    load (PDF / TXT / MD)  ->  section-aware chunking  ->  metadata extraction
        ->  embeddings (pluggable backend)  ->  ChromaDB (persistent, cosine)

Key design decisions
--------------------
* Section-aware chunking: a resume is split on headings (Summary, Skills,
  Experience, Education, ...). The Experience section is further split *per job*
  so one chunk == one role. Oversized chunks are split on bullet boundaries
  with overlap. Every chunk is embedded with a context header
  ("Candidate | Section | Role") so that short chunks (e.g. Education) are not
  semantically orphaned.
* Metadata (name, skills, total experience years, per-skill years, education)
  is computed once per resume and stored on EVERY chunk, so Chroma `where`
  filters can be applied at retrieval time.
* Embedding backends: sentence-transformers (HuggingFace, default), OpenAI,
  or an offline hashing baseline (no download / no API key).

CLI
---
    python resume_rag.py ingest --resumes data/resumes --reset
    python resume_rag.py stats
    python resume_rag.py search "kubernetes terraform sre" --k 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

log = logging.getLogger("resume_rag")

# --------------------------------------------------------------------------- #
# 1. Skill taxonomy (canonical name -> aliases).  Shared with job_matcher.py.
# --------------------------------------------------------------------------- #
SKILL_TAXONOMY: dict[str, list[str]] = {
    # languages
    "Python": ["python", "python3"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "Go": ["golang"],
    "C++": ["c++"],
    "C#": ["c#"],
    "Scala": ["scala"],
    "Swift": ["swift"],
    "Kotlin": ["kotlin"],
    "SQL": ["sql"],
    # ML / data science
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "neural networks", "neural network"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision"],
    "LLMs": ["llm", "llms", "large language models", "large language model"],
    "RAG": ["rag", "retrieval-augmented generation", "retrieval augmented generation"],
    "MLOps": ["mlops"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Statistics": ["statistics", "statistical modeling", "statistical modelling"],
    "A/B Testing": ["a/b testing", "a/b tests", "experimentation"],
    # data engineering
    "Spark": ["spark", "pyspark", "apache spark"],
    "Airflow": ["airflow", "apache airflow"],
    "Kafka": ["kafka", "apache kafka"],
    "dbt": ["dbt"],
    "Snowflake": ["snowflake"],
    "BigQuery": ["bigquery"],
    "ETL": ["etl", "elt", "data pipelines", "data pipeline"],
    # databases
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    # cloud / devops
    "AWS": ["aws", "amazon web services"],
    "GCP": ["gcp", "google cloud"],
    "Azure": ["azure"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Terraform": ["terraform"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous delivery"],
    "Jenkins": ["jenkins"],
    "Linux": ["linux"],
    "Prometheus": ["prometheus"],
    "Grafana": ["grafana"],
    # web / backend / mobile
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular"],
    "Vue": ["vue", "vue.js"],
    "Node.js": ["node.js", "nodejs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring Boot": ["spring boot", "spring"],
    "REST APIs": ["rest", "restful", "rest api", "rest apis"],
    "GraphQL": ["graphql"],
    "Microservices": ["microservices", "microservice"],
    "HTML/CSS": ["html", "css", "html5", "css3"],
    "Android": ["android"],
    "iOS": ["ios"],
    # product / analytics
    "Agile": ["agile", "scrum"],
    "Product Management": ["product management", "product roadmap", "roadmapping"],
    "Tableau": ["tableau"],
    "Excel": ["excel"],
    "Figma": ["figma"],
    "JIRA": ["jira"],
}

_BOUNDARY_L = r"(?<![A-Za-z0-9+#])"
_BOUNDARY_R = r"(?![A-Za-z0-9+#])"
_SKILL_PATTERNS: dict[str, re.Pattern] = {}
for _canon, _aliases in SKILL_TAXONOMY.items():
    _alts = sorted({re.escape(a) for a in _aliases + [_canon.lower()]}, key=len, reverse=True)
    _SKILL_PATTERNS[_canon] = re.compile(
        _BOUNDARY_L + "(?:" + "|".join(_alts) + ")" + _BOUNDARY_R, re.IGNORECASE
    )
# "Go" the language is too ambiguous as a word; accept it only when capitalised
# and followed by common language-context words.
_GO_PATTERN = re.compile(r"\bGo\b(?=\s*(?:\(|,|/|programming|language|services?))")


def extract_skills(text: str) -> list[str]:
    """Return canonical skill names found in `text` (order = taxonomy order)."""
    found = [c for c, p in _SKILL_PATTERNS.items() if p.search(text)]
    if "Go" not in found and _GO_PATTERN.search(text):
        found.append("Go")
    return found


# --------------------------------------------------------------------------- #
# 2. Embedding backends
# --------------------------------------------------------------------------- #
class Embedder:
    name: str = "base"
    dim: int = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """HuggingFace sentence-transformers (local, free).  Default: all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = "st-" + model_name.split("/")[-1]
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        v = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return v.tolist()


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings (needs OPENAI_API_KEY)."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name
        self.name = "openai-" + model_name
        self.dim = 1536 if "small" in model_name else 3072

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):
            resp = self.client.embeddings.create(model=self.model_name, input=texts[i : i + 96])
            out.extend(d.embedding for d in resp.data)
        return out


class HashingEmbedder(Embedder):
    """
    Offline baseline: sublinear-TF hashed uni+bi-grams, L2-normalised.
    Needs no model download / API key.  It is LEXICAL, not truly semantic --
    use it for smoke tests and as an ablation baseline against real embeddings.
    """

    def __init__(self, n_features: int = 4096):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(
            n_features=n_features, ngram_range=(1, 2), alternate_sign=False,
            norm=None, stop_words="english", lowercase=True,
            token_pattern=r"(?u)\b[\w+#.]+\b",
        )
        self.name = f"hash-{n_features}"
        self.dim = n_features

    def embed(self, texts: list[str]) -> list[list[float]]:
        X = self.vec.transform(texts).toarray().astype(np.float32)
        X = np.log1p(X)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (X / norms).tolist()


def get_embedder(backend: str = "auto") -> Embedder:
    """backend: auto | st | openai | hash   (auto = st, falling back to hash)."""
    backend = (backend or "auto").lower()
    if backend in ("st", "sentence-transformers", "hf", "huggingface"):
        return SentenceTransformerEmbedder()
    if backend == "openai":
        return OpenAIEmbedder()
    if backend == "hash":
        return HashingEmbedder()
    try:  # auto
        return SentenceTransformerEmbedder()
    except Exception as e:  # noqa: BLE001
        log.warning("sentence-transformers unavailable (%s); falling back to HashingEmbedder", e)
        return HashingEmbedder()


# --------------------------------------------------------------------------- #
# 3. Document loading
# --------------------------------------------------------------------------- #
SUPPORTED_EXT = {".pdf", ".txt", ".md"}


def load_resume_text(path: Path) -> str:
    """Load text from PDF / TXT / MD (mirrors the Milestone-1 file-system tool)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {path}")


def list_resume_files(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in SUPPORTED_EXT)


# --------------------------------------------------------------------------- #
# 4. Section parsing + chunking
# --------------------------------------------------------------------------- #
SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "objective", "about", "about me"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "technologies"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment", "employment history", "work history"],
    "education": ["education", "academic background", "academics", "education & training"],
    "projects": ["projects", "selected projects", "personal projects"],
    "certifications": ["certifications", "certificates", "licenses", "licenses & certifications"],
}
_HEADING_LOOKUP = {a: sec for sec, al in SECTION_ALIASES.items() for a in al}

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATE_RANGE_RE = re.compile(
    rf"(?:(?P<m1>{_MON})\s+)?(?P<y1>(?:19|20)\d{{2}})\s*(?:-|\u2013|\u2014|to)\s*"
    rf"(?:(?P<m2>{_MON})\s+)?(?P<y2>(?:19|20)\d{{2}}|present|current|now)",
    re.IGNORECASE,
)


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def today_month_index() -> int:
    t = date.today()
    return _month_index(t.year, t.month)


def parse_date_range(line: str) -> Optional[tuple[int, int]]:
    """Return (start_month_idx, end_month_idx) for a 'Jan 2019 - Present' style range."""
    m = DATE_RANGE_RE.search(line)
    if not m:
        return None
    y1 = int(m["y1"])
    m1 = _MONTHS[m["m1"][:3].lower()] if m["m1"] else 7  # year-only -> mid-year (unbiased)
    end_raw = m["y2"].lower()
    if end_raw in ("present", "current", "now"):
        end = today_month_index()
    else:
        y2 = int(end_raw)
        m2 = _MONTHS[m["m2"][:3].lower()] if m["m2"] else 7
        end = min(_month_index(y2, m2), today_month_index())
    start = _month_index(y1, m1)
    return (start, max(start, end))


def merge_intervals(iv: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def months_in(iv: Iterable[tuple[int, int]]) -> int:
    return sum(e - s for s, e in merge_intervals(iv))


@dataclass
class Section:
    name: str
    text: str


def split_sections(text: str) -> tuple[str, list[Section]]:
    """Return (header_block, sections).  header_block = lines before the first heading."""
    header: list[str] = []
    sections: list[Section] = []
    cur_name: Optional[str] = None
    cur_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        key = re.sub(r"[:\s]+$", "", line).lower()
        if key in _HEADING_LOOKUP and len(line) < 40:
            if cur_name is not None:
                sections.append(Section(cur_name, "\n".join(cur_lines).strip()))
            cur_name, cur_lines = _HEADING_LOOKUP[key], []
        elif cur_name is None:
            header.append(line)
        else:
            cur_lines.append(line)
    if cur_name is not None:
        sections.append(Section(cur_name, "\n".join(cur_lines).strip()))
    return "\n".join(header).strip(), sections


@dataclass
class Job:
    header: str
    text: str
    interval: Optional[tuple[int, int]]
    skills: list[str] = field(default_factory=list)


def split_jobs(experience_text: str) -> list[Job]:
    """Each line holding a date range starts a new job entry."""
    jobs: list[Job] = []
    cur_head, cur_body = None, []
    for line in experience_text.splitlines():
        if not line.strip():
            continue
        if DATE_RANGE_RE.search(line) and not line.lstrip().startswith(("-", "*", "\u2022")):
            if cur_head is not None:
                jobs.append(_mk_job(cur_head, cur_body))
            cur_head, cur_body = line.strip(), []
        elif cur_head is not None:
            cur_body.append(line.strip())
        else:  # text before the first dated entry
            cur_head, cur_body = line.strip(), []
    if cur_head is not None:
        jobs.append(_mk_job(cur_head, cur_body))
    return jobs


def _mk_job(head: str, body: list[str]) -> Job:
    text = "\n".join([head] + body)
    return Job(head, text, parse_date_range(head), extract_skills(text))


def split_long(text: str, max_chars: int = 900, overlap_lines: int = 1) -> list[str]:
    """Split on line (bullet) boundaries, never mid-sentence, with small overlap."""
    if len(text) <= max_chars:
        return [text]
    lines, chunks, cur = text.splitlines(), [], []
    for ln in lines:
        if cur and sum(len(x) + 1 for x in cur) + len(ln) > max_chars:
            chunks.append("\n".join(cur))
            cur = cur[-overlap_lines:] if overlap_lines else []
        cur.append(ln)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


@dataclass
class Chunk:
    text: str          # text shown to the user as an excerpt
    section: str       # summary | skills | experience | education | ...
    label: str = ""    # e.g. job title/company for experience chunks
    idx: int = 0

    def embed_text(self, name: str) -> str:
        head = f"Candidate: {name} | Section: {self.section}"
        if self.label:
            head += f" | {self.label}"
        return head + "\n" + self.text


def chunk_resume(header: str, sections: list[Section], full_text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not sections:  # unstructured resume -> fixed-size fallback
        for t in split_long(full_text):
            chunks.append(Chunk(t, "other"))
    else:
        if header:
            chunks.append(Chunk(header, "header"))
        for sec in sections:
            if not sec.text:
                continue
            if sec.name == "experience":
                for job in split_jobs(sec.text):
                    label = job.header.split("|")[0].strip()[:80]
                    for part in split_long(job.text):
                        chunks.append(Chunk(part, "experience", label))
            else:
                for part in split_long(sec.text):
                    chunks.append(Chunk(part, sec.name))
    for i, c in enumerate(chunks):
        c.idx = i
    return chunks


# --------------------------------------------------------------------------- #
# 5. Metadata extraction
# --------------------------------------------------------------------------- #
EDU_LEVELS = {0: "unknown", 1: "associate/diploma", 2: "bachelor", 3: "master", 4: "phd"}
_EDU_PATTERNS = [
    (4, re.compile(r"\b(ph\.?\s?d\.?|doctorate|doctor of philosophy)\b", re.I)),
    (3, re.compile(r"\b(m\.?\s?s\.?c?\.?|m\.?tech|m\.?eng|m\.?a\.?|mba|master'?s?)(?=\b|\s|\.)(?!\w)", re.I)),
    (2, re.compile(r"\b(b\.?\s?s\.?c?\.?|b\.?tech|b\.?e\.?|b\.?a\.?|bachelor'?s?)(?=\b|\s|\.)(?!\w)", re.I)),
    (1, re.compile(r"\b(associate|diploma)\b", re.I)),
]
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def extract_name(header: str, full_text: str) -> str:
    for ln in (header or full_text).splitlines():
        ln = ln.strip()
        if ln and not _EMAIL_RE.search(ln) and not re.search(r"\d{3}", ln) and len(ln.split()) <= 5:
            return re.sub(r"\s+", " ", ln).title() if ln.isupper() else ln
    return "Unknown"


def extract_education(edu_text: str) -> tuple[int, str]:
    """Return (highest degree level, short description)."""
    best, best_line = 0, ""
    for line in edu_text.splitlines():
        for level, pat in _EDU_PATTERNS:
            if pat.search(line) and level > best:
                best, best_line = level, line.strip()
                break
    return best, best_line


_CLAIM_RE = re.compile(r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:\w+\s+)?(?:experience\s+)?(?:in|with|using)\s+([^.\n;]{1,80})", re.I)


@dataclass
class ResumeMeta:
    name: str
    email: str
    skills: list[str]
    experience_years: float
    skill_years: dict[str, float]
    education_level: int
    education_degree: str


def extract_metadata(header: str, sections: list[Section], full_text: str) -> ResumeMeta:
    sec = {s.name: s.text for s in sections}
    name = extract_name(header, full_text)
    em = _EMAIL_RE.search(full_text)

    jobs = split_jobs(sec.get("experience", ""))
    dated = [j for j in jobs if j.interval]
    total_months = months_in([j.interval for j in dated])
    exp_years = round(total_months / 12, 1)
    if not dated:  # fall back to stated total, e.g. "8 years of experience"
        m = re.search(r"(\d+)\+?\s*years", sec.get("summary", full_text), re.I)
        exp_years = float(m.group(1)) if m else 0.0

    skill_iv: dict[str, list[tuple[int, int]]] = {}
    for j in dated:
        for s in j.skills:
            skill_iv.setdefault(s, []).append(j.interval)
    skill_years = {s: round(months_in(iv) / 12, 1) for s, iv in skill_iv.items()}

    # Explicit self-claims ("6 years of experience with Python") -> keep the max
    claim_zone = sec.get("summary", "") + "\n" + sec.get("skills", "")
    for m in _CLAIM_RE.finditer(claim_zone):
        for s in extract_skills(m.group(2)):
            skill_years[s] = max(skill_years.get(s, 0.0), float(m.group(1)))

    all_skills = sorted(set(extract_skills(
        sec.get("skills", "") + "\n" + sec.get("experience", "") + "\n" +
        sec.get("projects", "") + "\n" + sec.get("summary", "")
    )))
    for s in all_skills:
        skill_years.setdefault(s, 0.0)

    edu_level, edu_line = extract_education(sec.get("education", ""))
    return ResumeMeta(name, em.group(0) if em else "", all_skills, exp_years,
                      skill_years, edu_level, edu_line)


def skills_to_flat(skills: Iterable[str]) -> str:
    """'|python|sql|' -- lets Chroma `where_document`/`$contains` style checks work."""
    return "|" + "|".join(s.lower() for s in skills) + "|"


# --------------------------------------------------------------------------- #
# 6. Vector store wrapper
# --------------------------------------------------------------------------- #
def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-")[:40]


class ResumeRAG:
    def __init__(self, db_dir: str = "chroma_db", backend: str = "auto",
                 embedder: Optional[Embedder] = None):
        import chromadb

        self.embedder = embedder or get_embedder(backend)
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection_name = f"resumes_{_slug(self.embedder.name)}"
        self.collection = self.client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    # -- ingestion ---------------------------------------------------------- #
    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def process_resume(self, path: Path, root: Optional[Path] = None) -> dict:
        text = load_resume_text(path)
        header, sections = split_sections(text)
        meta = extract_metadata(header, sections, text)
        chunks = chunk_resume(header, sections, text)
        rel = str(path.relative_to(root.parent)) if root and root.parent in path.parents else str(path)
        cid = hashlib.md5(rel.encode()).hexdigest()[:12]
        return {"candidate_id": cid, "path": rel, "meta": meta, "chunks": chunks}

    def ingest_directory(self, resumes_dir: str, reset: bool = False) -> dict:
        root = Path(resumes_dir)
        files = list_resume_files(root)
        if not files:
            raise FileNotFoundError(f"No resumes found in {resumes_dir}")
        if reset:
            self.reset()
        t0 = time.perf_counter()
        ids, docs, embed_inputs, metas = [], [], [], []
        n_resumes = 0
        for f in files:
            try:
                rec = self.process_resume(f, root)
            except Exception as e:  # noqa: BLE001
                log.error("Failed to process %s: %s", f, e)
                continue
            n_resumes += 1
            m: ResumeMeta = rec["meta"]
            for c in rec["chunks"]:
                ids.append(f"{rec['candidate_id']}-{c.idx}")
                docs.append(c.text)
                embed_inputs.append(c.embed_text(m.name))
                metas.append({
                    "candidate_id": rec["candidate_id"],
                    "name": m.name,
                    "email": m.email,
                    "resume_path": rec["path"],
                    "section": c.section,
                    "label": c.label,
                    "chunk_idx": c.idx,
                    "skills": skills_to_flat(m.skills),
                    "experience_years": float(m.experience_years),
                    "education_level": int(m.education_level),
                    "education_degree": m.education_degree,
                    "skill_years_json": json.dumps(m.skill_years),
                })
        t_parse = time.perf_counter() - t0
        t1 = time.perf_counter()
        vectors = self.embedder.embed(embed_inputs)
        t_embed = time.perf_counter() - t1
        for i in range(0, len(ids), 500):
            sl = slice(i, i + 500)
            self.collection.upsert(ids=ids[sl], documents=docs[sl],
                                   embeddings=vectors[sl], metadatas=metas[sl])
        stats = {
            "resumes": n_resumes, "chunks": len(ids), "embedder": self.embedder.name,
            "parse_seconds": round(t_parse, 3), "embed_seconds": round(t_embed, 3),
            "total_seconds": round(time.perf_counter() - t0, 3),
        }
        log.info("Ingested %s", stats)
        return stats

    # -- read helpers ------------------------------------------------------- #
    def candidates(self) -> dict[str, dict]:
        """One metadata record per candidate (deserialised)."""
        res = self.collection.get(include=["metadatas"])
        out: dict[str, dict] = {}
        for m in res["metadatas"]:
            cid = m["candidate_id"]
            if cid not in out:
                out[cid] = {**m, "skills_list": [s for s in m["skills"].strip("|").split("|") if s],
                            "skill_years": json.loads(m["skill_years_json"])}
        return out

    def search_chunks(self, query: str, k: int = 10, where: Optional[dict] = None):
        qv = self.embedder.embed([query])
        return self.collection.query(query_embeddings=qv, n_results=k, where=where,
                                     include=["documents", "metadatas", "distances"])


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Resume RAG ingestion / inspection")
    ap.add_argument("cmd", choices=["ingest", "stats", "search", "inspect"])
    ap.add_argument("query", nargs="?", help="query text (search) or file path (inspect)")
    ap.add_argument("--resumes", default="data/resumes")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--backend", default=os.getenv("EMBED_BACKEND", "auto"),
                    help="auto | st | openai | hash")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    if a.cmd == "inspect":  # show parsing result for a single resume
        p = Path(a.query)
        text = load_resume_text(p)
        h, secs = split_sections(text)
        meta = extract_metadata(h, secs, text)
        print(json.dumps(asdict(meta), indent=2))
        for c in chunk_resume(h, secs, text):
            print(f"\n--- chunk {c.idx} [{c.section}] {c.label}\n{c.text[:300]}")
        return

    rag = ResumeRAG(a.db, a.backend)
    if a.cmd == "ingest":
        print(json.dumps(rag.ingest_directory(a.resumes, reset=a.reset), indent=2))
    elif a.cmd == "stats":
        c = rag.candidates()
        print(json.dumps({"collection": rag.collection_name, "chunks": rag.collection.count(),
                          "candidates": len(c)}, indent=2))
    elif a.cmd == "search":
        r = rag.search_chunks(a.query, a.k)
        for doc, md, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
            print(f"{1 - dist:.3f}  {md['name']:<22} [{md['section']}] {doc[:90]!r}")


if __name__ == "__main__":
    main()
