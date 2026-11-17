"""
job_matcher.py  --  Part B: semantic + keyword hybrid job matching engine
=========================================================================

    JD text
      |-- parse_job_description()  -> required / preferred skills, must-have rules
      |-- embed JD  -> Chroma query (chunk level; total-years / education pushed
      |                 down as metadata `where` filters)
      |-- aggregate chunk hits to candidate level  (semantic score)
      |-- keyword score  = weighted coverage of critical skills (hybrid)
      |-- experience-fit score
      |-- hard must-have filter (e.g. "5+ years Python")
      '-- score 0-100  +  reasoning  +  matched sections / excerpts

    final = 100 * (w_sem * semantic + w_kw * skill_coverage + w_exp * exp_fit)
    (defaults 0.55 / 0.35 / 0.10; `semantic` is min-max normalised over the
     retrieved candidate pool, so scores are relative to the pool.)

CLI
---
    python job_matcher.py --jd data/job_descriptions/jd_01_senior_ml_engineer.txt --top-k 10
    python job_matcher.py --jd-text "Looking for 5+ years Python and AWS ..." --no-filter
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from resume_rag import ResumeRAG, extract_skills, EDU_LEVELS

log = logging.getLogger("job_matcher")

# --------------------------------------------------------------------------- #
# Job-description parsing
# --------------------------------------------------------------------------- #
_REQ_HEAD = re.compile(r"^(?:#+\s*)?(requirements?|must[- ]haves?|required|qualifications|what you.?ll need|"
                       r"minimum qualifications|you have)\b", re.I)
_PREF_HEAD = re.compile(r"^(?:#+\s*)?(nice[- ]to[- ]haves?|preferred|bonus|good to have|a plus|"
                        r"preferred qualifications)\b", re.I)
_OTHER_HEAD = re.compile(r"^(?:#+\s*)?(responsibilities|what you.?ll do|about|the role|benefits|"
                         r"we offer|about us|overview)\b", re.I)
_YEARS_RE = re.compile(
    r"(\d+)\+?\s*(?:\+\s*)?years?\s*(?:of\s+)?(?:(?:professional|hands-on|relevant|commercial|industry|"
    r"practical)\s+)?(?:experience\s+)?(?:(?:in|with|using|of|building|developing)\s+)?([^.\n;]{0,90})", re.I)
_EDU_REQ = [
    (4, re.compile(r"\b(ph\.?d|doctorate)\b", re.I)),
    (3, re.compile(r"\b(master'?s?|m\.?s\.?)\b(?:\s+degree)?", re.I)),
    (2, re.compile(r"\b(bachelor'?s?|b\.?s\.?|b\.?tech|undergraduate degree)\b", re.I)),
]


@dataclass
class JobSpec:
    title: str
    text: str
    required_skills: list[str] = field(default_factory=list)
    required_groups: list[list[str]] = field(default_factory=list)  # 'A or B' alternatives = one group
    preferred_skills: list[str] = field(default_factory=list)
    skill_min_years: dict[str, float] = field(default_factory=dict)  # must-have skill-years
    min_total_years: float = 0.0
    min_education: int = 0

    def must_haves_text(self) -> list[str]:
        out = []
        if self.min_total_years:
            out.append(f"{self.min_total_years:g}+ years total experience")
        for s, y in self.skill_min_years.items():
            out.append(f"{y:g}+ years {s}")
        if self.min_education:
            out.append(f"{EDU_LEVELS[self.min_education]} degree or higher")
        return out


def parse_job_description(text: str, must_haves: Optional[dict] = None) -> JobSpec:
    """
    Heuristic JD parser.  `must_haves` lets the caller override / extend:
        {"skills": {"Python": 5}, "min_total_years": 5, "min_education": "master"}
    """
    lines = [l.strip() for l in text.splitlines()]
    title = next((l for l in lines if l), "Job")
    title = re.sub(r"^(job title|title|position)\s*:\s*", "", title, flags=re.I)[:100]

    mode = "other"
    req_lines, pref_lines, other_lines = [], [], []
    for l in lines:
        if _PREF_HEAD.match(l):
            mode = "pref"; continue
        if _REQ_HEAD.match(l):
            mode = "req"; continue
        if _OTHER_HEAD.match(l):
            mode = "other"; continue
        {"req": req_lines, "pref": pref_lines, "other": other_lines}[mode].append(l)

    req_text, pref_text = "\n".join(req_lines), "\n".join(pref_lines)
    required = extract_skills(req_text)
    preferred = [s for s in extract_skills(pref_text) if s not in required]
    if not required:  # JD without a requirements section: use every mentioned skill
        required = extract_skills(text)
        preferred = []

    spec = JobSpec(title=title, text=text, required_skills=required, preferred_skills=preferred)
    groups: list[list[str]] = []
    for l in (req_lines or lines):
        sk = extract_skills(l)
        if len(sk) > 1 and re.search(r"\bor\b", l, re.I) and not re.search(r"\band\b|,", l, re.I):
            groups.append(sk)                      # alternatives: any one satisfies the group
        else:
            groups.extend([x] for x in sk)
    seen, spec.required_groups = set(), []
    for g in groups:
        key = tuple(g)
        if key not in seen and not any(set(g) <= set(h) and g != h and len(h) > 1 for h in groups):
            seen.add(key); spec.required_groups.append(g)
    spec.required_skills = [s_ for g in spec.required_groups for s_ in g] or required

    # must-have rules come only from the *required* part of the JD
    for m in _YEARS_RE.finditer(req_text):
        years, tail = float(m.group(1)), m.group(2)
        skills = extract_skills(tail)
        if skills:
            for s in skills:
                spec.skill_min_years[s] = max(spec.skill_min_years.get(s, 0), years)
        else:
            spec.min_total_years = max(spec.min_total_years, years)
    for line in req_lines:
        if "prefer" in line.lower():
            continue
        for lvl, pat in _EDU_REQ:
            if pat.search(line) and re.search(r"degree|\bb\.?s|\bm\.?s|ph\.?d|bachelor|master", line, re.I):
                spec.min_education = max(spec.min_education, lvl)
                break

    if must_haves:  # explicit overrides
        for s, y in (must_haves.get("skills") or {}).items():
            spec.skill_min_years[s] = float(y)
            if s not in spec.required_skills:
                spec.required_skills.append(s)
                spec.required_groups.append([s])
        if "min_total_years" in must_haves:
            spec.min_total_years = float(must_haves["min_total_years"])
        if "min_education" in must_haves:
            v = must_haves["min_education"]
            spec.min_education = v if isinstance(v, int) else {
                "associate": 1, "bachelor": 2, "master": 3, "phd": 4}[str(v).lower()]
    return spec


# --------------------------------------------------------------------------- #
# Matching engine
# --------------------------------------------------------------------------- #
@dataclass
class Weights:
    semantic: float = 0.55
    keyword: float = 0.35
    experience: float = 0.10

    def normalised(self) -> "Weights":
        t = self.semantic + self.keyword + self.experience
        return Weights(self.semantic / t, self.keyword / t, self.experience / t)


def _excerpt(text: str, max_len: int = 280) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= max_len else t[: max_len - 1].rstrip() + "…"


class JobMatcher:
    def __init__(self, rag: ResumeRAG, weights: Optional[Weights] = None,
                 pool_chunks: int = 120):
        self.rag = rag
        self.weights = (weights or Weights()).normalised()
        self.pool_chunks = pool_chunks
        self._cands: Optional[dict[str, dict]] = None

    @property
    def candidates(self) -> dict[str, dict]:
        if self._cands is None:
            self._cands = self.rag.candidates()
        return self._cands

    def refresh(self) -> None:
        self._cands = None

    # ---- must-have checking ---------------------------------------------- #
    @staticmethod
    def check_must_haves(c: dict, spec: JobSpec) -> list[str]:
        """Return list of failure reasons (empty list == passes all must-haves)."""
        fails = []
        if spec.min_total_years and c["experience_years"] < spec.min_total_years:
            fails.append(f"{c['experience_years']:g} yrs total experience < required {spec.min_total_years:g}")
        for skill, need in spec.skill_min_years.items():
            have = c["skill_years"].get(skill, 0.0)
            if have < need:
                fails.append(f"{skill}: {have:g} yrs < required {need:g}")
        if spec.min_education and c["education_level"] < spec.min_education:
            fails.append(f"education {EDU_LEVELS[c['education_level']]} < required {EDU_LEVELS[spec.min_education]}")
        return fails

    # ---- chroma pre-filter ---------------------------------------------- #
    @staticmethod
    def build_where(spec: JobSpec) -> Optional[dict]:
        conds = []
        if spec.min_total_years:
            conds.append({"experience_years": {"$gte": float(spec.min_total_years)}})
        if spec.min_education:
            conds.append({"education_level": {"$gte": int(spec.min_education)}})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    # ---- main entry ------------------------------------------------------ #
    def match(self, job_description: str, top_k: int = 10, apply_filters: bool = True,
              must_haves: Optional[dict] = None, weights: Optional[Weights] = None,
              strict: bool = True) -> dict:
        """
        strict=True  -> candidates failing must-haves are removed (listed in `filtered_out`)
        strict=False -> they stay but are down-weighted x0.6 and flagged
        """
        t0 = time.perf_counter()
        w = (weights or self.weights).normalised()
        spec = parse_job_description(job_description, must_haves)
        n_chunks = self.rag.collection.count()
        n_query = min(self.pool_chunks, n_chunks)

        # 1) semantic retrieval (chunk level) with metadata pre-filter ------------
        where = self.build_where(spec) if (apply_filters and strict) else None
        res = self.rag.search_chunks(job_description, k=n_query, where=where)
        hits = list(zip(res["documents"][0], res["metadatas"][0],
                        [1.0 - d for d in res["distances"][0]]))

        by_cand: dict[str, list[tuple[str, dict, float]]] = {}
        for doc, md, sim in hits:
            by_cand.setdefault(md["candidate_id"], []).append((doc, md, sim))

        # Candidates not seen in the chunk pool still get scored on keywords/exp
        # (pure-keyword recall safety net for hybrid search).  Candidates removed
        # by the metadata pre-filter are reported in `filtered_out` with reasons.
        filtered_out: list[dict] = []
        pool_ids = set(by_cand)
        for cid, c in self.candidates.items():
            if cid in pool_ids:
                continue
            if where and (c["experience_years"] < spec.min_total_years or
                          c["education_level"] < spec.min_education):
                filtered_out.append({"candidate_name": c["name"], "resume_path": c["resume_path"],
                                     "reasons": self.check_must_haves(c, spec)})
                continue
            by_cand[cid] = []

        # 2) candidate-level semantic score ---------------------------------------
        raw_sem: dict[str, float] = {}
        floor = min((h[2] for lst in by_cand.values() for h in lst), default=0.0)
        for cid, lst in by_cand.items():
            if not lst:
                raw_sem[cid] = floor
                continue
            sims = sorted((h[2] for h in lst), reverse=True)
            raw_sem[cid] = 0.6 * sims[0] + 0.4 * float(np.mean(sims[:3]))
        lo, hi = min(raw_sem.values(), default=0.0), max(raw_sem.values(), default=1.0)
        span = (hi - lo) or 1.0

        # 3) assemble scored results --------------------------------------------
        scored = []
        groups = spec.required_groups or [[s] for s in spec.required_skills]
        crit_total = (len(groups) + 0.4 * len(spec.preferred_skills)) or 1.0

        for cid, lst in by_cand.items():
            c = self.candidates[cid]
            fails = self.check_must_haves(c, spec) if apply_filters else []
            if fails and strict:
                filtered_out.append({"candidate_name": c["name"], "resume_path": c["resume_path"],
                                     "reasons": fails})
                continue
            have = set(c["skills_list"])
            matched_req = [s for g in groups for s in g if s.lower() in have]
            matched_pref = [s for s in spec.preferred_skills if s.lower() in have]
            n_groups_hit = sum(any(s.lower() in have for s in g) for g in groups)
            missing = ["/".join(g) for g in groups if not any(s.lower() in have for s in g)]
            kw = (n_groups_hit + 0.4 * len(matched_pref)) / crit_total
            sem = (raw_sem[cid] - lo) / span
            target = spec.min_total_years or 3.0
            exp_fit = min(1.0, c["experience_years"] / target)
            score = 100.0 * (w.semantic * sem + w.keyword * kw + w.experience * exp_fit)
            if fails:
                score *= 0.6
            scored.append((score, cid, c, lst, sem, kw, exp_fit, matched_req, matched_pref, missing, fails))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 4) format output ------------------------------------------------------
        top = []
        for rank, (score, cid, c, lst, sem, kw, exp_fit, mreq, mpref, missing, fails) in enumerate(scored[:top_k], 1):
            lst_sorted = sorted(lst, key=lambda h: h[2], reverse=True)
            seen_secs, sections, excerpts = [], [], []
            for doc, md, sim in lst_sorted:
                tag = md["section"] + (f" ({md['label']})" if md["label"] else "")
                if md["section"] not in seen_secs:
                    seen_secs.append(md["section"])
                if len(excerpts) < 3 and tag not in sections:
                    sections.append(tag)
                    excerpts.append(f"[{md['section']}] {_excerpt(doc)}")
            top.append({
                "rank": rank,
                "candidate_name": c["name"],
                "resume_path": c["resume_path"],
                "match_score": int(round(max(0.0, min(100.0, score)))),
                "matched_skills": mreq + mpref,
                "missing_required_skills": missing,
                "matched_sections": seen_secs,
                "relevant_excerpts": excerpts,
                "experience_years": c["experience_years"],
                "education": c["education_degree"],
                "score_breakdown": {"semantic": round(sem, 3), "keyword": round(kw, 3),
                                    "experience_fit": round(exp_fit, 3)},
                "must_have_failures": fails,
                "reasoning": self._reason(c, spec, mreq, mpref, missing, seen_secs, sem, fails, len(groups)),
            })

        return {
            "job_description": job_description,
            "parsed_requirements": {
                "title": spec.title,
                "required_skills": spec.required_skills,
                "preferred_skills": spec.preferred_skills,
                "must_haves": spec.must_haves_text(),
            },
            "top_matches": top,
            "filtered_out": sorted(filtered_out, key=lambda x: x["candidate_name"]),
            "meta": {
                "k": top_k, "candidates_scored": len(scored), "candidates_filtered": len(filtered_out),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "embedder": self.rag.embedder.name,
                "weights": asdict(w),
            },
        }

    @staticmethod
    def _reason(c, spec, mreq, mpref, missing, sections, sem, fails, n_groups) -> str:
        parts = []
        strength = "Strong" if sem > 0.75 else "Good" if sem > 0.5 else "Partial"
        parts.append(f"{strength} semantic match to the role"
                     + (f" (strongest in: {', '.join(sections[:3])})" if sections else ""))
        if n_groups:
            parts.append(f"covers {n_groups - len(missing)}/{n_groups} required skills"
                         + (f" ({', '.join(mreq[:6])})" if mreq else ""))
        if missing:
            parts.append(f"missing: {', '.join(missing[:5])}")
        if mpref:
            parts.append(f"bonus skills: {', '.join(mpref[:4])}")
        yrs = [f"{s} {c['skill_years'].get(s, 0):g}y" for s in spec.skill_min_years]
        parts.append(f"{c['experience_years']:g} yrs total experience" + (f" ({', '.join(yrs)})" if yrs else ""))
        if fails:
            parts.append("FAILS must-haves: " + "; ".join(fails))
        elif spec.must_haves_text():
            parts.append("meets all must-have requirements")
        return "; ".join(parts) + "."


# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Match resumes to a job description")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--jd", help="path to job description .txt")
    g.add_argument("--jd-text", help="job description as a string")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--backend", default=os.getenv("EMBED_BACKEND", "auto"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--no-filter", action="store_true", help="disable must-have filtering")
    ap.add_argument("--soft", action="store_true", help="penalise instead of exclude")
    ap.add_argument("--out", help="write JSON to this path")
    a = ap.parse_args()

    jd = Path(a.jd).read_text(encoding="utf-8") if a.jd else a.jd_text
    matcher = JobMatcher(ResumeRAG(a.db, a.backend))
    if matcher.rag.collection.count() == 0:
        raise SystemExit("Vector DB is empty - run: python resume_rag.py ingest --resumes data/resumes")
    result = matcher.match(jd, top_k=a.top_k, apply_filters=not a.no_filter, strict=not a.soft)
    out = json.dumps(result, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
        log.info("wrote %s", a.out)
    print(out)


if __name__ == "__main__":
    main()
