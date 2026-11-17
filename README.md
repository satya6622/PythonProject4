# Resume RAG + Job Matching Engine

Semantic resume search with section-aware chunking, metadata filtering, hybrid scoring and explainable 0-100 match scores.

```
resumes (PDF/TXT) -> load -> section-aware chunks -> metadata (name, skills, years, education)
      -> embeddings -> ChromaDB (cosine) -> hybrid retrieval -> must-have filter -> score + reasoning (JSON)
```

## Files
| File | Purpose |
|---|---|
| `resume_rag.py` | **Part A.** Loading, chunking, metadata extraction, embedding backends, ChromaDB ingestion (+ CLI) |
| `job_matcher.py` | **Part B.** JD parsing, semantic + keyword hybrid search, scoring, must-have filters, reasoning (+ CLI) |
| `generate_dataset.py` | Builds 40 resumes (PDF/TXT), 8 JDs, `data/ground_truth.json` |
| `evaluate.py` | Retrieval accuracy (P@5, R-Prec, R@10, MRR, nDCG, Hit@1), latency, extraction accuracy |
| `experiments.ipynb` | Experiments, ablations, scaling, error analysis (executed, outputs saved) |
| `results/` | `metrics.json`, `retrieval_by_mode.png`, `sample_output_jd01.json` |
| `data/` | `resumes/` (40), `job_descriptions/` (8), `ground_truth.json` |

## Quick start
```bash
pip install -r requirements.txt
python generate_dataset.py                     # (data/ is already included)
python resume_rag.py ingest --resumes data/resumes --reset      # default: MiniLM embeddings
python job_matcher.py --jd data/job_descriptions/jd_01_senior_ml_engineer.txt --top-k 10 --out out.json
python evaluate.py --out results               # metrics + chart
jupyter notebook experiments.ipynb
```
Embedding backend: `--backend auto|st|openai|hash` or `EMBED_BACKEND=...`
(`st` = sentence-transformers MiniLM, `openai` = text-embedding-3-small with `OPENAI_API_KEY`,
`hash` = offline lexical baseline, no download). `auto` tries MiniLM, falls back to `hash`.

**Important:** the results shipped in `results/` and the notebook were produced with the offline `hash` backend
(the build sandbox had no access to HuggingFace). Re-run `python evaluate.py --backend st` and re-execute the notebook
(`EMBED_BACKEND=st jupyter nbconvert --to notebook --execute --inplace experiments.ipynb`) to get MiniLM numbers for your submission.

## Design
* **Chunking:** split on resume headings (Summary / Skills / Experience / Education / Projects / Certifications), Experience split *per job*, long chunks split on bullet boundaries with 1-line overlap. Each chunk embedded with a `Candidate | Section | Role` header.
* **Metadata (stored on every chunk):** `name, email, skills` (taxonomy of ~70 skills with aliases, e.g. k8s -> Kubernetes), `experience_years` (union of dated job intervals, overlaps merged), `skill_years_json` (per-skill years from the jobs that mention the skill), `education_level` (0-4) and degree text.
* **Search:** JD embedded -> Chroma top-N chunks (total-years / education pushed down as `where` filters) -> aggregated to candidates: `0.6*best_chunk + 0.4*mean(top3)`.
* **Hybrid score:** `100 * (0.55*semantic + 0.35*skill_coverage + 0.10*experience_fit)`. Skill coverage = required skill groups hit (+0.4 weight for nice-to-haves; "A or B" counts once). Semantic is min-max normalised over the candidate pool.
* **Must-haves** (parsed from JD, e.g. "5+ years of experience with Python", "Bachelor's degree", or passed as `must_haves={"skills": {"Python": 5}}`): `strict=True` removes failures (listed with reasons in `filtered_out`); `strict=False` penalises x0.6 and flags.
* **Output** follows the required schema (`job_description`, `top_matches[...]` with `candidate_name, resume_path, match_score, matched_skills, relevant_excerpts, reasoning`) plus `missing_required_skills, matched_sections, score_breakdown, must_have_failures, filtered_out, meta`.

## Python API
```python
from resume_rag import ResumeRAG
from job_matcher import JobMatcher
rag = ResumeRAG("chroma_db", backend="auto"); rag.ingest_directory("data/resumes", reset=True)
result = JobMatcher(rag).match(open("data/job_descriptions/jd_01_senior_ml_engineer.txt").read(), top_k=10,
                               must_haves={"skills": {"Python": 5}})
```

## Demo video script (3-4 min)
1. **0:00-0:30 Problem & architecture.** Show the diagram above; say what RAG means here: retrieve resume chunks, then rank and explain.
2. **0:30-1:15 Part A.** `ls data/resumes` (PDF + TXT mix); `python resume_rag.py inspect data/resumes/01_*.pdf` shows sections, chunks and extracted metadata; run `ingest --reset` and show chunk / resume counts.
3. **1:15-2:30 Part B.** `python job_matcher.py --jd data/job_descriptions/jd_01_senior_ml_engineer.txt --top-k 5`: point at `parsed_requirements.must_haves`, `match_score`, `matched_skills`, `relevant_excerpts`, `reasoning`, and `filtered_out` (who was removed and why). Then re-run with `--no-filter` and show juniors reappearing; try another JD (devops or product manager).
4. **2:30-3:30 Metrics.** Open the notebook: embedding PCA plot, mode comparison table (semantic vs keyword vs hybrid vs hybrid+filters), latency / scaling chart.
5. **3:30-4:00 Limitations.** Synthetic data, rule-based JD parsing, relative scores, bias / human-in-the-loop.

## Known limitations
Synthetic, clean resumes (real ones need layout handling / OCR); heuristic parsing of "N+ years" and "A or B"; fixed skill taxonomy; scores relative to the candidate pool; ground truth made by the same generator as the resumes. Hiring is high-stakes: audit for bias and keep a human in the loop.
