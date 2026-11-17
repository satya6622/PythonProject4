"""
evaluate.py -- performance metrics for the resume RAG system
============================================================
Retrieval accuracy (vs. data/ground_truth.json), latency, and extraction accuracy.

Relevance grades per (JD, resume):
    2 = right archetype AND truly satisfies the JD's must-haves   (binary "relevant")
    1 = right archetype but fails must-haves, OR secondary archetype that satisfies them
    0 = otherwise
Metrics @K=10: Precision@5, Precision@10, Recall@10, MRR, nDCG@10 (graded), Hit@1.

    python evaluate.py --backend auto --out results
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

import numpy as np

from job_matcher import JobMatcher, Weights, parse_job_description
from resume_rag import (ResumeRAG, extract_metadata, load_resume_text, split_sections)

MODES = {
    "semantic only":      dict(weights=Weights(1, 0, 0),         apply_filters=False),
    "keyword only":       dict(weights=Weights(0, 1, 0),         apply_filters=False),
    "hybrid":             dict(weights=Weights(0.55, 0.35, 0.10), apply_filters=False),
    "hybrid + filters":   dict(weights=Weights(0.55, 0.35, 0.10), apply_filters=True),
}


# ------------------------------- ground truth -------------------------------- #
def load_truth(data_dir: str) -> dict:
    return json.loads((Path(data_dir) / "ground_truth.json").read_text())


def meets_must_haves(t: dict, must: dict) -> bool:
    for s, y in (must.get("skills") or {}).items():
        if t["skill_years"].get(s, 0.0) < y:
            return False
    if t["total_years"] < must.get("min_total_years", 0):
        return False
    return t["education_level"] >= must.get("min_education", 0)


def grade(t: dict, jd: dict) -> int:
    ok = meets_must_haves(t, jd["must_haves"])
    if t["archetype"] in jd["primary"]:
        return 2 if ok else 1
    if t["archetype"] in jd["secondary"] and ok:
        return 1
    return 0


# --------------------------------- metrics ----------------------------------- #
def ranking_metrics(ranked_paths: list[str], grades: dict[str, int], k: int = 10) -> dict:
    g = [grades.get(Path(p).name, 0) for p in ranked_paths[:k]]
    n_rel = sum(1 for v in grades.values() if v == 2)
    rel_flags = [1 if x == 2 else 0 for x in g]
    dcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(g))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(ideal)) or 1.0
    first = next((i for i, f in enumerate(rel_flags) if f), None)
    # binary nDCG (only grade-2 counts) and R-precision (precision at R = #relevant)
    bdcg = sum(f / math.log2(i + 2) for i, f in enumerate(rel_flags))
    bideal = sum(1 / math.log2(i + 2) for i in range(min(n_rel, k))) or 1.0
    r_ranked = [1 if grades.get(Path(p).name, 0) == 2 else 0 for p in ranked_paths[:n_rel]]
    return {
        "P@5": sum(rel_flags[:5]) / 5,
        "P@10": sum(rel_flags[:10]) / 10,
        "R@10": (sum(rel_flags) / n_rel) if n_rel else float("nan"),
        "MRR": 1 / (first + 1) if first is not None else 0.0,
        "R-Prec": (sum(r_ranked) / n_rel) if n_rel else float("nan"),
        "nDCG@10": dcg / idcg,
        "nDCG-bin@10": bdcg / bideal,
        "Hit@1": float(bool(rel_flags and rel_flags[0])),
        "n_relevant": n_rel,
    }


def mean_metrics(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k != "n_relevant"]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


# ------------------------------- evaluations --------------------------------- #
def evaluate_retrieval(matcher: JobMatcher, data_dir: str, k: int = 10) -> dict:
    truth = load_truth(data_dir)
    out: dict = {}
    for mode, cfg in MODES.items():
        per_jd = {}
        for slug, jd in truth["job_descriptions"].items():
            text = (Path(data_dir) / "job_descriptions" / f"{slug}.txt").read_text()
            grades = {fn: grade(t, jd) for fn, t in truth["resumes"].items()}
            res = matcher.match(text, top_k=k, weights=cfg["weights"], apply_filters=cfg["apply_filters"])
            per_jd[slug] = ranking_metrics([m["resume_path"] for m in res["top_matches"]], grades, k)
        out[mode] = {"mean": mean_metrics(list(per_jd.values())), "per_jd": per_jd}
    return out


def evaluate_latency(matcher: JobMatcher, data_dir: str, repeats: int = 15) -> dict:
    truth = load_truth(data_dir)
    texts = [(Path(data_dir) / "job_descriptions" / f"{s}.txt").read_text() for s in truth["job_descriptions"]]
    matcher.match(texts[0])  # warm-up (model load, chroma caches)
    e2e, embed = [], []
    for _ in range(repeats):
        for t in texts:
            t0 = time.perf_counter()
            matcher.match(t, top_k=10)
            e2e.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            matcher.rag.embedder.embed([t])
            embed.append((time.perf_counter() - t0) * 1000)
    pct = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    return {
        "queries": len(e2e),
        "end_to_end_ms": {"mean": float(np.mean(e2e)), "p50": pct(e2e, 50), "p95": pct(e2e, 95), "max": max(e2e)},
        "jd_embedding_ms": {"mean": float(np.mean(embed)), "p95": pct(embed, 95)},
    }


def evaluate_extraction(data_dir: str) -> dict:
    truth = load_truth(data_dir)["resumes"]
    name_ok = edu_ok = 0
    yr_err, f1s, skill_year_err = [], [], []
    for fn, t in truth.items():
        text = load_resume_text(Path(data_dir) / "resumes" / fn)
        h, secs = split_sections(text)
        m = extract_metadata(h, secs, text)
        name_ok += m.name.lower() == t["name"].lower()
        edu_ok += m.education_level == t["education_level"]
        yr_err.append(abs(m.experience_years - t["total_years"]))
        p, g = set(m.skills), set(t["skills"])
        tp = len(p & g)
        prec, rec = tp / max(len(p), 1), tp / max(len(g), 1)
        f1s.append(2 * prec * rec / (prec + rec) if tp else 0.0)
        for s, y in t["skill_years"].items():
            if s in m.skill_years:
                skill_year_err.append(abs(m.skill_years[s] - y))
    n = len(truth)
    return {
        "resumes": n,
        "name_accuracy": name_ok / n,
        "education_level_accuracy": edu_ok / n,
        "skills_f1_mean": float(np.mean(f1s)),
        "experience_years_MAE": float(np.mean(yr_err)),
        "experience_within_1yr": float(np.mean(np.array(yr_err) <= 1.0)),
        "skill_years_MAE": float(np.mean(skill_year_err)),
    }


def evaluate_jd_parsing(data_dir: str) -> dict:
    truth = load_truth(data_dir)["job_descriptions"]
    ok = 0
    for slug, jd in truth.items():
        sp = parse_job_description((Path(data_dir) / "job_descriptions" / f"{slug}.txt").read_text())
        exp = jd["must_haves"]
        ok += (sp.skill_min_years == {k: float(v) for k, v in exp.get("skills", {}).items()}
               and sp.min_education == exp.get("min_education", 0)
               and sp.min_total_years == exp.get("min_total_years", 0))
    return {"jds": len(truth), "must_have_parse_accuracy": ok / len(truth)}


def evaluate_filters(matcher: JobMatcher, data_dir: str) -> dict:
    """How well do hard must-have filters agree with ground-truth must-have satisfaction?"""
    truth = load_truth(data_dir)
    tp = fp = fn_ = 0
    for slug, jd in truth["job_descriptions"].items():
        text = (Path(data_dir) / "job_descriptions" / f"{slug}.txt").read_text()
        res = matcher.match(text, top_k=len(truth["resumes"]), apply_filters=True)
        passed = {Path(m["resume_path"]).name for m in res["top_matches"]}
        for fnm, t in truth["resumes"].items():
            should = meets_must_haves(t, jd["must_haves"])
            tp += should and fnm in passed
            fp += (not should) and fnm in passed
            fn_ += should and fnm not in passed
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn_, 1)
    return {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn_}


def run_all(db: str = "chroma_db", backend: str = "auto", data_dir: str = "data",
            reset: bool = True) -> dict:
    rag = ResumeRAG(db, backend)
    ingest = rag.ingest_directory(f"{data_dir}/resumes", reset=reset)
    matcher = JobMatcher(rag)
    return {
        "embedder": rag.embedder.name,
        "ingestion": ingest,
        "retrieval": evaluate_retrieval(matcher, data_dir),
        "latency": evaluate_latency(matcher, data_dir),
        "extraction": evaluate_extraction(data_dir),
        "jd_parsing": evaluate_jd_parsing(data_dir),
        "must_have_filter": evaluate_filters(matcher, data_dir),
    }


def make_plots(report: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(report["retrieval"])
    metrics = ["R-Prec", "MRR", "nDCG-bin@10", "Hit@1"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    for i, m in enumerate(metrics):
        ax.bar(np.arange(len(modes)) + i * w, [report["retrieval"][md]["mean"][m] for md in modes], w, label=m)
    ax.set_xticks(np.arange(len(modes)) + 1.5 * w)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Retrieval quality by search mode  (embedder: {report['embedder']})")
    ax.legend(ncol=4, loc="lower right")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "retrieval_by_mode.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    report = run_all(a.db, a.backend, a.data)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    make_plots(report, out)

    print(f"\nEmbedder: {report['embedder']}   ingestion: {report['ingestion']}")
    print("\nRetrieval (mean over 8 JDs, K=10)")
    print(f"{'mode':<18}" + "".join(f"{m:>12}" for m in ["P@5", "R-Prec", "R@10", "MRR", "nDCG@10", "nDCG-bin@10", "Hit@1"]))
    for mode, r in report["retrieval"].items():
        print(f"{mode:<18}" + "".join(f"{r['mean'][m]:>12.3f}" for m in ["P@5", "R-Prec", "R@10", "MRR", "nDCG@10", "nDCG-bin@10", "Hit@1"]))
    print("\nLatency:", json.dumps(report["latency"], indent=1))
    print("Extraction:", json.dumps(report["extraction"], indent=1))
    print("JD parsing:", report["jd_parsing"], "\nMust-have filter:", report["must_have_filter"])


if __name__ == "__main__":
    main()
