"""Measure retrieval quality for each pipeline mode.

Usage: uv run python -m eval.run_eval
Reads eval/questions.jsonl ({"q": ..., "paper": "<arXiv id>"}) and writes
eval/results.json + eval/results.md.

Metrics are at the paper level: a hit means a passage from the paper the
question was written from is in the top results.
  Hit@5  - share of questions where the right paper is in the top 5 passages
  MRR@10 - average of 1/rank of the first right passage (0 if not in top 10)
"""
import json
import time
from pathlib import Path

from rag.retrieve import MODES, Retriever

HERE = Path(__file__).resolve().parent


SETS = {"keyword": "questions.jsonl", "reworded": "questions_reworded.jsonl"}


def main() -> None:
    """Usage: uv run python -m eval.run_eval [--name NAME]
    Writes eval/results-NAME.json and .md (NAME defaults to the reranker model's short name).
    Env: RERANK_MODEL picks the reranker; WEB_SIM=1 applies the web page's licence limits."""
    import argparse
    import os
    from rag.models import RERANK_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=RERANK_MODEL.split("/")[-1])
    name = ap.parse_args().name
    web = os.environ.get("WEB_SIM") == "1"
    r = Retriever(web=web)
    r.search("warm up", "hybrid+rerank")
    report = {"config": {"reranker": RERANK_MODEL, "web_sim": web},
              "sets": {s: run_set(r, s, f) for s, f in SETS.items()}}
    (HERE / f"results-{name}.json").write_text(json.dumps(report, indent=1))
    write_md(report, name)
    print(f"wrote eval/results-{name}.json/.md")


def run_set(r: Retriever, name: str, fname: str) -> dict:
    qs = [json.loads(l) for l in (HERE / fname).open() if l.strip()]
    print(f"--- {name} questions ({len(qs)})")
    out = {"n_questions": len(qs), "modes": {}, "per_question": []}
    per_q = {i: {"q": q["q"], "paper": q["paper"]} for i, q in enumerate(qs)}
    for mode in MODES:
        hits, rr, t0 = 0, 0.0, time.time()
        for i, q in enumerate(qs):
            papers = [c["paper"] for c in r.search(q["q"], mode, k=10).final]
            rank = papers.index(q["paper"]) + 1 if q["paper"] in papers else None
            hits += bool(rank and rank <= 5)
            rr += 1 / rank if rank else 0
            per_q[i][mode] = rank
        out["modes"][mode] = {"hit@5": hits / len(qs), "mrr@10": rr / len(qs),
                              "ms_per_query": 1000 * (time.time() - t0) / len(qs)}
        m = out["modes"][mode]
        print(f"{mode:14s} Hit@5 {m['hit@5']:.3f}  MRR@10 {m['mrr@10']:.3f}  "
              f"{m['ms_per_query']:.0f} ms/query")
    out["per_question"] = list(per_q.values())
    return out


def write_md(report: dict, run_name: str) -> None:
    c = report["config"]
    lines = [f"# Retrieval results: {run_name}", "",
             f"Reranker: `{c['reranker']}`. Web licence limits applied: {c['web_sim']}.", "",
             "Rank shown = position of the first passage from the right paper (- = not in top 10).", ""]
    for name, out in report["sets"].items():
        lines += [f"## {name.capitalize()} questions ({out['n_questions']})", "",
                  "| Pipeline | Hit@5 | MRR@10 | ms / query |", "|---|---|---|---|"]
        for mode, m in out["modes"].items():
            lines.append(f"| {mode} | {m['hit@5']:.3f} | {m['mrr@10']:.3f} | {m['ms_per_query']:.0f} |")
        lines += ["", "Questions where the pipelines disagree:", "",
                  "| Question | " + " | ".join(MODES) + " |", "|---|" + "---|" * len(MODES)]
        for p in out["per_question"]:
            ranks = [p[m] for m in MODES]
            if len(set(ranks)) > 1:
                cells = " | ".join("-" if x is None else str(x) for x in ranks)
                lines.append(f"| {p['q']} | {cells} |")
        lines.append("")
    (HERE / f"results-{run_name}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
