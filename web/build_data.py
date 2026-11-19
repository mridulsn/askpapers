"""Export the index for the in-browser demo (web/data/).

Usage: uv run python web/build_data.py [--record]
  --record  also record real answers for the example questions through the local
            Claude Code CLI (no API key needed), saved to web/data/answers.json
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.fetch import META  # noqa: E402
from rag.index import CHUNKS, EMB  # noqa: E402

OUT = ROOT / "web" / "data"

EXAMPLES = [
    "How does FLARE decide when to retrieve?",
    "How can you evaluate a RAG system without human-written reference answers?",
    "What happens when the retrieved documents are wrong or irrelevant?",
    "How can an attacker poison the documents a RAG system searches?",
    "Do knowledge graphs help retrieval for multi-hop questions?",
    "How should long documents be split into chunks?",
]


def lic_name(url: str) -> str:
    if "creativecommons.org/licenses/" in url:
        return "CC " + url.split("/licenses/")[1].strip("/").replace("/", " ").upper()
    return "arXiv licence (excerpts only)"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    papers = json.loads(META.read_text())
    pidx = {p["id"]: i for i, p in enumerate(papers)}
    chunks = [json.loads(l) for l in CHUNKS.open()]
    lic_all = json.loads((ROOT / "data" / "licenses.json").read_text())
    (OUT / "papers.json").write_text(json.dumps(
        [{"id": p["id"], "title": p["title"], "authors": p["authors"][:3],
          "more": max(len(p["authors"]) - 3, 0), "date": p["published"],
          "license": lic_name(lic_all.get(p["id"], ""))} for p in papers],
        separators=(",", ":")))
    # Licences: CC-licensed papers are shown in full (with attribution). Papers under arXiv's
    # non-exclusive licence don't allow redistribution, so for those the page gets only a short
    # excerpt to display plus pre-counted words for keyword search (same ranking, no full text).
    # Abstracts are arXiv metadata (CC0), so they are always shown.
    lic = json.loads((ROOT / "data" / "licenses.json").read_text())
    from rag.retrieve import tokenize
    rows, n_short = [], 0
    for c in chunks:
        open_lic = "creativecommons.org" in lic.get(c["paper"], "")
        if open_lic or c["page"] == 0:
            rows.append([pidx[c["paper"]], c["page"], c["text"]])
        else:
            n_short += 1
            words = c["text"].split()
            tf = {}
            for w in tokenize(c["search_text"]):
                tf[w] = tf.get(w, 0) + 1
            rows.append([pidx[c["paper"]], c["page"], " ".join(words[:40]) + " ...", tf])
    (OUT / "chunks.json").write_text(json.dumps(rows, separators=(",", ":")))
    print(f"{n_short} passages from non-CC papers shipped as excerpt + word counts")
    emb = np.load(EMB)
    q = np.clip(np.round(emb * 127), -127, 127).astype(np.int8)   # 4x smaller, ~no loss
    q.tofile(OUT / "emb.i8")
    err = np.abs(q.astype(np.float32) / 127 - emb).max()
    print(f"{len(papers)} papers, {len(chunks)} chunks, emb {q.shape} int8 (max abs err {err:.4f})")

    # The page shows numbers measured on the page's own pipeline (tests/web_eval.mjs runs
    # web/search.js with the same int8 vectors and q8 models under Node).
    res = ROOT / "eval" / "results-browser.json"
    if res.exists():
        r = json.loads(res.read_text())["sets"]
        (OUT / "eval.json").write_text(json.dumps(
            {k: {"n": v["n_questions"], "modes": v["modes"]} for k, v in r.items()}, indent=1))
    else:
        print("eval/results-browser.json missing: run node tests/web_eval.mjs, then rebuild")

    if "--record" in sys.argv:
        from rag.generate import answer_cli
        from rag.retrieve import Retriever
        r = Retriever()
        rec = []
        for qn in EXAMPLES:
            hits = r.search(qn, "hybrid+rerank", k=5).final
            a = answer_cli(qn, hits, model="sonnet")
            rec.append({"question": qn, "answer": a["answer"], "model": a["model"],
                        "via": a["via"], "date": date.today().isoformat(),
                        "sources": [c["id"] for c in hits]})
            print("recorded:", qn)
        (OUT / "answers.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
