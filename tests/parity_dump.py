"""Dump Python rankings for every eval question (input for tests/parity.mjs)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.retrieve import Retriever, rrf  # noqa: E402

r = Retriever()
qs = [json.loads(l)["q"] for f in ("eval/questions.jsonl", "eval/questions_reworded.jsonl") for l in open(f)]
out = []
for q in qs + ["FLARE", "Ragas faithfulness", "x"]:
    b = r.bm25_rank(q, 50)
    d = r.dense_rank(q, 50)
    out.append({"q": q, "bm25": b, "dense50": d, "hybrid": [i for i, _ in rrf([b, d])][:10]})
print(json.dumps(out))
