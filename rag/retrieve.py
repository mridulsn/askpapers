"""Hybrid retrieval: BM25 + dense vectors, fused with Reciprocal Rank Fusion, then reranked.

    keyword (BM25) top 50 --+
                            +-- RRF fuse -- top 30 -- cross-encoder rerank -- top 5
    vector (bge)   top 50 --+
"""
import json
import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from .index import CHUNKS, EMB
from .models import QUERY_PREFIX, embedder, reranker

STOP = set("""a an and are as at be by for from has have in into is it its of on or that the
this to was were what which who will with how why when does do can we our their they these
those than then there via""".split())

MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")


def tokenize(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", t.lower()) if w not in STOP]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank).

    Uses only ranks, so BM25 scores (unbounded) and cosine scores (0-1) never
    have to be put on the same scale.
    """
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking, 1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r)
    return sorted(score.items(), key=lambda x: -x[1])


@dataclass
class Result:
    final: list[dict]
    stages: dict = field(default_factory=dict)   # stage name -> ranked chunk ids


class Retriever:
    def __init__(self, web: bool = False) -> None:
        """web=True mimics the public web demo: passages from papers whose licence doesn't
        allow redistribution are reranked on a 40-word excerpt (all the page can hold)."""
        self.chunks = [json.loads(l) for l in CHUNKS.open()]
        self.emb = np.load(EMB)
        self.bm25 = BM25Okapi([tokenize(c["search_text"]) for c in self.chunks])
        self.rerank_text = [c["search_text"] for c in self.chunks]
        self.fixed: set[int] = set()   # web mode: excerpt-only passages keep their fused slot
        if web:
            lic = json.loads((CHUNKS.parent / "licenses.json").read_text())
            for i, c in enumerate(self.chunks):
                if c["page"] and "creativecommons.org" not in lic.get(c["paper"], ""):
                    self.fixed.add(i)

    def bm25_rank(self, q: str, k: int) -> list[int]:
        s = self.bm25.get_scores(tokenize(q))
        return [int(i) for i in np.argsort(-s, kind="stable")[:k] if s[i] > 0]

    def dense_rank(self, q: str, k: int) -> list[int]:
        v = embedder().encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
        return [int(i) for i in np.argsort(-(self.emb @ v), kind="stable")[:k]]

    def rerank(self, q: str, ids: list[int]) -> list[tuple[int, float]]:
        """Reorder ids by cross-encoder score. Passages in self.fixed (web mode: only an
        excerpt is public, so the reranker can't judge them fairly) stay in their slot and
        the others are reordered around them."""
        movable = [i for i in ids if i not in self.fixed]
        scores = reranker().predict([(q, self.rerank_text[i]) for i in movable],
                                    batch_size=32) if movable else []
        ranked = iter(sorted(zip(movable, map(float, scores)), key=lambda x: -x[1]))
        return [(i, 0.0) if i in self.fixed else next(ranked) for i in ids]

    def search(self, q: str, mode: str = "hybrid+rerank", k: int = 5,
               pool: int = 50, rerank_pool: int = 30) -> Result:
        stages: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            stages["bm25"] = self.bm25_rank(q, pool)
        if mode in ("dense", "hybrid", "hybrid+rerank"):
            stages["dense"] = self.dense_rank(q, pool)
        if mode == "bm25":
            ranked = [(i, 0.0) for i in stages["bm25"]]
        elif mode == "dense":
            ranked = [(i, 0.0) for i in stages["dense"]]
        else:
            ranked = rrf([stages["bm25"], stages["dense"]])
            stages["hybrid"] = [i for i, _ in ranked]
            if mode == "hybrid+rerank":
                ranked = self.rerank(q, stages["hybrid"][:rerank_pool])
                stages["rerank"] = [i for i, _ in ranked]
        final = [{**self.chunks[i], "score": s} for i, s in ranked[:k]]
        return Result(final=final, stages=stages)
