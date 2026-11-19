"""Local models (downloaded once from Hugging Face, then cached)."""
from functools import lru_cache

import torch

import os

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "mixedbread-ai/mxbai-rerank-xsmall-v1")
# bge models expect this instruction in front of a search query (not in front of documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def device() -> str:
    # MPS hung on the reranker under load; CPU is reliable and fast enough here.
    return os.environ.get("RAG_DEVICE", "cpu")


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device=device())


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    # float32 on purpose: some rerankers ship float16 weights, which are slow and flaky off-GPU.
    return CrossEncoder(RERANK_MODEL, device=os.environ.get("RERANK_DEVICE", device()),
                        max_length=512, model_kwargs={"torch_dtype": torch.float32})
