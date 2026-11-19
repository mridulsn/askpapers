"""Turn the downloaded PDFs into searchable chunks + embeddings.

Usage: uv run python -m rag.index
Writes data/chunks.jsonl and data/embeddings.npy.
"""
import json
import re

import numpy as np
import pymupdf

from .fetch import DATA, META, PAPERS
from .models import EMBED_MODEL, embedder

CHUNKS = DATA / "chunks.jsonl"
EMB = DATA / "embeddings.npy"
WORDS, OVERLAP = 250, 50
REF_HEAD = re.compile(r"^\s*(references|bibliography|reference)\s*$", re.I | re.M)


def clean(t: str) -> str:
    t = re.sub(r"-\n(\w)", r"\1", t)          # re-join words hyphenated across lines
    return " ".join(t.split())


def paper_pages(path) -> list[tuple[int, str]]:
    """Text per page, stopping at the reference list (citations are noise for search)."""
    out = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, 1):
            raw = page.get_text()
            m = REF_HEAD.search(raw)
            if m and i > 2:
                out.append((i, clean(raw[: m.start()])))
                break
            out.append((i, clean(raw)))
    return [(p, t) for p, t in out if t]


def chunk_paper(p: dict) -> list[dict]:
    head = f"{p['title']}. "
    chunks = [{"paper": p["id"], "page": 0, "section": "abstract",
               "text": p["abstract"]}]
    # Flatten words but remember which page each word came from.
    words, pages = [], []
    for pg, txt in paper_pages(PAPERS / f"{p['id']}.pdf"):
        w = txt.split()
        words += w
        pages += [pg] * len(w)
    step = WORDS - OVERLAP
    for s in range(0, max(len(words) - OVERLAP, 1), step):
        piece = words[s: s + WORDS]
        if len(piece) < 40:
            continue
        chunks.append({"paper": p["id"], "page": pages[s], "section": "body",
                       "text": " ".join(piece)})
    for c in chunks:
        c["title"] = p["title"]
        # Prefixing the title gives each chunk context it lost when it was cut out.
        c["search_text"] = head + c["text"]
    return chunks


def main() -> None:
    papers = json.loads(META.read_text())
    chunks = []
    for p in papers:
        try:
            chunks += chunk_paper(p)
        except Exception as ex:
            print(f"skip {p['id']}: {ex}")
    for i, c in enumerate(chunks):
        c["id"] = i
    with CHUNKS.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"{len(papers)} papers -> {len(chunks)} chunks; embedding with {EMBED_MODEL} ...")
    emb = embedder().encode([c["search_text"] for c in chunks], batch_size=64,
                            normalize_embeddings=True, show_progress_bar=True)
    np.save(EMB, emb.astype(np.float32))
    print(f"saved {CHUNKS.name} and {EMB.name} {emb.shape}")


if __name__ == "__main__":
    main()
