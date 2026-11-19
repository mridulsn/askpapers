"""Download a corpus of arXiv papers (metadata + PDF).

Usage: uv run python -m rag.fetch --query 'all:"retrieval augmented generation"' --n 60

arXiv asks API clients to wait ~3 s between requests, so this is slow on purpose.
"""
import argparse
import json
import time
from pathlib import Path

import feedparser
import requests

API = "https://export.arxiv.org/api/query"
DATA = Path(__file__).resolve().parent.parent / "data"
PAPERS = DATA / "papers"
META = DATA / "papers.json"
HEADERS = {"User-Agent": "hybrid-rag-arxiv/0.1 (research demo)"}


def search(query: str, n: int) -> list[dict]:
    r = requests.get(
        API,
        params={"search_query": query, "start": 0, "max_results": n,
                "sortBy": "relevance", "sortOrder": "descending"},
        headers=HEADERS, timeout=60,
    )
    r.raise_for_status()
    out = []
    for e in feedparser.parse(r.text).entries:
        aid = e.id.rsplit("/abs/", 1)[-1]          # e.g. 2312.10997v5
        base = aid.split("v")[0] if "v" in aid else aid
        out.append({
            "id": base,
            "version": aid,
            "title": " ".join(e.title.split()),
            "authors": [a.name for a in e.authors],
            "published": e.published[:10],
            "abstract": " ".join(e.summary.split()),
            "url": f"https://arxiv.org/abs/{base}",
            "pdf": f"https://arxiv.org/pdf/{aid}",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default='all:"retrieval augmented generation"')
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--delay", type=float, default=3.0)
    a = ap.parse_args()

    PAPERS.mkdir(parents=True, exist_ok=True)
    papers = search(a.query, a.n)
    print(f"found {len(papers)} papers")
    kept = []
    for i, p in enumerate(papers, 1):
        dest = PAPERS / f"{p['id']}.pdf"
        if not dest.exists():
            time.sleep(a.delay)
            try:
                r = requests.get(p["pdf"], headers=HEADERS, timeout=120)
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    raise ValueError("not a PDF")
                dest.write_bytes(r.content)
            except Exception as ex:  # keep going; report what failed
                print(f"  [{i}] FAILED {p['id']}: {ex}")
                continue
        kept.append(p)
        print(f"  [{i}/{len(papers)}] {p['id']}  {p['title'][:70]}")
    META.write_text(json.dumps(kept, indent=1))
    print(f"saved {len(kept)} papers -> {META}")


if __name__ == "__main__":
    main()
