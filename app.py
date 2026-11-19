"""Streamlit demo: ask a question about the arXiv RAG papers.

Run: uv run streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st

from rag.generate import available, check_citations, stream_answer
from rag.retrieve import MODES, Retriever

st.set_page_config(page_title="Ask the RAG papers", layout="wide")

st.markdown("""
<style>
:root { --ink:#1d2433; --muted:#5b6474; --line:#d9dee7; --accent:#2f6fde; }
.block-container { max-width: 1100px; padding-top: 2rem; }
p, li { line-height: 1.6; }
.src { border-top:1px solid var(--line); padding:12px 0; }
.src h4 { margin:0 0 4px 0; font-size:1rem; }
.meta { color:var(--muted); font-size:.85rem; }
.rank { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading search index and models ...")
def load() -> Retriever:
    r = Retriever()
    r.search("warm up", "hybrid+rerank")      # load both models once
    return r


r = load()
n_papers = len({c["paper"] for c in r.chunks})

st.title("Ask the RAG papers")
st.caption(f"{n_papers} arXiv papers on retrieval-augmented generation, "
           f"{len(r.chunks):,} passages. Keyword + vector search, merged and reranked, "
           "then answered with citations.")

with st.sidebar:
    st.subheader("Search settings")
    mode = st.radio("Pipeline", MODES, index=3,
                    help="Switch modes to compare what each step adds.")
    k = st.slider("Passages sent to the answer", 3, 10, 5)
    st.divider()
    res_file = Path("eval/results.json")
    if res_file.exists():
        st.subheader("Measured on the test set")
        rows = json.loads(res_file.read_text())["modes"]
        st.table({m: {"Hit@5": f"{v['hit@5']:.2f}", "MRR@10": f"{v['mrr@10']:.2f}"}
                  for m, v in rows.items()})

examples = ["How does Self-RAG decide when to retrieve?",
            "What are common ways to evaluate RAG systems?",
            "Why does hybrid search beat pure vector search?"]
q = st.text_input("Your question", placeholder=examples[0])
cols = st.columns(len(examples))
for c, ex in zip(cols, examples):
    if c.button(ex, use_container_width=True):
        q = ex

if q:
    res = r.search(q, mode, k=k)
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Answer")
        stream = stream_answer(q, res.final)
        if stream is None:
            st.info("No Anthropic API key set, so this shows the retrieved passages only. "
                    "Set ANTHROPIC_API_KEY to get a written answer with citations.")
        else:
            answer = st.write_stream(stream)
            cit = check_citations(answer, len(res.final))
            if cit["invalid"]:
                st.warning(f"Answer cited sources that don't exist: {cit['invalid']}")
            elif not cit["cited"]:
                st.caption("No sources cited.")

    with right:
        st.subheader("Sources")
        for n, c in enumerate(res.final, 1):
            where = "abstract" if c["page"] == 0 else f"p. {c['page']}"
            url = f"https://arxiv.org/abs/{c['paper']}"
            st.markdown(
                f"<div class='src'><h4>[{n}] <a href='{url}' target='_blank'>{c['title']}</a></h4>"
                f"<div class='meta'>arXiv:{c['paper']} &middot; {where}</div></div>",
                unsafe_allow_html=True)
            with st.expander("Passage"):
                st.write(c["text"])

    with st.expander("How the search ranked these passages"):
        st.caption("Rank of each final passage in every stage (blank = not in that stage's top 50). "
                   "Reranking often lifts passages that neither search ranked first.")
        rows = []
        for n, c in enumerate(res.final, 1):
            row = {"Source": f"[{n}] {c['title'][:60]}"}
            for stage, ids in res.stages.items():
                row[stage] = ids.index(c["id"]) + 1 if c["id"] in ids else None
            rows.append(row)
        st.dataframe(rows, hide_index=True, use_container_width=True)
