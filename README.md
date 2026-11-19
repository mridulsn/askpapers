# Ask the RAG papers: hybrid RAG with reranking

Search 60 arXiv papers on retrieval-augmented generation and get answers that cite their sources.
Keyword search and vector search run side by side, their results are merged with Reciprocal Rank
Fusion, a cross-encoder reranks the top 30, and Claude answers from the best 5 with numbered citations.

**Live demo:** https://mridulsaravanan.com/askpapers/ (runs fully in the browser, no server, no API key)

```
question
   |
   +--> BM25 keyword search ------ top 50 --+
   |                                         +--> Reciprocal Rank Fusion --> top 30
   +--> bge-small vector search -- top 50 --+                                  |
                                                                               v
                                         cross-encoder reranker (mxbai-rerank-xsmall)
                                                                               |
                                                                            top 5
                                                                               |
                                              Claude answers with [n] citations, or says
                                              "The retrieved papers don't answer this."
```

## Results

Two hand-written test sets, scored at the paper level: a hit means a passage from the paper the
question was written from is in the top 5. MRR@10 rewards putting it first.

- **Keyword-style (42 questions):** each question uses words from the paper, like method names.
- **Reworded (41 questions):** the same kind of questions, deliberately written without the
  papers' own terms ("Where does the name of a town come from?" for a paper on place-name origins).

| Pipeline | Keyword-style Hit@5 | MRR@10 | Reworded Hit@5 | MRR@10 |
|---|---|---|---|---|
| BM25 only | 97.6% | 0.952 | 75.6% | 0.578 |
| Vector only (bge-small) | 88.1% | 0.864 | 90.2% | 0.732 |
| Hybrid (RRF) | 95.2% | 0.960 | 87.8% | 0.739 |
| Hybrid + rerank, **mxbai-rerank-xsmall** (default) | **100%** | **1.000** | **92.7%** | **0.797** |
| Hybrid + rerank, bge-reranker-base | 100% | 0.984 | 87.8% | 0.774 |
| Hybrid + rerank, ms-marco-MiniLM-L-6 | 97.6% | 0.945 | 85.4% | 0.715 |

What this shows:
- **Each retriever fails where the other is strong.** BM25 is near perfect when the question shares
  the paper's words and drops to 76% when it doesn't. Vectors handle rewording (90%) but miss exact names.
- **Fusion alone is not a free win.** RRF lands between the two on each set, because it also pulls in
  the weaker list's mistakes. Its value is being robust across both kinds of question.
- **The reranker is what makes hybrid pay off.** It reads the question and passage together, and lifts
  the reworded set from 87.8% to 92.7%. The small mxbai model beat the 4x larger bge-reranker-base here,
  so model size alone didn't decide it. The MiniLM reranker (trained on web search queries) made things worse.
- Latency on an Apple M4 CPU, measured while other jobs shared the machine: BM25 3 ms, vectors 10 ms,
  reranking 30 passages about 5 s with mxbai (0.5 s with MiniLM). Reranking is the cost of the accuracy.

**The browser version has an extra constraint** (see Licences below): for 24 papers the page may only
hold a 40-word excerpt. Reranking those on the excerpt alone hurt (keyword-style Hit@5 fell to 95.2%,
reworded to 85.4%), because the reranker can't judge a passage from 40 words. The fix: excerpt-only
passages keep their fused slot and the reranker reorders only the full-text passages around them.
Measured under the page's real limits (`WEB_SIM=1 uv run python -m eval.run_eval`):

| Browser pipeline | Keyword-style Hit@5 | MRR@10 | Reworded Hit@5 | MRR@10 |
|---|---|---|---|---|
| Hybrid + rerank, excerpts reranked | 95.2% | 0.921 | 85.4% | 0.807 |
| Hybrid + rerank, excerpts keep their slot (shipped) | 97.6% | 0.961 | 97.6% | 0.797 |

The shipped browser version scores higher on reworded Hit@5 than the full-text pipeline (97.6% vs
92.7%). That is 2 questions out of 41, inside the noise of a set this small, and the MRR is the same,
so I read it as "no loss from the licence limit", not as an improvement.

**Limitations, plainly:** I wrote both question sets myself from passages I had read, so they test
retrieval of the right paper, not answer quality. 42 and 41 questions give wide error bars (one
question = 2.4 points). Answer quality and citation accuracy are tested separately in
[evalbench](https://github.com/mridulsn/evalbench), which runs LLM-as-judge on this project's answers.

## Run it

```bash
uv sync
uv run python -m rag.fetch --n 60      # download papers (about 3 minutes, arXiv asks for a pause between requests)
uv run python -m rag.index             # chunk + embed (about 1 minute)
uv run python -m eval.run_eval         # the table above
export ANTHROPIC_API_KEY=...           # optional, for written answers
uv run streamlit run app.py            # local app with streamed, cited answers
```

Without an API key the app still searches and shows the passages. `uv run pytest` runs the unit tests,
and `node tests/parity.mjs` checks that the browser BM25 and fusion give the same rankings as Python.

## Design decisions

- **Chunks of 250 words with 50 overlap, each prefixed with the paper title.** Without the title, a
  passage like "our method improves recall by 4 points" has no idea which method it is talking about.
  References sections are cut off before chunking, because citation lists match almost any query.
- **Fusion by rank, not score.** BM25 scores are unbounded and cosine scores sit between 0 and 1.
  RRF (`1 / (60 + rank)`) avoids having to tune weights to put them on one scale.
- **Rerank only the top 30.** A cross-encoder is accurate but slow, because it reads each passage
  together with the question. Running it on 30 fused candidates keeps it near one second.
- **Answers are grounded or they refuse.** The prompt allows only the numbered sources, requires a
  citation on every factual sentence, and gives an exact refusal sentence when the sources don't cover
  the question. The app checks that cited numbers exist. It uses `claude-opus-5-5` through the official
  SDK with streaming, at low effort (grounded Q&A doesn't need long reasoning), with server-side refusal
  fallback switched on.

## The web demo

`web/` is a static page that reproduces the pipeline in the browser: the same BM25 (ported to JS and
checked against Python), the same int8-quantised embeddings, bge-small and the mxbai reranker running
on transformers.js. It needs no server, so the written-answer step works like this:
- **The six example questions** show answers Claude Sonnet 5.5 actually wrote from the same pipeline
  and prompt, recorded through the Claude Code CLI (`web/build_data.py --record`) and labelled as
  recorded runs on the page.
- **Your own questions** get the three passage sentences closest in meaning to the question, quoted
  word for word and cited.

**Licences.** 36 of the 60 papers are CC BY or CC BY-NC-ND and are shown in full with attribution. The
other 24 are under arXiv's non-exclusive licence, which doesn't allow redistribution. For those the page
holds only a 40-word excerpt plus word counts for keyword search, so their keyword ranking is identical,
with a link to read the paper on arXiv. Abstracts are arXiv metadata and are always shown. PDFs and full
text are never committed to this repo.

## Files

- `rag/fetch.py`: arXiv search and PDF download
- `rag/index.py`: PDF text, chunking, embeddings
- `rag/retrieve.py`: BM25, vectors, RRF, reranking
- `rag/generate.py`: prompt, Claude streaming answer, citation check, recorded-run helper
- `eval/`: question sets, `run_eval.py`, results
- `app.py`: Streamlit app
- `web/`: the static demo (`build_data.py` exports its data)
