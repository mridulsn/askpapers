// Measures the page's own pipeline: same search.js, same int8 vectors, same q8 models
// (bge-small embedder, mxbai reranker) run through transformers.js, here in Node.
// Run: npm i && node tests/web_eval.mjs   -> eval/results-browser.json
import { readFileSync, writeFileSync } from "node:fs";
import { pipeline, AutoTokenizer, AutoModelForSequenceClassification } from "@huggingface/transformers";
import { BM25, QUERY_PREFIX, hybridSearch } from "../web/search.js";

const papers = JSON.parse(readFileSync("web/data/papers.json"));
const chunks = JSON.parse(readFileSync("web/data/chunks.json"))
  .map(([pi, page, text, tf], id) => ({ id, paper: papers[pi], page, text, tf }));
const emb = new Int8Array(readFileSync("web/data/emb.i8").buffer.slice(0));
const bm25 = new BM25(chunks.map((c) => c.tf || `${c.paper.title}. ${c.text}`));
const embedder = await pipeline("feature-extraction", "Xenova/bge-small-en-v1.5", { dtype: "q8" });
const RR = "mixedbread-ai/mxbai-rerank-xsmall-v1";
const tok = await AutoTokenizer.from_pretrained(RR);
const model = await AutoModelForSequenceClassification.from_pretrained(RR, { dtype: "q8" });
const embedQuery = async (q) => (await embedder([QUERY_PREFIX + q], { pooling: "cls", normalize: true })).tolist()[0];
const rerankScores = async (q, docs) => {
  const inputs = tok(new Array(docs.length).fill(q), { text_pair: docs, padding: true, truncation: true, max_length: 512 });
  return (await model(inputs)).logits.tolist().map((x) => x[0]);
};

const MODES = ["bm25", "dense", "hybrid", "hybrid+rerank"];
const sets = {};
for (const [name, file] of [["keyword", "eval/questions.jsonl"], ["reworded", "eval/questions_reworded.jsonl"]]) {
  const qs = readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const out = { n_questions: qs.length, modes: {} };
  for (const mode of MODES) {
    let hits = 0, rr = 0;
    const t0 = performance.now();
    for (const q of qs) {
      const { final } = await hybridSearch({ q: q.q, mode, chunks, emb, dim: 384, bm25, embedQuery, rerankScores, k: 10 });
      const rank = final.map((c) => c.paper.id).indexOf(q.paper) + 1;
      if (rank && rank <= 5) hits++;
      if (rank) rr += 1 / rank;
    }
    out.modes[mode] = { "hit@5": hits / qs.length, "mrr@10": rr / qs.length, ms_per_query: (performance.now() - t0) / qs.length };
    console.log(name, mode.padEnd(14), (100 * hits / qs.length).toFixed(1) + "%", (rr / qs.length).toFixed(3));
  }
  sets[name] = out;
}
writeFileSync("eval/results-browser.json", JSON.stringify({
  config: { pipeline: "web/search.js + transformers.js 3.8.1 (q8 models) under Node", reranker: RR, embeddings: "int8 web/data/emb.i8" },
  sets }, null, 1));
console.log("wrote eval/results-browser.json");
