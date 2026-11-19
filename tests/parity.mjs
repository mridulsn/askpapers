// Checks the browser BM25 + RRF give the same rankings as the Python pipeline.
// Usage: node tests/parity.mjs  (reads web/data and a Python dump on stdin)
import { readFileSync } from "node:fs";
import { BM25, topK, rrf } from "../web/search.js";
const papers = JSON.parse(readFileSync("web/data/papers.json"));
const chunks = JSON.parse(readFileSync("web/data/chunks.json"));
const bm = new BM25(chunks.map(([p, , t]) => `${papers[p].title}. ${t}`));
const want = JSON.parse(readFileSync(0, "utf8"));
let bad = 0;
for (const w of want) {
  const got = topK(bm.scores(w.q), 10, true);
  const same = got.join() === w.bm25.join();
  const fused = rrf([topK(bm.scores(w.q), 50, true), w.dense50]).slice(0, 10);
  const sameF = fused.join() === w.hybrid.join();
  if (!same || !sameF) { bad++; console.log("MISMATCH", w.q, { got, want: w.bm25, fused, wantF: w.hybrid }); }
}
console.log(`${want.length - bad}/${want.length} queries identical`);
process.exit(bad ? 1 : 0);
