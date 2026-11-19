// Checks the browser BM25 + RRF give the same rankings as the Python pipeline.
// Run with:  uv run python tests/parity_dump.py | node tests/parity.mjs
import { readFileSync } from "node:fs";
import { BM25, topK, rrf } from "../web/search.js";
const papers = JSON.parse(readFileSync("web/data/papers.json"));
const chunks = JSON.parse(readFileSync("web/data/chunks.json"));
// Same index the page builds: pre-counted words for excerpt-only passages, text otherwise.
const bm = new BM25(chunks.map(([p, , t, tf]) => tf || `${papers[p].title}. ${t}`));
const want = JSON.parse(readFileSync(0, "utf8"));
let bad = 0;
for (const w of want) {
  const b50 = topK(bm.scores(w.q), 50, true);
  const same = b50.join() === w.bm25.join();
  const fused = rrf([b50, w.dense50]).slice(0, 10);
  const sameF = fused.join() === w.hybrid.join();
  if (!same || !sameF) { bad++; console.log("MISMATCH:", w.q); }
}
console.log(`${want.length - bad}/${want.length} queries: identical BM25 top-50 and fused top-10`);
process.exit(bad ? 1 : 0);
