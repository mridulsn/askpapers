// Hybrid search core, shared by the page and the parity test (no DOM here).
// Mirrors rag/retrieve.py: same tokenizer, same BM25 (rank_bm25 Okapi), same RRF.

const STOP = new Set(("a an and are as at be by for from has have in into is it its of on or that the " +
  "this to was were what which who will with how why when does do can we our their they these " +
  "those than then there via").split(/\s+/));

export const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";

export function tokenize(t) {
  return (t.toLowerCase().match(/[a-z0-9]+(?:-[a-z0-9]+)*/g) || []).filter((w) => !STOP.has(w));
}

export class BM25 {
  constructor(docs, k1 = 1.5, b = 0.75, epsilon = 0.25) {
    this.k1 = k1; this.b = b;
    this.tf = []; this.len = new Float64Array(docs.length);
    const df = new Map();
    let total = 0;
    // A doc is either text, or a pre-counted {word: count} object (used for papers whose
    // licence doesn't allow shipping the full text; ranking stays identical).
    docs.forEach((d, i) => {
      let m, n = 0;
      if (typeof d === "string") {
        const toks = tokenize(d); m = new Map();
        for (const w of toks) m.set(w, (m.get(w) || 0) + 1);
        n = toks.length;
      } else {
        m = new Map(Object.entries(d));
        for (const v of m.values()) n += v;
      }
      this.tf.push(m); this.len[i] = n; total += n;
      for (const w of m.keys()) df.set(w, (df.get(w) || 0) + 1);
    });
    this.avgdl = total / docs.length;
    this.idf = new Map();
    let sum = 0; const neg = [];
    for (const [w, n] of df) {
      const v = Math.log(docs.length - n + 0.5) - Math.log(n + 0.5);
      this.idf.set(w, v); sum += v;
      if (v < 0) neg.push(w);
    }
    const eps = epsilon * (sum / this.idf.size);
    for (const w of neg) this.idf.set(w, eps);
  }
  scores(query) {
    const s = new Float64Array(this.tf.length);
    for (const q of tokenize(query)) {
      const idf = this.idf.get(q) || 0;
      if (!idf) continue;
      for (let i = 0; i < this.tf.length; i++) {
        const f = this.tf[i].get(q);
        if (!f) continue;
        s[i] += idf * (f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + this.b * this.len[i] / this.avgdl));
      }
    }
    return s;
  }
}

export function topK(scores, k, positiveOnly = false) {
  const idx = [];
  for (let i = 0; i < scores.length; i++) if (!positiveOnly || scores[i] > 0) idx.push(i);
  idx.sort((a, b) => scores[b] - scores[a] || a - b);   // ties: lower index first (= Python stable argsort)
  return idx.slice(0, k);
}

// Dot product of a float query vector with int8 document vectors (scaled by 127).
export function denseScores(q, emb, dim) {
  const n = emb.length / dim, s = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0; const o = i * dim;
    for (let j = 0; j < dim; j++) acc += q[j] * emb[o + j];
    s[i] = acc / 127;
  }
  return s;
}

// Reciprocal Rank Fusion: uses ranks only, so BM25 and cosine scores never need a common scale.
export function rrf(lists, k = 60) {
  const score = new Map();
  for (const list of lists) list.forEach((id, r) => score.set(id, (score.get(id) || 0) + 1 / (k + r + 1)));
  return [...score.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => id);
}

// Whole sentences only: PDF text has headers, table cells and cut-off chunk edges, so drop
// anything that does not start with a capital letter and end with a full stop.
export function splitSentences(text) {
  const parts = text.split(/(?<=[.!?])\s+(?=[A-Z])/).map((s) => s.trim());
  // The first and last pieces of a chunk are usually cut mid-sentence.
  return parts.slice(1, -1).filter((s) =>
    s.length >= 60 && s.length <= 400 && /^[A-Z][a-z]/.test(s) && /[a-z][.!?]$/.test(s) &&
    !/^(Fig(ure)?|Table|Algorithm|Equation|Section|RQ\d)\b/.test(s) &&
    !/\bet al\.$/.test(s) &&
    !/\b\d+(\.\d+)+\s+[A-Z]/.test(s) &&   // run-in section headings like "5.2.3 A Case Study"
    (s.match(/\d/g) || []).length < s.length * 0.08);
}

export const POOL = 50, RERANK_POOL = 30;

// The whole retrieval pipeline, shared by the page (web/app.js) and the Node evaluation
// (tests/web_eval.mjs), so the published numbers measure exactly what the page runs.
//   embedQuery(q)          -> query vector (normalised, length dim)
//   rerankScores(q, docs)  -> one score per doc text
// Ties keep index order, matching Python's stable argsort.
export async function hybridSearch({ q, mode, chunks, emb, dim, bm25, embedQuery, rerankScores, onStage, k = 5 }) {
  const stages = {};
  if (mode !== "dense") stages.bm25 = topK(bm25.scores(q), POOL, true);
  if (mode !== "bm25") stages.dense = topK(denseScores(await embedQuery(q), emb, dim), POOL);
  let ranked;
  if (mode === "bm25") ranked = stages.bm25;
  else if (mode === "dense") ranked = stages.dense;
  else {
    stages.hybrid = rrf([stages.bm25, stages.dense]);
    ranked = stages.hybrid;
    if (mode === "hybrid+rerank") {
      onStage?.("rerank");
      // Excerpt-only passages (licence) can't be judged fairly from 40 words, so they keep
      // their merged slot and only the full-text passages are reordered around them.
      const ids = stages.hybrid.slice(0, RERANK_POOL);
      const movable = ids.filter((i) => !chunks[i].tf);
      let order = [];
      if (movable.length) {
        const s = await rerankScores(q, movable.map((i) => `${chunks[i].paper.title}. ${chunks[i].text}`));
        order = movable.map((id, j) => [id, s[j], j]).sort((a, b) => b[1] - a[1] || a[2] - b[2]).map(([id]) => id);
      }
      let n = 0;
      stages.rerank = ids.map((id) => (chunks[id].tf ? id : order[n++]));
      ranked = stages.rerank;
    }
  }
  return { final: ranked.slice(0, k).map((i) => chunks[i]), stages };
}
