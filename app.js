import { pipeline, AutoTokenizer, AutoModelForSequenceClassification, env }
  from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { BM25, QUERY_PREFIX, denseScores, rrf, splitSentences, topK } from "./search.js";

env.allowLocalModels = false;
const EMBED_ID = "Xenova/bge-small-en-v1.5";
const RERANK_ID = "mixedbread-ai/mxbai-rerank-xsmall-v1";
const DIM = 384, POOL = 50, RERANK_POOL = 30, K = 5;

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const state = { mode: "hybrid+rerank" };
let papers, chunks, emb, bm25, answers, evalData, embedder, rrTok, rrModel;

function status(msg, busy = true) {
  $("#status").textContent = msg;
  $("#status").classList.toggle("busy", busy);
}

async function loadData() {
  const [p, c, e, a, ev] = await Promise.all([
    fetch("data/papers.json").then((r) => r.json()),
    fetch("data/chunks.json").then((r) => r.json()),
    fetch("data/emb.i8").then((r) => r.arrayBuffer()),
    fetch("data/answers.json").then((r) => (r.ok ? r.json() : [])).catch(() => []),
    fetch("data/eval.json").then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  papers = p; answers = a; evalData = ev;
  chunks = c.map(([pi, page, text, tf], id) => ({ id, paper: papers[pi], page, text, tf }));
  emb = new Int8Array(e);
  bm25 = new BM25(chunks.map((ch) => ch.tf || `${ch.paper.title}. ${ch.text}`));
}

// Each model loads once; a search started during the background prefetch waits on the same promise.
let embedP, rerankP;
function getEmbedder(quiet = false) {
  if (!embedP) {
    if (!quiet) status("Downloading the search model (34 MB, only the first time) ...");
    embedP = pipeline("feature-extraction", EMBED_ID, { dtype: "q8" });
    embedP.catch(() => { embedP = null; });
  } else if (!quiet && !embedder) status("Finishing the search model download ...");
  return embedP.then((m) => (embedder = m));
}

function getReranker(quiet = false) {
  if (!rerankP) {
    if (!quiet) status("Downloading the reranker (87 MB, only the first time) ...");
    rerankP = Promise.all([
      AutoTokenizer.from_pretrained(RERANK_ID),
      AutoModelForSequenceClassification.from_pretrained(RERANK_ID, { dtype: "q8" }),
    ]);
    rerankP.catch(() => { rerankP = null; });
  } else if (!quiet && !rrModel) status("Finishing the reranker download ...");
  return rerankP.then(([t, m]) => { rrTok = t; rrModel = m; return [t, m]; });
}

// Start downloading both models while the visitor reads, so the first search is quick.
function prefetch() {
  getEmbedder(true).then(() => getReranker(true))
    .catch((err) => console.warn("prefetch failed; models load on first search", err));
}

async function embed(texts) {
  const ex = await getEmbedder();
  const out = await ex(texts, { pooling: "cls", normalize: true });
  return out.tolist();
}

async function search(q, mode) {
  const stages = {};
  const t0 = performance.now();
  if (mode !== "dense") stages.bm25 = topK(bm25.scores(q), POOL, true);
  if (mode !== "bm25") {
    const [qv] = await embed([QUERY_PREFIX + q]);
    stages.dense = topK(denseScores(qv, emb, DIM), POOL);
  }
  let ranked;
  if (mode === "bm25") ranked = stages.bm25;
  else if (mode === "dense") ranked = stages.dense;
  else {
    stages.hybrid = rrf([stages.bm25, stages.dense]);
    ranked = stages.hybrid;
    if (mode === "hybrid+rerank") {
      const [tok, model] = await getReranker();
      status("Reranking the top 30 passages ...");
      // Excerpt-only passages (licence) can't be judged fairly from 40 words, so they keep
      // their merged slot and only the full-text passages are reordered around them.
      const ids = stages.hybrid.slice(0, RERANK_POOL);
      const movable = ids.filter((i) => !chunks[i].tf);
      const docs = movable.map((i) => `${chunks[i].paper.title}. ${chunks[i].text}`);
      let order = [];
      if (docs.length) {
        const inputs = tok(new Array(docs.length).fill(q), { text_pair: docs, padding: true, truncation: true, max_length: 512 });
        const { logits } = await model(inputs);
        const s = logits.tolist().map((x) => x[0]);
        order = movable.map((id, j) => [id, s[j]]).sort((a, b) => b[1] - a[1]).map(([id]) => id);
      }
      let k = 0;
      stages.rerank = ids.map((id) => (chunks[id].tf ? id : order[k++]));
      ranked = stages.rerank;
    }
  }
  return { final: ranked.slice(0, K).map((i) => chunks[i]), stages, ms: performance.now() - t0 };
}

async function keySentences(q, final) {
  const cands = [];
  final.forEach((c, n) => splitSentences(c.text).forEach((s) => cands.push({ s, n: n + 1 })));
  if (!cands.length) return [];
  const [qv] = await embed([QUERY_PREFIX + q]);
  const vs = await embed(cands.map((c) => c.s));
  cands.forEach((c, i) => { c.score = vs[i].reduce((a, v, j) => a + v * qv[j], 0); });
  // Best 3, at most 2 from the same source so one passage doesn't crowd out the rest.
  const picked = [], per = {};
  for (const c of cands.sort((a, b) => b.score - a.score)) {
    if ((per[c.n] || 0) >= 2) continue;
    picked.push(c); per[c.n] = (per[c.n] || 0) + 1;
    if (picked.length === 3) break;
  }
  return picked;
}

function prettyModel(id) {
  const m = id.match(/^claude-([a-z]+)-(\d+)-(\d+)$/);
  return m ? `Claude ${m[1][0].toUpperCase()}${m[1].slice(1)} ${m[2]}.${m[3]}` : id;
}

function inline(s) {
  return esc(s).replace(/\[(\d+)\]/g, (m, n) => `<a class="cite" href="#src-${n}">[${n}]</a>`)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

// Minimal Markdown: paragraphs and (nested) "- " bullet lists, which is all the answers use.
function renderAnswer(text) {
  let html = "", depth = 0;
  for (const line of text.split("\n")) {
    const m = line.match(/^(\s*)- (.*)$/);
    if (m) {
      const d = Math.floor(m[1].length / 2) + 1;
      while (depth < d) { html += "<ul>"; depth++; }
      while (depth > d) { html += "</ul>"; depth--; }
      html += `<li>${inline(m[2])}</li>`;
    } else {
      while (depth > 0) { html += "</ul>"; depth--; }
      if (line.trim()) html += `<p>${inline(line)}</p>`;
    }
  }
  while (depth > 0) { html += "</ul>"; depth--; }
  return html;
}

function renderSources(final) {
  $("#sources").innerHTML = final.map((c, i) => {
    const p = c.paper, where = c.page === 0 ? "abstract" : `page ${c.page}`;
    const who = p.authors.join(", ") + (p.more ? ` +${p.more}` : "");
    return `<article class="src" id="src-${i + 1}">
      <div class="src-n">${i + 1}</div>
      <div class="src-body">
        <a class="src-title" href="https://arxiv.org/abs/${esc(p.id)}" target="_blank" rel="noopener">${esc(p.title)}</a>
        <div class="meta">${esc(who)} &middot; ${esc(p.date.slice(0, 4))} &middot; arXiv:${esc(p.id)} &middot; ${where} &middot; ${esc(p.license)}</div>
        <details><summary>${c.tf ? "Read an excerpt" : "Read the passage"}</summary><p>${esc(c.text)}${c.tf ? ` <a href="https://arxiv.org/abs/${esc(p.id)}" target="_blank" rel="noopener">Read the full paper on arXiv</a>` : ""}</p></details>
      </div></article>`;
  }).join("");
}

function renderStages(final, stages) {
  const names = { bm25: "Keyword", dense: "Vector", hybrid: "Merged", rerank: "Reranked" };
  const cols = Object.keys(stages);
  const rows = final.map((c, i) => `<tr><td><span class="pn">[${i + 1}]</span><span class="pt"> ${esc(c.paper.title.slice(0, 56))}${c.paper.title.length > 56 ? "..." : ""}</span></td>` +
    cols.map((k) => { const r = stages[k].indexOf(c.id); return `<td class="num">${r < 0 ? "&ndash;" : r + 1}</td>`; }).join("") + "</tr>").join("");
  $("#stages").innerHTML = `<table><thead><tr><th>Passage</th>${cols.map((k) => `<th class="num">${names[k]}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
}

async function run(q) {
  q = q.trim();
  if (!q) return;
  $("#q").value = q;
  $("#results").hidden = false;
  $("#answer").innerHTML = "";
  try {
    const rec = answers.find((a) => a.question.toLowerCase() === q.toLowerCase());
    const res = await search(q, state.mode);
    let final = res.final;
    if (rec) {
      // Show the exact sources the recorded answer was written from.
      final = rec.sources.map((id) => chunks[id]);
      $("#answer-label").textContent = `Recorded run, ${prettyModel(rec.model)}, ${rec.date}`;
      $("#answer-note").textContent = "Written by Claude from the five passages on the right, using the Python version of this pipeline. Questions you type yourself get quoted key sentences instead, so this page needs no API key.";
      $("#answer").innerHTML = renderAnswer(rec.answer);
    } else {
      status("Picking the key sentences ...");
      const ks = await keySentences(q, final);
      $("#answer-label").textContent = "Key sentences, quoted from the papers";
      $("#answer-note").textContent = "Nothing here is generated. Each sentence is copied from a source passage and picked by how close its meaning is to your question.";
      $("#answer").innerHTML = ks.length
        ? `<ul class="quotes">${ks.map((k) => `<li>&ldquo;${esc(k.s)}&rdquo; <a class="cite" href="#src-${k.n}">[${k.n}]</a></li>`).join("")}</ul>`
        : "<p>No clear sentence matched. Read the passages on the right.</p>";
    }
    renderSources(final);
    renderStages(final, res.stages);
    const first = !state.searched; state.searched = true;
    status(`Searched ${chunks.length.toLocaleString()} passages in ${(res.ms / 1000).toFixed(1)} s` +
      (first && res.ms > 5000 ? " (the first search includes downloading the models)" : ""), false);
  } catch (err) {
    console.error(err);
    status(`Something went wrong: ${err.message}. Try again or switch to Keyword mode.`, false);
  }
}

function renderEval() {
  if (!evalData) return;
  const labels = { bm25: "Keyword only", dense: "Vector only", hybrid: "Hybrid", "hybrid+rerank": "Hybrid + rerank" };
  const sets = Object.entries(evalData);
  const modes = Object.keys(sets[0][1].modes);
  $("#eval").innerHTML = `<table><thead><tr><th>Pipeline</th>${sets.map(([k, v]) => `<th class="num">${k === "keyword" ? "Keyword-style" : "Reworded"} (${v.n})</th>`).join("")}</tr></thead><tbody>` +
    modes.map((m) => `<tr><td>${labels[m] || m}</td>${sets.map(([, v]) => `<td class="num">${Math.round(v.modes[m]["hit@5"] * 100)}%</td>`).join("")}</tr>`).join("") +
    "</tbody></table>";
}

function init() {
  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => {
    state.mode = b.dataset.mode;
    document.querySelectorAll("[data-mode]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    if ($("#q").value.trim()) run($("#q").value);
  }));
  $("#form").addEventListener("submit", (e) => { e.preventDefault(); run($("#q").value); });
  $("#examples").innerHTML = answers.map((a) => `<button type="button" class="chip">${esc(a.question)}</button>`).join("");
  $("#examples").addEventListener("click", (e) => { if (e.target.matches(".chip")) run(e.target.textContent); });
  $("#count").textContent = `${papers.length} papers, ${chunks.length.toLocaleString()} passages`;
  renderEval();
  status("Ready. Ask a question or pick an example.", false);
  $("#ask").disabled = false;
  // Shareable links: ?q=question&mode=hybrid
  const params = new URLSearchParams(location.search);
  const m = params.get("mode");
  if (m) document.querySelector(`[data-mode="${CSS.escape(m)}"]`)?.click();
  if (params.get("q")) run(params.get("q"));
  else prefetch();
}

loadData().then(init).catch((err) => status(`Could not load the index: ${err.message}`, false));
