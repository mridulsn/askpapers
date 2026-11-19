"""Answer a question from retrieved chunks, with numbered citations.

Uses Claude through the Anthropic SDK. Without credentials it returns None and the
app shows the retrieved passages only (retrieval still works fully offline).
"""
import os
import re
from collections.abc import Iterator

MODEL = os.environ.get("RAG_MODEL", "claude-opus-5-5")

SYSTEM = """You answer questions about research papers using ONLY the numbered sources provided.

Rules:
- Every factual sentence ends with the source number(s) it came from, like [2] or [1][3].
- Use only what the sources say. Do not add outside knowledge, even if you know it.
- If the sources do not contain the answer, say "The retrieved papers don't answer this." and stop.
- If sources disagree, say so and cite both sides.
- Be concise: a short paragraph or a few bullets. Plain language."""


def format_sources(chunks: list[dict]) -> str:
    parts = []
    for n, c in enumerate(chunks, 1):
        where = "abstract" if c["page"] == 0 else f"page {c['page']}"
        parts.append(f"[{n}] {c['title']} (arXiv:{c['paper']}, {where})\n{c['text']}")
    return "\n\n".join(parts)


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def stream_answer(question: str, chunks: list[dict]) -> Iterator[str] | None:
    """Yield the answer text as it is generated, or return None if no credentials."""
    if not available():
        return None
    import anthropic

    client = anthropic.Anthropic()
    user = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"

    def gen() -> Iterator[str]:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
            # If the model declines, the API retries on a suitable fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            yield from stream.text_stream
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            yield "\n\n(The model declined to answer this question.)"
        elif msg.stop_reason == "max_tokens":
            yield "\n\n(Answer cut off at the length limit.)"

    return gen()


CLI_MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}


def answer_cli(question: str, chunks: list[dict], model: str = "sonnet") -> dict:
    """Answer through the local Claude Code CLI (for recorded demo runs without an API key).

    The flags isolate the call: no MCP servers, no plugins' servers, no hooks.
    """
    import json
    import subprocess
    import tempfile

    user = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
    cmd = ["claude", "-p", "--model", model, "--output-format", "json",
           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--settings", '{"disableAllHooks":true}', "--system-prompt", SYSTEM]
    out = subprocess.run(cmd, input=user, capture_output=True, text=True,
                         cwd=tempfile.mkdtemp(), timeout=300)
    if out.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {out.stderr.strip()[:300]}")
    data = json.loads(out.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude CLI error: {data.get('result', '')[:300]}")
    return {"answer": data["result"], "model": CLI_MODELS.get(model, model),
            "via": "Claude Code CLI", "usage": data.get("usage", {}),
            "cost_usd": data.get("total_cost_usd")}


def check_citations(answer: str, n_sources: int) -> dict:
    """Which sources the answer cited, and any citation numbers that don't exist."""
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    return {"cited": sorted({x for x in nums if 1 <= x <= n_sources}),
            "invalid": sorted({x for x in nums if not 1 <= x <= n_sources})}
