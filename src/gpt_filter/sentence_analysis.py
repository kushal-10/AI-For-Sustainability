#!/usr/bin/env python3
import json, re
from pathlib import Path
from typing import Any, Dict, List, Optional

IN_DIR = Path("data/classification_batches")
CTRL_CHARS = re.compile(r"[\x00-\x1F\x7F]")
short_sents = []

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _content_to_text(content: Any) -> str:
    # OpenAI-style can be a plain string or a list of blocks.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
        return "\n".join(parts)
    return ""

def _extract_sentence_from_obj(obj: Dict[str, Any]) -> Optional[str]:
    # Your batches: sentence is the content of the last user message in body.messages
    body = obj.get("body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            body = None

    if not isinstance(body, dict):
        # fallback to request.body if needed
        req = obj.get("request") or {}
        body = req.get("body")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                body = None

    if not isinstance(body, dict):
        return None

    messages = body.get("messages")
    if not isinstance(messages, list):
        inp = body.get("input")
        if isinstance(inp, dict):
            messages = inp.get("messages")

    if not isinstance(messages, list):
        return None

    last_user = None
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            last_user = m
    if not isinstance(last_user, dict):
        return None

    sent = _content_to_text(last_user.get("content"))
    sent = CTRL_CHARS.sub(" ", sent).strip()
    return sent or None

def _count_tokens(s: str) -> int:
    s = re.sub(r"\s+", " ", s.strip())
    if not s:
        return 0
    return len(s.split(" "))

def main():
    # Only *.jsonl (recursively)
    files = sorted(IN_DIR.rglob("*.jsonl"))
    if not files:
        print(f"No JSONL files found under {IN_DIR}/**")
        return

    total_lines = 0
    extracted = 0
    considered = 0
    short = 0
    token_sum = 0

    for path in files:
        for obj in _iter_jsonl(path):
            total_lines += 1
            s = _extract_sentence_from_obj(obj)
            if not isinstance(s, str):
                continue
            extracted += 1

            tok = _count_tokens(s)
            if tok == 0:
                continue
            considered += 1
            token_sum += tok
            if tok < 10:
                short += 1
                short_sents.append(s)

    pct = (short / considered * 100.0) if considered else 0.0
    avg = (token_sum / considered) if considered else 0.0
    print(f"{pct:.2f}% (<10 tokens)")
    print(f"(Total lines: {total_lines}, Extracted sentences: {extracted}, "
          f"Considered (non-empty): {considered}, Short: {short}, "
          f"Average length: {avg:.2f} tokens)")
    print(f"Short sentences: {short_sents}")

if __name__ == "__main__":
    main()
