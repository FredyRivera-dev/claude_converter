from __future__ import annotations

import json
import sys
from pathlib import Path

from .utils import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    YELLOW,
    InspectorSchema,
    _c,
    _hr,
    _truncate,
    convert_base64_to_pil_image,
    finalize_content,
    load_jsonl,
    merge_content_parts,
    run_inspection,
)

# ── parseo especifico de Claude Code (unico contenido no compartible) ──

def _block_to_str(block: dict) -> str:
    """Flatten a single content block to plain text."""
    btype = block.get("type", "")

    if btype == "text":
        return block.get("text", "")

    if btype == "thinking":
        return f"<thinking>{block.get('thinking', '')}</thinking>"

    if btype == "tool_use":
        name = block.get("name", "")
        inp = json.dumps(block.get("input", {}), ensure_ascii=False)
        return f"<tool_use name='{name}'>{inp}</tool_use>"

    if btype == "tool_result":
        content = block.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        return f"<tool_result>{content}</tool_result>"

    if btype == "image":
        return "<image>"

    return ""


def _decode_claude_image(block: dict):
    """
    Decode a Claude Code image block:
    {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}

    Returns None (with a stderr warning) instead of raising, for two
    known real-world cases: a documented Claude Code bug where
    source.data arrives as an empty string over remote control, and
    plain corrupted/undecodable base64.
    """
    source = block.get("source", {})
    if source.get("type") != "base64":
        print(f"X Skipping image with unsupported source type: {source.get('type')!r}", file=sys.stderr)
        return None

    data = source.get("data", "")
    if not data:
        print("X Skipping image block with empty base64 data", file=sys.stderr)
        return None

    try:
        return convert_base64_to_pil_image(data)
    except (ImportError, ValueError) as e:
        print(f"X Skipping unreadable image block: {e}", file=sys.stderr)
        return None


def _block_to_parts(block: dict) -> dict | None:
    """
    Like _block_to_str, but keeps an image block as a distinct multimodal
    part {"type": "image", "image": PIL.Image} instead of flattening it
    into text. Returns None for empty/unsupported/undecodable blocks.
    """
    btype = block.get("type", "")

    if btype == "image":
        image = _decode_claude_image(block)
        return {"type": "image", "image": image} if image is not None else None

    text = _block_to_str(block)
    return {"type": "text", "text": text} if text.strip() else None


def _content_to_str(content: str | list | None) -> str:
    """Flatten a content field (string or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_block_to_str(b) for b in content if isinstance(b, dict)]
        return "\n".join(p for p in parts if p.strip())
    return ""


def _content_to_parts(content: str | list | None) -> list[dict]:
    """
    Like _content_to_str, but keeps text and image blocks as separate
    multimodal content parts (the `transformers` chat-template format:
    [{"type": "text", "text": ...}, {"type": "image", "image": PIL.Image}])
    instead of flattening everything into one string.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if isinstance(content, list):
        parts = [_block_to_parts(b) for b in content if isinstance(b, dict)]
        return [p for p in parts if p is not None]
    return []


def _collect_block_types(content: str | list | None) -> list[str]:
    if isinstance(content, str):
        return ["raw_string"]
    if isinstance(content, list):
        return [b.get("type", "unknown") for b in content if isinstance(b, dict)]
    return ["unknown"]


def _render_block(block: dict, indent: str = "       ") -> str:
    btype = block.get("type", "?")
    block_colors = {"text": GREEN, "thinking": MAGENTA, "tool_use": YELLOW, "tool_result": CYAN, "image": BOLD + CYAN}
    cf = block_colors.get(btype, RED)
    lines = []

    if btype == "text":
        lines.append(f"{indent}{_c('text', cf)}: {_truncate(block.get('text', ''), 200)}")

    elif btype == "thinking":
        lines.append(f"{indent}{_c('thinking', cf)}: {_truncate(block.get('thinking', ''), 120)}")

    elif btype == "tool_use":
        inp = _truncate(json.dumps(block.get("input", {}), ensure_ascii=False), 120)
        lines.append(f"{indent}{_c('tool_use', cf)}: {_c(block.get('name', '?'), BOLD)} [{_c(block.get('id', '?'), DIM)}]")
        lines.append(f"{indent}  input: {inp}")

    elif btype == "tool_result":
        content = block.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        lines.append(f"{indent}{_c('tool_result', cf)}: [{_c(block.get('tool_use_id', '?'), DIM)}]")
        lines.append(f"{indent}  content: {_truncate(str(content), 120)}")

    elif btype == "image":
        source = block.get("source", {})
        data_len = len(source.get("data", "") or "")
        lines.append(f"{indent}{_c('image', cf)}: {source.get('media_type', '?')} ({data_len} base64 chars)")

    else:
        lines.append(f"{indent}{_c(btype, RED)}: {_truncate(str(block), 100)}")

    return "\n".join(lines)


# ── API publica: carga y conversion ──

def load_session(path: str | Path) -> list[dict]:
    """
    Load a Claude Code .jsonl session file.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of record dicts, one per valid line in the JSONL.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not .jsonl or .json,
                    or if the file contains no valid records.
    """
    return load_jsonl(path)


def records_to_messages(records: list[dict]) -> list[dict]:
    """
    Convert a list of Claude Code session records to the Transformers messages format:
    [{"role": "user" | "assistant", "content": str | list[dict]}, ...]

    System records are skipped. Text, thinking, tool_use and tool_result
    blocks are flattened to plain text using XML-style tags. Image blocks
    are kept as separate multimodal parts (e.g. [{"type": "text", ...},
    {"type": "image", "image": PIL.Image}]) instead of being dropped;
    messages with no images keep the plain string "content" format used
    before multimodal support existed.

    Args:
        records: List of records returned by load_session().

    Returns:
        List of dicts with "role" and "content" keys.
    """
    turns: list[dict] = []  # [{"role": ..., "parts": [...]}]

    for record in records:
        rtype = record.get("type")
        if rtype not in ("user", "assistant"):
            continue

        msg = record.get("message", {})
        role = msg.get("role", rtype)
        parts = _content_to_parts(msg.get("content", ""))

        if not parts:
            continue

        if turns and turns[-1]["role"] == role:
            merge_content_parts(turns[-1]["parts"], parts)
        else:
            turns.append({"role": role, "parts": list(parts)})

    return [{"role": t["role"], "content": finalize_content(t["parts"])} for t in turns]


def session_to_messages(
    path: str | Path,
    output: str | Path | None = None,
) -> list[dict]:
    """
    Load a Claude Code session and convert it to the Transformers messages format.

    Args:
        path:   Path to the .jsonl session file.
        output: If provided, saves the result as JSON to this path.

    Returns:
        List of dicts {"role": ..., "content": ...} ready for apply_chat_template().

    Example:
        messages = session_to_messages("session.jsonl", output="messages.json")
        tokenizer.apply_chat_template(messages, tokenize=True, ...)
    """
    records = load_session(path)
    messages = records_to_messages(records)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    return messages


# ── schema para el reporte generico compartido (utils.run_inspection) ──

def _record_type_of(r: dict) -> str:
    return r.get("type", "unknown")


def _is_message(r: dict) -> bool:
    return r.get("type") in ("user", "assistant")


def _role_of(r: dict) -> str:
    return r.get("message", {}).get("role", r.get("type", "unknown"))


def _text_of(r: dict) -> str:
    return _content_to_str(r.get("message", {}).get("content", ""))


def _timestamp_of(r: dict) -> str:
    return r.get("timestamp", "")[:19].replace("T", " ")


def _tokens_of(r: dict) -> int:
    return r.get("message", {}).get("usage", {}).get("output_tokens", 0)


def _block_types_of(r: dict) -> list[str]:
    return _collect_block_types(r.get("message", {}).get("content", []))


def _blocks_of(r: dict) -> list[dict]:
    content = r.get("message", {}).get("content", [])
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _total_tokens_of(records: list[dict]) -> dict[str, int]:
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    for r in records:
        u = r.get("message", {}).get("usage", {})
        totals["input"] += u.get("input_tokens", 0)
        totals["output"] += u.get("output_tokens", 0)
        totals["cache_read"] += u.get("cache_read_input_tokens", 0)
        totals["cache_write"] += u.get("cache_creation_input_tokens", 0)

    if totals["input"] == 0 and totals["output"] == 0:
        return {}

    totals["total"] = totals["input"] + totals["output"]
    return totals


CLAUDE_SCHEMA = InspectorSchema(
    label="CLAUDE CODE",
    record_type_of=_record_type_of,
    is_message=_is_message,
    role_of=_role_of,
    text_of=_text_of,
    timestamp_of=_timestamp_of,
    total_tokens_of=_total_tokens_of,
    tokens_of=_tokens_of,
    block_types_of=_block_types_of,
    blocks_of=_blocks_of,
    render_block=_render_block,
)


def _print_session_metadata(records: list[dict]) -> None:
    """Extra header line specific to Claude Code: sessionId, cwd, git branch."""
    first_user = next((r for r in records if r.get("type") == "user"), None)
    if not first_user:
        return

    print(f"  Session : {_c(first_user.get('sessionId', '?'), DIM)}")
    print(f"  CWD     : {_c(first_user.get('cwd', '?'), DIM)}")
    if branch := first_user.get("gitBranch"):
        print(f"  Branch  : {_c(branch, DIM)}")


def _print_raw_examples(records: list[dict]) -> None:
    """
    Richer raw-record dump than utils.print_raw_examples: also inspects the
    first content block, since Claude Code nests tool calls inside "message".
    """
    print()
    print(_c("  RAW RECORD EXAMPLES (one per type)", BOLD))
    print(_hr())

    seen: set[str] = set()
    for r in records:
        rtype = r.get("type", "unknown")
        if rtype in seen:
            continue
        seen.add(rtype)

        print(_c(f"  type = {rtype}", BOLD + YELLOW))
        print(_c("  top-level keys:", DIM))
        print(f"    {_c(list(r.keys()), DIM)}")

        msg = r.get("message", {})
        if msg:
            print(_c("  message keys:", DIM))
            print(f"    {_c(list(msg.keys()), DIM)}")

            content = msg.get("content", [])
            if isinstance(content, list) and content:
                block = content[0]
                if isinstance(block, dict):
                    print(_c("  content[0]:", DIM))
                    print(f"    type = {_c(block.get('type', '?'), YELLOW)}")
                    for k, v in block.items():
                        if k != "type":
                            print(f"    {k}: {_truncate(str(v), 100)}")
            elif isinstance(content, str):
                print(_c("  content (string):", DIM))
                print(f"    {_truncate(content, 100)}")

        print(_hr("."))


def inspect_session(
    path: str | Path,
    show_flow: bool = False,
    show_blocks: bool = False,
    show_raw: bool = False,
) -> None:
    """
    Print a color-coded inspection report for a Claude Code session.

    Args:
        path:        Path to the .jsonl file.
        show_flow:   If True, prints the timestamped conversation flow.
                     Defaults to False.
        show_blocks: If True, prints the content of each block inline in the flow.
                     Requires show_flow=True to have any effect.
        show_raw:    If True, appends one raw record example per record type found.
    """
    path = Path(path)
    records = load_session(path)

    run_inspection(
        path, records, CLAUDE_SCHEMA,
        show_flow=show_flow, show_blocks=show_blocks,
        extra_header=_print_session_metadata,
    )

    if show_raw:
        _print_raw_examples(records)
        print(_hr("="))
        print()