from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
MAGENTA = "\033[35m"

Role = Literal["user", "assistant", "system"]


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET


def _hr(char: str = "─", width: int = 70) -> str:
    return _c(char * width, DIM)


def _truncate(text: str, max_len: int = 120) -> str:
    text = str(text).replace("\n", "\\n")
    return text[:max_len] + _c("…", DIM) if len(text) > max_len else text


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
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix not in (".jsonl", ".json"):
        raise ValueError(f"Unrecognized extension: {path.suffix}. Expected .jsonl or .json")

    records: list[dict] = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"X Line {i} is invalid JSON: {e}", file=sys.stderr)

    if not records:
        raise ValueError("File is empty or contains no valid records.")

    return records


def _block_to_str(block: dict) -> str:
    """Flatten a single content block to plain text."""
    btype = block.get("type", "")

    if btype == "text":
        return block.get("text", "")

    if btype == "thinking":
        return f"<thinking>{block.get('thinking', '')}</thinking>"

    if btype == "tool_use":
        name = block.get("name", "")
        inp  = json.dumps(block.get("input", {}), ensure_ascii=False)
        return f"<tool_use name='{name}'>{inp}</tool_use>"

    if btype == "tool_result":
        content = block.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        return f"<tool_result>{content}</tool_result>"

    return ""


def _content_to_str(content: str | list | None) -> str:
    """Flatten a content field (string or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_block_to_str(b) for b in content if isinstance(b, dict)]
        return "\n".join(p for p in parts if p.strip())
    return ""


def _collect_block_types(content: str | list | None) -> list[str]:
    if isinstance(content, str):
        return ["raw_string"]
    if isinstance(content, list):
        return [b.get("type", "unknown") for b in content if isinstance(b, dict)]
    return ["unknown"]


def records_to_messages(records: list[dict]) -> list[dict]:
    """
    Convert a list of Claude Code session records to the Transformers messages format:
    [{"role": "user" | "assistant", "content": str}, ...]

    System records are skipped. Content blocks (tool_use, tool_result, thinking)
    are flattened to plain text using XML-style tags.

    Args:
        records: List of records returned by load_session().

    Returns:
        List of dicts with "role" and "content" keys.
    """
    messages: list[dict] = []

    for record in records:
        rtype = record.get("type")
        if rtype not in ("user", "assistant"):
            continue

        msg     = record.get("message", {})
        role    = msg.get("role", rtype)
        content = msg.get("content", "")
        text    = _content_to_str(content)

        if not text.strip():
            continue

        messages.append({"role": role, "content": text})

    return messages


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
    records  = load_session(path)
    messages = records_to_messages(records)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    return messages


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
    path    = Path(path)
    records = load_session(path)

    _print_header(path, records)
    _print_record_types(records)
    _print_block_summary(records)
    _print_token_usage(records)

    if show_flow:
        _print_flow(records, show_blocks=show_blocks)

    if show_raw:
        _print_raw_examples(records)

    print(_hr("═"))
    print()


def _print_header(path: Path, records: list[dict]) -> None:
    first_user = next((r for r in records if r.get("type") == "user"), None)

    print()
    print(_hr("═"))
    print(_c("  CLAUDE CODE SESSION INSPECTOR", BOLD + CYAN))
    print(_hr("═"))
    print(f"  File    : {_c(str(path), BOLD)}")
    print(f"  Size    : {_c(f'{path.stat().st_size / 1024:.1f} KB', BOLD)}")
    print(f"  Lines   : {_c(len(records), BOLD)}")

    if first_user:
        print(f"  Session : {_c(first_user.get('sessionId', '?'), DIM)}")
        print(f"  CWD     : {_c(first_user.get('cwd', '?'), DIM)}")
        if branch := first_user.get("gitBranch"):
            print(f"  Branch  : {_c(branch, DIM)}")

    print(_hr("═"))


def _scaled_bar(count: int, max_count: int, width: int = 40) -> str:
    """Longitud de barra proporcional al máximo del conjunto, no a un tope fijo."""
    if max_count <= 0:
        return ""
    length = max(1, round((count / max_count) * width))
    return "█" * length


def _print_record_types(records: list[dict]) -> None:
    print()
    print(_c("  RECORD TYPES", BOLD))
    print(_hr())

    counts      = Counter(r.get("type", "unknown") for r in records)
    role_colors = {"user": GREEN, "assistant": CYAN, "system": YELLOW}
    max_count   = counts.most_common(1)[0][1] if counts else 0

    for rtype, count in counts.most_common():
        bar = _scaled_bar(count, max_count)
        cf  = role_colors.get(rtype, DIM)
        print(f"  {_c(rtype.ljust(12), cf)} {_c(count, BOLD):>4}  {_c(bar, cf)}")


def _print_block_summary(records: list[dict]) -> None:
    print()
    print(_c("  CONTENT BLOCKS (by record type)", BOLD))
    print(_hr())

    by_role: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        rtype   = r.get("type", "unknown")
        content = r.get("message", {}).get("content", [])
        for btype in _collect_block_types(content):
            by_role[rtype][btype] += 1

    role_colors = {"user": GREEN, "assistant": CYAN, "system": YELLOW}
    for role, counter in sorted(by_role.items()):
        cf = role_colors.get(role, DIM)
        print(f"  {_c(role, cf + BOLD)}")
        for btype, cnt in counter.most_common():
            print(f"    {_c('·', DIM)} {btype.ljust(20)} {_c(cnt, BOLD)}")


def _print_token_usage(records: list[dict]) -> None:
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    for r in records:
        u = r.get("message", {}).get("usage", {})
        totals["input"]       += u.get("input_tokens", 0)
        totals["output"]      += u.get("output_tokens", 0)
        totals["cache_read"]  += u.get("cache_read_input_tokens", 0)
        totals["cache_write"] += u.get("cache_creation_input_tokens", 0)

    if totals["input"] == 0 and totals["output"] == 0:
        return

    print()
    print(_c("  TOKEN USAGE", BOLD))
    print(_hr())
    inp   = f"{totals['input']:,}"
    out   = f"{totals['output']:,}"
    cread = f"{totals['cache_read']:,}"
    cwrit = f"{totals['cache_write']:,}"
    total = f"{totals['input'] + totals['output']:,}"
    print(f"  Input       : {_c(inp, BOLD)}")
    print(f"  Output      : {_c(out, BOLD)}")
    print(f"  Cache read  : {_c(cread, DIM)}")
    print(f"  Cache write : {_c(cwrit, DIM)}")
    print(f"  Total       : {_c(total, BOLD + CYAN)}")


def _render_block(block: dict, indent: str = "       ") -> str:
    btype        = block.get("type", "?")
    block_colors = {"text": GREEN, "thinking": MAGENTA, "tool_use": YELLOW, "tool_result": CYAN}
    cf           = block_colors.get(btype, RED)
    lines        = []

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

    else:
        lines.append(f"{indent}{_c(btype, RED)}: {_truncate(str(block), 100)}")

    return "\n".join(lines)


def _print_flow(records: list[dict], show_blocks: bool = False) -> None:
    print()
    print(_c("  CONVERSATION FLOW", BOLD))
    print(_hr())

    msg_records = [r for r in records if r.get("type") in ("user", "assistant")]

    for i, r in enumerate(msg_records):
        rtype   = r.get("type")
        msg     = r.get("message", {})
        content = msg.get("content", [])
        ts      = r.get("timestamp", "")[:19].replace("T", " ")
        tokens  = msg.get("usage", {}).get("output_tokens", 0)

        cf         = GREEN if rtype == "user" else CYAN
        role_label = _c(f"[{rtype.upper()}]".ljust(12), cf + BOLD)
        tk_label   = _c(f"{tokens}tk", DIM) if tokens else ""
        btypes     = _collect_block_types(content)

        print(f"  {i+1:>3}. {role_label} {_c(ts, DIM)} {tk_label}")
        print(f"       {_c('blocks: ' + ', '.join(btypes), DIM)}")

        if show_blocks and isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    print(_render_block(block))

        print()


def _print_raw_examples(records: list[dict]) -> None:
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

        print(_hr("·"))