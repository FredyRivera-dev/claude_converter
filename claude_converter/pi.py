from __future__ import annotations

import json
from pathlib import Path

from claude_converter.utils import InspectorSchema, load_jsonl, run_inspection


def load_session_pi(path: str | Path) -> list[dict]:
    """
    Load a Pi .jsonl session file.

    Conversational turns are lines shaped like:
        {"type": "message", "id": ..., "parentId": ..., "message": {"role": ..., "content": [...]}}
    Other lines ("session", "model_change", "thinking_level_change", ...) are metadata.
    """
    return load_jsonl(path)


def _block_to_text(block: dict) -> str:
    btype = block.get("type", "")

    if btype == "text":
        return block.get("text", "")

    if btype == "thinking":
        return f"<thinking>{block.get('thinking', '')}</thinking>"

    if btype == "toolCall":
        name = block.get("name", "")
        args = json.dumps(block.get("arguments", {}), ensure_ascii=False)
        return f"<tool_use name='{name}'>{args}</tool_use>"

    return ""


def _message_of(record: dict) -> dict | None:
    if record.get("type") != "message":
        return None
    return record.get("message", {})


def _record_role(record: dict) -> str | None:
    message = _message_of(record)
    if message is None:
        return None

    role = message.get("role", "")
    if role == "toolResult":
        return "user"  # tool output fed back, equivalent to claude's tool_result block
    return role if role in ("user", "assistant") else None


def _text_of(record: dict) -> str:
    message = _message_of(record) or {}
    role = message.get("role", "")

    if role == "toolResult":
        content = message.get("content", [])
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p.strip())
        tool_name = message.get("toolName", "")
        tool_id = message.get("toolCallId", "")
        return f"<tool_result name='{tool_name}' id='{tool_id}'>{text}</tool_result>"

    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_block_to_text(b) for b in content if isinstance(b, dict)]
        return "\n".join(p for p in parts if p.strip())
    return ""


def records_to_messages_pi(records: list[dict]) -> list[dict]:
    """
    Convert Pi session records to the Transformers messages format:
    [{"role": "user" | "assistant", "content": str}, ...]

    Tool calls and tool results are flattened with XML-style tags, and
    consecutive records that resolve to the same role are merged so
    roles alternate cleanly.
    """
    messages: list[dict] = []

    for record in records:
        role = _record_role(record)
        if role is None:
            continue

        text = _text_of(record)
        if not text.strip():
            continue

        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + text
        else:
            messages.append({"role": role, "content": text})

    return messages


def session_to_messages_pi(
    path: str | Path,
    output: str | Path | None = None,
) -> list[dict]:
    """
    Load a Pi session and convert it to the Transformers messages format.
    """
    records = load_session_pi(path)
    messages = records_to_messages_pi(records)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    return messages


def _record_type_of(record: dict) -> str:
    rtype = record.get("type", "unknown")
    if rtype == "message":
        role = record.get("message", {}).get("role", "unknown")
        return f"message:{role}"
    return rtype


def _timestamp_of(record: dict) -> str:
    return str(record.get("timestamp", ""))[:19].replace("T", " ")


def _total_tokens_of(records: list[dict]) -> dict[str, int]:
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    found = False

    for r in records:
        if r.get("type") != "message":
            continue
        usage = r.get("message", {}).get("usage")
        if not usage:
            continue
        found = True
        totals["input"] += usage.get("input", 0)
        totals["output"] += usage.get("output", 0)
        totals["cache_read"] += usage.get("cacheRead", 0)
        totals["cache_write"] += usage.get("cacheWrite", 0)

    return totals if found else {}


PI_SCHEMA = InspectorSchema(
    label="PI",
    record_type_of=_record_type_of,
    is_message=lambda r: _record_role(r) is not None,
    role_of=lambda r: _record_role(r) or "other",
    text_of=_text_of,
    timestamp_of=_timestamp_of,
    total_tokens_of=_total_tokens_of,
)


def inspect_session_pi(
    path: str | Path,
    show_flow: bool = False,
    show_blocks: bool = False,
    show_raw: bool = False,
) -> None:
    """Print a color-coded inspection report for a Pi session."""
    records = load_session_pi(path)
    run_inspection(path, records, PI_SCHEMA, show_flow=show_flow, show_blocks=show_blocks)