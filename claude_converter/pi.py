from __future__ import annotations

import json
import sys
from pathlib import Path

from claude_converter.utils import (
    InspectorSchema,
    convert_base64_to_pil_image,
    dget,
    dlist,
    finalize_content,
    load_jsonl,
    merge_content_parts,
    messages_to_json_safe,
    run_inspection,
)


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

    if btype == "image":
        return "<image>"

    return ""


def _decode_pi_image(block: dict):
    """
    Decode a Pi image content block: {"type": "image", "data": "<base64>",
    "mimeType": "image/png"}. Returns None (with a stderr warning) instead
    of raising on empty or undecodable base64.
    """
    data = block.get("data", "")
    if not data:
        print("X Skipping image block with empty base64 data", file=sys.stderr)
        return None
    try:
        return convert_base64_to_pil_image(data)
    except (ImportError, ValueError) as e:
        print(f"X Skipping unreadable image block: {e}", file=sys.stderr)
        return None


def _message_of(record: dict) -> dict | None:
    if record.get("type") != "message":
        return None
    return dget(record, "message")


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
        parts = []
        for b in dlist(message, "content"):
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif b.get("type") == "image":
                parts.append("<image>")
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


def _message_parts(record: dict) -> list[dict]:
    """
    Like _text_of, but keeps text and image blocks as separate multimodal
    content parts (the `transformers` chat-template format) instead of
    flattening everything into one string. Both plain user/assistant
    messages and toolResult content can carry ImageContent per Pi's
    session format.
    """
    message = _message_of(record) or {}
    role = message.get("role", "")

    if role == "toolResult":
        content = dlist(message, "content")
        has_image = any(isinstance(b, dict) and b.get("type") == "image" for b in content)

        if not has_image:
            # No images inside: collapse to the exact same single text part
            # _text_of produces, so text-only tool results stay byte-for-byte
            # identical to the pre-multimodal output.
            text = _text_of(record)
            return [{"type": "text", "text": text}] if text.strip() else []

        tool_name = message.get("toolName", "")
        tool_id = message.get("toolCallId", "")
        parts: list[dict] = [{"type": "text", "text": f"<tool_result name='{tool_name}' id='{tool_id}'>"}]

        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and b.get("text", "").strip():
                parts.append({"type": "text", "text": b["text"]})
            elif b.get("type") == "image":
                image = _decode_pi_image(b)
                if image is not None:
                    parts.append({"type": "image", "image": image})

        parts.append({"type": "text", "text": "</tool_result>"})
        return parts

    content = message.get("content", "")
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "image":
                image = _decode_pi_image(b)
                if image is not None:
                    parts.append({"type": "image", "image": image})
            else:
                text = _block_to_text(b)
                if text.strip():
                    parts.append({"type": "text", "text": text})
        return parts
    return []


def records_to_messages_pi(records: list[dict]) -> list[dict]:
    """
    Convert Pi session records to the Transformers messages format:
    [{"role": "user" | "assistant", "content": str | list[dict]}, ...]

    Tool calls and tool results are flattened with XML-style tags, and
    consecutive records that resolve to the same role are merged so
    roles alternate cleanly. Image content (in user/assistant messages
    or in toolResult content) is kept as a multimodal part instead of
    being dropped; text-only turns keep the plain string content format.
    """
    turns: list[dict] = []  # [{"role": ..., "parts": [...]}]

    for record in records:
        role = _record_role(record)
        if role is None:
            continue

        parts = _message_parts(record)
        if not parts:
            continue

        if turns and turns[-1]["role"] == role:
            merge_content_parts(turns[-1]["parts"], parts)
        else:
            turns.append({"role": role, "parts": list(parts)})

    return [{"role": t["role"], "content": finalize_content(t["parts"])} for t in turns]


def session_to_messages_pi(
    path: str | Path,
    output: str | Path | None = None,
) -> list[dict]:
    """
    Load a Pi session and convert it to the Transformers messages format.

    output: if provided, saves the result as JSON. Image parts are
    re-encoded as base64 data URIs for the saved file; the in-memory
    list returned still has real PIL.Image objects.
    """
    records = load_session_pi(path)
    messages = records_to_messages_pi(records)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(messages_to_json_safe(messages), f, ensure_ascii=False, indent=2)

    return messages


def _record_type_of(record: dict) -> str:
    rtype = record.get("type", "unknown")
    if rtype == "message":
        role = dget(record, "message").get("role", "unknown")
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
        usage = dget(r, "message").get("usage")
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