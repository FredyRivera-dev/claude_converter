from __future__ import annotations

import json
import sys
from pathlib import Path

from claude_converter.utils import (
    InspectorSchema,
    convert_base64_to_pil_image,
    load_jsonl,
    run_inspection,
)

# response_item payload types that don't have an explicit "role" but that
# represent an implicit assistant action (tool call, reasoning) or an
# implicit user-side response (tool output fed back into the model).
_TOOL_ROLE = {
    "reasoning": "assistant",
    "function_call": "assistant",
    "custom_tool_call": "assistant",
    "function_call_output": "user",
    "custom_tool_call_output": "user",
}


def load_session_codex(path: str | Path) -> list[dict]:
    """
    Load a Codex .jsonl session file.

    Each line is a top-level record shaped like:
        {"timestamp": ..., "type": "response_item" | "event_msg" | ..., "payload": {...}}
    """
    return load_jsonl(path)


def _payload_text(payload: dict) -> str:
    """Flatten a response_item payload to plain text using XML-style tags."""
    ptype = payload.get("type", "")

    if ptype == "message":
        parts = []
        for block in payload.get("content", []):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("input_text", "output_text", "text"):
                parts.append(block.get("text", ""))
            elif btype == "input_image":
                parts.append("<image>")
        return "\n".join(p for p in parts if p.strip())

    if ptype == "reasoning":
        parts = [
            s.get("text", "")
            for s in payload.get("summary", [])
            if isinstance(s, dict)
        ]
        text = "\n".join(p for p in parts if p.strip())
        return f"<thinking>{text}</thinking>" if text else ""

    if ptype == "function_call":
        name = payload.get("name", "")
        args = payload.get("arguments", "")
        return f"<tool_use name='{name}'>{args}</tool_use>"

    if ptype == "function_call_output":
        return f"<tool_result>{payload.get('output', '')}</tool_result>"

    if ptype == "custom_tool_call":
        name = payload.get("name", "")
        inp = payload.get("input", "")
        return f"<tool_use name='{name}'>{inp}</tool_use>"

    if ptype == "custom_tool_call_output":
        return f"<tool_result>{payload.get('output', '')}</tool_result>"

    return ""


def _record_role(record: dict) -> str | None:
    """Return the conversational role of a record, or None if it isn't one."""
    if record.get("type") != "response_item":
        return None

    payload = record.get("payload", {})
    ptype = payload.get("type", "")

    if ptype == "message":
        role = payload.get("role", "")
        if role == "developer":
            return None  # system-style instructions, skipped like claude's "system"
        return role if role in ("user", "assistant") else None

    return _TOOL_ROLE.get(ptype)


def _payload_parts(payload: dict) -> list[dict]:
    """
    Like _payload_text, but keeps text and image blocks as separate
    multimodal content parts (the `transformers` chat-template format:
    [{"type": "text", "text": ...}, {"type": "image", "image": PILImage}])
    instead of flattening everything into one string.

    Only "message" payloads can contain images. Every other payload type
    (tool calls, tool output, reasoning) collapses to a single text part,
    same flattening as _payload_text.
    """
    ptype = payload.get("type", "")

    if ptype == "message":
        parts: list[dict] = []
        for block in payload.get("content", []):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")

            if btype in ("input_text", "output_text", "text"):
                text = block.get("text", "")
                if text.strip():
                    parts.append({"type": "text", "text": text})

            elif btype == "input_image":
                image_url = block.get("image_url", "")
                if not image_url:
                    continue
                try:
                    image = convert_base64_to_pil_image(image_url)
                except (ImportError, ValueError) as e:
                    print(f"X Skipping unreadable input_image: {e}", file=sys.stderr)
                    continue
                parts.append({"type": "image", "image": image})

        return parts

    text = _payload_text(payload)
    return [{"type": "text", "text": text}] if text.strip() else []


def _merge_parts(existing: list[dict], new: list[dict]) -> None:
    """Extend `existing` in place with `new`, joining adjacent text parts
    instead of leaving them as separate fragments."""
    for part in new:
        if existing and existing[-1]["type"] == "text" and part["type"] == "text":
            existing[-1] = {"type": "text", "text": existing[-1]["text"] + "\n" + part["text"]}
        else:
            existing.append(dict(part))


def _finalize_content(parts: list[dict]) -> str | list[dict]:
    """
    Collapse to a plain string when a message has no images, so text-only
    conversations keep the exact same "content": str format as before.
    Keep the list-of-parts form only when at least one image is present,
    since that's the shape `transformers` multimodal chat templates expect.
    """
    if all(p["type"] == "text" for p in parts):
        return "\n".join(p["text"] for p in parts)
    return parts


def records_to_messages_codex(records: list[dict]) -> list[dict]:
    """
    Convert Codex session records to the Transformers messages format:
    [{"role": "user" | "assistant", "content": str | list[dict]}, ...]

    Tool calls, tool outputs, and reasoning are flattened into the
    surrounding turn using XML-style tags, and consecutive records that
    resolve to the same role are merged so roles alternate cleanly.

    Messages containing images use the multimodal list-of-parts content
    format (e.g. [{"type": "text", "text": ...}, {"type": "image", "image":
    PIL.Image}]); text-only messages keep the plain string format.
    """
    turns: list[dict] = []  # [{"role": ..., "parts": [...]}]

    for record in records:
        role = _record_role(record)
        if role is None:
            continue

        parts = _payload_parts(record.get("payload", {}))
        if not parts:
            continue

        if turns and turns[-1]["role"] == role:
            _merge_parts(turns[-1]["parts"], parts)
        else:
            turns.append({"role": role, "parts": list(parts)})

    return [{"role": t["role"], "content": _finalize_content(t["parts"])} for t in turns]


def session_to_messages_codex(
    path: str | Path,
    output: str | Path | None = None,
) -> list[dict]:
    """
    Load a Codex session and convert it to the Transformers messages format.
    """
    records = load_session_codex(path)
    messages = records_to_messages_codex(records)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    return messages


def _record_type_of(record: dict) -> str:
    rtype = record.get("type", "unknown")
    payload = record.get("payload", {})
    sub = payload.get("type") if isinstance(payload, dict) else None
    return f"{rtype}:{sub}" if sub else rtype


def _timestamp_of(record: dict) -> str:
    return str(record.get("timestamp", ""))[:19].replace("T", " ")


def _total_tokens_of(records: list[dict]) -> dict[str, int]:
    """Codex reports cumulative usage, so keep the latest snapshot, not a sum."""
    latest: dict = {}
    for r in records:
        payload = r.get("payload", {})
        if isinstance(payload, dict) and payload.get("type") == "token_count":
            usage = payload.get("info", {}).get("total_token_usage")
            if usage:
                latest = usage
    if not latest:
        return {}
    return {
        "input_tokens": latest.get("input_tokens", 0),
        "output_tokens": latest.get("output_tokens", 0),
        "cached_input_tokens": latest.get("cached_input_tokens", 0),
        "total_tokens": latest.get("total_tokens", 0),
    }


CODEX_SCHEMA = InspectorSchema(
    label="CODEX",
    record_type_of=_record_type_of,
    is_message=lambda r: _record_role(r) is not None,
    role_of=lambda r: _record_role(r) or "other",
    text_of=lambda r: _payload_text(r.get("payload", {})),
    timestamp_of=_timestamp_of,
    total_tokens_of=_total_tokens_of,
)


def inspect_session_codex(
    path: str | Path,
    show_flow: bool = False,
    show_blocks: bool = False,
    show_raw: bool = False,
) -> None:
    """Print a color-coded inspection report for a Codex session."""
    records = load_session_codex(path)
    run_inspection(path, records, CODEX_SCHEMA, show_flow=show_flow, show_blocks=show_blocks)