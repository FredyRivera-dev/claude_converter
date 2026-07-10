from __future__ import annotations

import base64
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
MAGENTA = "\033[35m"

DEFAULT_ROLE_COLORS = {"user": GREEN, "assistant": CYAN, "system": YELLOW}


def _c(text: object, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET


def _hr(char: str = "-", width: int = 70) -> str:
    return _c(char * width, DIM)


def _truncate(text: object, max_len: int = 120) -> str:
    text = str(text).replace("\n", "\\n")
    return text[:max_len] + _c("...", DIM) if len(text) > max_len else text


def load_jsonl(path: str | Path) -> list[dict]:
    """
    Load any newline-delimited JSON session file into a list of dicts.

    This loader is format-agnostic: it does not know anything about
    Claude Code, Codex, or Pi. Each tool-specific converter is responsible
    for interpreting the records it returns.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the extension is not .jsonl/.json, or the file has
                    no valid records.
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


def convert_base64_to_pil_image(uri: str):
    """
    Decode a base64 image (a bare base64 string, or a data URI like
    "data:image/png;base64,....") into a PIL.Image.Image, ready to drop
    into a `transformers` multimodal content part:
        {"type": "image", "image": convert_base64_to_pil_image(uri)}

    Pillow is imported lazily here, not at module level: this package is
    otherwise zero-dependency, and most sessions never touch images, so
    importing `claude_converter` shouldn't require installing Pillow.

    Raises:
        ImportError: if Pillow is not installed.
        ValueError:  if the payload isn't a decodable image.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required to decode embedded images. "
            "Install it with: pip install claude-converter[images]"
        ) from e

    b64 = uri.split(",", 1)[1] if "," in uri else uri

    try:
        return Image.open(BytesIO(base64.b64decode(b64)))
    except Exception as e:
        raise ValueError(f"Could not decode base64 image: {e}") from e


def _pil_to_data_uri(image) -> str:
    """Re-encode a PIL.Image back into a base64 data URI, for JSON dumps."""
    fmt = (getattr(image, "format", None) or "PNG").upper()
    buf = BytesIO()
    try:
        image.save(buf, format=fmt)
    except Exception:
        fmt = "PNG"
        buf = BytesIO()
        image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def messages_to_json_safe(messages: list[dict]) -> list[dict]:
    """
    Return a copy of `messages` safe to pass to json.dump(). Every
    converter's records_to_messages_* can put a real PIL.Image object
    inside a multimodal content part ({"type": "image", "image": ...}),
    which json.dump cannot serialize on its own (raises
    "TypeError: Object of type PngImageFile is not JSON serializable").

    Only used for the on-disk `output=...` path: the in-memory list
    returned to the caller keeps real PIL.Image objects untouched, since
    that's what `transformers` processors expect when building tensors
    directly. Images are re-encoded here as base64 data URIs under the
    same "image" key `transformers` chat templates already recognize as
    a base64 source. Confirmed directly against `transformers`' source
    (image_utils.py, load_image()): it explicitly strips a "data:image/"
    prefix up to the first comma before base64-decoding, so the
    "data:image/png;base64,<b64>" string produced by _pil_to_data_uri is
    exactly what load_image() expects — not a custom/guessed format.
    """
    def convert_content(content):
        if not isinstance(content, list):
            return content
        converted = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image" and "image" in part:
                image = part["image"]
                try:
                    from PIL import Image as PILImage
                except ImportError:
                    converted.append({"type": "image", "image": repr(image)})
                    continue
                if isinstance(image, PILImage.Image):
                    converted.append({"type": "image", "image": _pil_to_data_uri(image)})
                    continue
            converted.append(part)
        return converted

    return [{**m, "content": convert_content(m.get("content"))} for m in messages]


def merge_content_parts(existing: list[dict], new: list[dict]) -> None:
    """
    Extend `existing` in place with `new` multimodal content parts,
    joining adjacent text parts instead of leaving separate fragments.
    Shared by every converter's records_to_messages_*, since "merge
    consecutive same-role turns" is the same operation regardless of
    which tool produced the session.
    """
    for part in new:
        if existing and existing[-1]["type"] == "text" and part["type"] == "text":
            existing[-1] = {"type": "text", "text": existing[-1]["text"] + "\n" + part["text"]}
        else:
            existing.append(dict(part))


def finalize_content(parts: list[dict]) -> str | list[dict]:
    """
    Collapse a parts list to a plain string when there are no images, so
    text-only turns keep the exact "content": str shape used before
    multimodal support existed. Only keep the list-of-parts form (the
    shape `transformers` multimodal chat templates expect) when at least
    one non-text part (an image) is present.
    """
    if all(p["type"] == "text" for p in parts):
        return "\n".join(p["text"] for p in parts)
    return parts


def dget(d: object, key: str) -> dict:
    """
    Safe "get a nested dict" accessor. Like `d.get(key, {})`, but also
    falls back to {} when the key is *present* with an explicit `null`,
    since dict.get's default only kicks in when the key is absent.

    Real coding-agent session JSONL files have fields that are sometimes
    an object and sometimes null depending on the event type (observed in
    the wild: a Codex token_count event with "info": null, which crashed
    with AttributeError: 'NoneType' object has no attribute 'get'). This
    guards every chained `.get(...).get(...)` against that failure mode.
    """
    if not isinstance(d, dict):
        return {}
    value = d.get(key)
    return value if isinstance(value, dict) else {}


def dlist(d: object, key: str) -> list:
    """Same idea as dget, but for fields expected to be a list (falls
    back to [] on an explicit null, a missing key, or a non-dict parent)."""
    if not isinstance(d, dict):
        return []
    value = d.get(key)
    return value if isinstance(value, list) else []


@dataclass
class InspectorSchema:
    """
    Describes how to pull generic fields out of a tool-specific record,
    so the same report/printing logic can be reused for any converter.

    Attributes:
        label:           Display name for the report header (e.g. "CODEX").
        record_type_of:  Returns a display label for the record's raw type.
        is_message:      Whether a record represents a conversation turn.
        role_of:         Returns the turn's role (user/assistant/etc.).
        text_of:         Flattens a record's content to plain text.
        timestamp_of:    Returns a short display timestamp.
        role_colors:     Maps role -> ANSI color.
        total_tokens_of: Optional; computes a token-usage summary dict
                         from the full record list.
        block_types_of:  Optional; returns the list of content-block type
                         names for a record (e.g. ["text", "tool_use"]).
                         Enables the per-role block-type breakdown report.
        blocks_of:       Optional; returns the raw content blocks for a
                         record. Enables inline block-by-block rendering
                         in the conversation flow (show_blocks=True).
        render_block:    Optional; renders a single raw block to a
                         colored, indented display string. Required
                         together with blocks_of for inline rendering.
    """

    label: str
    record_type_of: Callable[[dict], str]
    is_message: Callable[[dict], bool]
    role_of: Callable[[dict], str]
    text_of: Callable[[dict], str]
    timestamp_of: Callable[[dict], str] = lambda r: ""
    role_colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROLE_COLORS))
    total_tokens_of: Optional[Callable[[list[dict]], dict[str, int]]] = None
    tokens_of: Optional[Callable[[dict], int]] = None
    block_types_of: Optional[Callable[[dict], list[str]]] = None
    blocks_of: Optional[Callable[[dict], list[dict]]] = None
    render_block: Optional[Callable[[dict], str]] = None


def print_header(path: Path, records: list[dict], schema: InspectorSchema) -> None:
    print()
    print(_hr("="))
    print(_c(f"  {schema.label} SESSION INSPECTOR", BOLD + CYAN))
    print(_hr("="))
    print(f"  File  : {_c(str(path), BOLD)}")
    print(f"  Size  : {_c(f'{path.stat().st_size / 1024:.1f} KB', BOLD)}")
    print(f"  Lines : {_c(len(records), BOLD)}")


def _scaled_bar(count: int, max_count: int, width: int = 40) -> str:
    """Bar length proportional to the largest count in the set, not a fixed cap."""
    if max_count <= 0:
        return ""
    length = max(1, round((count / max_count) * width))
    return "█" * length


def print_record_types(records: list[dict], schema: InspectorSchema) -> None:
    print()
    print(_c("  RECORD TYPES", BOLD))
    print(_hr())

    counts = Counter(schema.record_type_of(r) for r in records)
    max_count = counts.most_common(1)[0][1] if counts else 0
    for rtype, count in counts.most_common():
        cf = schema.role_colors.get(rtype, DIM)
        bar = _scaled_bar(count, max_count)
        print(f"  {_c(rtype.ljust(28), cf)} {_c(count, BOLD):>4}  {_c(bar, cf)}")


def print_block_summary(records: list[dict], schema: InspectorSchema) -> None:
    """
    Content-block breakdown per role (e.g. how many 'text' vs 'tool_use'
    blocks each role produced). No-op if the schema doesn't provide
    block_types_of, since not every format exposes nested content blocks.
    """
    if schema.block_types_of is None:
        return

    by_role: dict[str, Counter] = {}
    for r in records:
        role = schema.role_of(r) if schema.is_message(r) else schema.record_type_of(r)
        counter = by_role.setdefault(role, Counter())
        for btype in schema.block_types_of(r):
            counter[btype] += 1

    if not by_role:
        return

    print()
    print(_c("  CONTENT BLOCKS (by record type)", BOLD))
    print(_hr())

    for role, counter in sorted(by_role.items()):
        if not counter:
            continue
        cf = schema.role_colors.get(role, DIM)
        print(f"  {_c(role, cf + BOLD)}")
        for btype, cnt in counter.most_common():
            print(f"    {_c('·', DIM)} {btype.ljust(20)} {_c(cnt, BOLD)}")


def print_message_roles(records: list[dict], schema: InspectorSchema) -> None:
    messages = [r for r in records if schema.is_message(r)]
    if not messages:
        return

    print()
    print(_c("  MESSAGE ROLES", BOLD))
    print(_hr())

    counts = Counter(schema.role_of(r) for r in messages)
    max_count = counts.most_common(1)[0][1] if counts else 0
    for role, count in counts.most_common():
        cf = schema.role_colors.get(role, DIM)
        bar = _scaled_bar(count, max_count)
        print(f"  {_c(role.ljust(12), cf)} {_c(count, BOLD):>4}  {_c(bar, cf)}")


def print_token_usage(records: list[dict], schema: InspectorSchema) -> None:
    if schema.total_tokens_of is None:
        return

    totals = schema.total_tokens_of(records)
    if not totals:
        return

    print()
    print(_c("  TOKEN USAGE", BOLD))
    print(_hr())
    for key, value in totals.items():
        label = key.replace("_", " ").title().ljust(20)
        print(f"  {label} {_c(f'{value:,}', BOLD)}")


def print_flow(records: list[dict], schema: InspectorSchema, show_text: bool = False) -> None:
    messages = [r for r in records if schema.is_message(r)]
    if not messages:
        return

    print()
    print(_c("  CONVERSATION FLOW", BOLD))
    print(_hr())

    can_render_blocks = schema.blocks_of is not None and schema.render_block is not None

    for i, r in enumerate(messages):
        role = schema.role_of(r)
        cf = schema.role_colors.get(role, DIM)
        ts = schema.timestamp_of(r)
        tk_label = ""
        if schema.tokens_of is not None:
            tokens = schema.tokens_of(r)
            tk_label = _c(f"{tokens}tk", DIM) if tokens else ""

        print(f"  {i + 1:>3}. {_c(f'[{role.upper()}]'.ljust(14), cf + BOLD)} {_c(ts, DIM)} {tk_label}")

        if schema.block_types_of is not None:
            print(f"       {_c('blocks: ' + ', '.join(schema.block_types_of(r)), DIM)}")

        if show_text and can_render_blocks:
            for block in schema.blocks_of(r):
                print(schema.render_block(block))
        elif show_text:
            print(f"       {_truncate(schema.text_of(r), 200)}")

        print()


def print_raw_examples(records: list[dict], schema: InspectorSchema) -> None:
    """One raw record example per record type found, format-agnostic."""
    print()
    print(_c("  RAW RECORD EXAMPLES (one per type)", BOLD))
    print(_hr())

    seen: set[str] = set()
    for r in records:
        rtype = schema.record_type_of(r)
        if rtype in seen:
            continue
        seen.add(rtype)

        print(_c(f"  type = {rtype}", BOLD + YELLOW))
        print(_c("  top-level keys:", DIM))
        print(f"    {_c(list(r.keys()), DIM)}")

        if schema.is_message(r):
            print(_c("  flattened text:", DIM))
            print(f"    {_truncate(schema.text_of(r), 150)}")

        print(_hr("."))


def run_inspection(
    path: str | Path,
    records: list[dict],
    schema: InspectorSchema,
    show_flow: bool = False,
    show_blocks: bool = False,
    extra_header: Optional[Callable[[list[dict]], None]] = None,
) -> None:
    """
    Generic inspection report, usable by any converter that provides an
    InspectorSchema. show_blocks only takes effect when show_flow is True.

    extra_header: optional callback invoked right after the header block,
    for format-specific metadata (e.g. Claude Code's sessionId/cwd/branch)
    that doesn't generalize across formats.
    """
    path = Path(path)

    print_header(path, records, schema)
    if extra_header is not None:
        extra_header(records)
    print(_hr("="))

    print_record_types(records, schema)
    print_block_summary(records, schema)
    print_message_roles(records, schema)
    print_token_usage(records, schema)

    if show_flow:
        print_flow(records, schema, show_text=show_blocks)

    print(_hr("="))
    print()