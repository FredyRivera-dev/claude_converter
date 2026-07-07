from __future__ import annotations

from pathlib import Path
from typing import Literal

from claude_converter.claude import (
    inspect_session as inspect_session_claude,
)
from claude_converter.claude import (
    load_session as load_session_claude,
)
from claude_converter.claude import (
    records_to_messages as records_to_messages_claude,
)
from claude_converter.claude import (
    session_to_messages as session_to_messages_claude,
)
from claude_converter.codex import (
    inspect_session_codex,
    load_session_codex,
    records_to_messages_codex,
    session_to_messages_codex,
)
from claude_converter.pi import (
    inspect_session_pi,
    load_session_pi,
    records_to_messages_pi,
    session_to_messages_pi,
)

ConverterName = Literal["claude-code", "codex", "pi"]


class Converter:
    """
    Thin dispatcher over the per-tool converters (claude-code, codex, pi).
    Each method just routes to the matching tool-specific function, so the
    public API stays the same regardless of which tool produced the session.
    """

    def __init__(self, converter: ConverterName = "claude-code"):
        self.converter = converter

    def load_session(self, path: str | Path) -> list[dict]:
        match self.converter:
            case "claude-code":
                return load_session_claude(path)
            case "codex":
                return load_session_codex(path)
            case "pi":
                return load_session_pi(path)
            case _:
                raise ValueError(f"Unknown converter: {self.converter!r}")

    def records_to_messages(self, records: list[dict]) -> list[dict]:
        match self.converter:
            case "claude-code":
                return records_to_messages_claude(records)
            case "codex":
                return records_to_messages_codex(records)
            case "pi":
                return records_to_messages_pi(records)
            case _:
                raise ValueError(f"Unknown converter: {self.converter!r}")

    def session_to_messages(
        self,
        path: str | Path,
        output: str | Path | None = None,
    ) -> list[dict]:
        match self.converter:
            case "claude-code":
                return session_to_messages_claude(path, output=output)
            case "codex":
                return session_to_messages_codex(path, output=output)
            case "pi":
                return session_to_messages_pi(path, output=output)
            case _:
                raise ValueError(f"Unknown converter: {self.converter!r}")

    def inspect_session(
        self,
        path: str | Path,
        show_flow: bool = False,
        show_blocks: bool = False,
        show_raw: bool = False,
    ) -> None:
        match self.converter:
            case "claude-code":
                return inspect_session_claude(
                    path, show_flow=show_flow, show_blocks=show_blocks, show_raw=show_raw
                )
            case "codex":
                return inspect_session_codex(
                    path, show_flow=show_flow, show_blocks=show_blocks, show_raw=show_raw
                )
            case "pi":
                return inspect_session_pi(
                    path, show_flow=show_flow, show_blocks=show_blocks, show_raw=show_raw
                )
            case _:
                raise ValueError(f"Unknown converter: {self.converter!r}")