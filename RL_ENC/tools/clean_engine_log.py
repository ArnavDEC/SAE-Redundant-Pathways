#!/usr/bin/env python3
"""Normalize and clean engine logs by removing ANSI and noisy TUI lines."""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path
from typing import Iterable, TextIO

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)

TUI_SUBSTRINGS = (
    "Claude Code",
    "bypass permissions",
    "Esc to interrupt",
    "shift+Tab to cycle",
    "docs.anthropic.com/en/docs/claude-code",
    "Anthropic marketplace",
    "Run `claude install`",
    "ctrl+b ctrl+b (twice) to run in background",
    "\u2234 Thinking",
    "Thinking\u2026",
)

SHORT_WORD_ALLOWLIST = {
    "ok",
    "no",
    "yes",
    "true",
    "false",
    "none",
    "null",
    "done",
    "pass",
    "fail",
    "skip",
    "cuda",
    "cpu",
    "gpu",
    "info",
    "warn",
    "error",
    "debug",
    "trace",
    "json",
    "yaml",
    "yml",
}


def strip_leading_symbols(value: str) -> str:
    index = 0
    while index < len(value) and not value[index].isalnum():
        index += 1
    return value[index:]


def is_box_drawing_line(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    return all(0x2500 <= ord(ch) <= 0x257F for ch in stripped)


def is_tui_line(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return value != ""
    if stripped.endswith(" tokens"):
        parts = stripped.split()
        if len(parts) == 2 and parts[0].isdigit():
            return True
    if "inking)" in stripped or "thought for" in stripped:
        return True
    if "tokens" in stripped and ("thinking" in stripped or "thought" in stripped):
        return True
    if stripped.isdigit() and len(stripped) <= 2:
        return True
    # Keep this heuristic conservative: filtering 3-4 letter words causes
    # false positives on real model output (e.g. "loss", "grad").
    if (
        stripped.isalpha()
        and len(stripped) <= 2
        and stripped.lower() not in SHORT_WORD_ALLOWLIST
    ):
        return True
    if not any(ch.isspace() for ch in stripped):
        if len(stripped) <= 8 and any(not ch.isalnum() for ch in stripped):
            return True
    if stripped == "\u276f":
        return True
    if is_box_drawing_line(value):
        return True
    normalized = strip_leading_symbols(stripped)
    if normalized.startswith("Elucidating"):
        return True
    for token in TUI_SUBSTRINGS:
        if token in value:
            return True
    return False


def clean_engine_log_text(text: str) -> str:
    ends_with_newline = text.endswith("\n") or text.endswith("\r\n")
    text = OSC_RE.sub("", text)
    text = ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n")

    lines = text.split("\n")
    if ends_with_newline and lines and lines[-1] == "":
        lines = lines[:-1]

    normalized = [
        max(value.split("\r"), key=lambda segment: (len(segment.strip()), len(segment)))
        for value in lines
    ]

    filtered: list[str] = []
    last_was_blank = False
    for value in normalized:
        if is_tui_line(value):
            continue
        if value == "":
            if last_was_blank:
                continue
            last_was_blank = True
        else:
            last_was_blank = False
        filtered.append(value)

    while filtered and filtered[-1] == "":
        filtered.pop()

    output = "\n".join(filtered)
    if ends_with_newline:
        output += "\n"
    return output


def iter_cleaned_stream(lines: Iterable[str]) -> Iterable[str]:
    last_was_blank = False
    for line in lines:
        line = OSC_RE.sub("", line)
        line = ANSI_RE.sub("", line)
        newline = line.endswith("\n")
        if newline:
            line = line[:-1]
        if line.endswith("\r"):
            line = line[:-1]
        if "\r" in line:
            segments = line.split("\r")
            line = max(segments, key=lambda segment: (len(segment.strip()), len(segment)))
        if is_tui_line(line):
            continue
        if line == "":
            if last_was_blank:
                continue
            last_was_blank = True
        else:
            last_was_blank = False
        if newline:
            yield line + "\n"
        else:
            yield line


def _read_input(path: Path | None) -> str:
    if path is None:
        return sys.stdin.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _open_stream_input(path: Path | None) -> TextIO:
    if path is None:
        return io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def _open_output(path: Path | None) -> TextIO:
    if path is None:
        return sys.stdout
    return path.open("w", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean rl-env engine logs")
    parser.add_argument("--stream", action="store_true", help="Clean incrementally for streaming input")
    parser.add_argument("--input", type=Path, default=None, help="Optional input file path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.stream:
        infile = _open_stream_input(args.input)
        outfile = _open_output(args.output)
        try:
            for line in iter_cleaned_stream(infile):
                outfile.write(line)
                outfile.flush()
        finally:
            if args.input is not None:
                infile.close()
            if args.output is not None:
                outfile.close()
        return 0

    raw_text = _read_input(args.input)
    cleaned = clean_engine_log_text(raw_text)
    if args.output is None:
        sys.stdout.write(cleaned)
        return 0
    args.output.write_text(cleaned, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
