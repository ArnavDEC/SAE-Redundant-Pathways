#!/usr/bin/env python3
"""Refresh every Google Doc the critique-task-idea skill reads, into llm/task_ideas/.

The fixed docs are re-exported every run; the meeting transcripts, which accumulate
without bound, only when Drive's copy is newer than ours.

What each doc gives a critique is in
`.claude/skills/critique-task-idea/references/sources.md`.
"""

import re
import sys
from pathlib import Path

import gdoc
from common import repo_root

DOC_IDS = {
    "task-vetting": "1XECMSNMhITvyYnINtFYNA7u6PbOJ7rv0Ak_a-hO7sMI",
    "scrie": "1ljFm4jbuxxIQK2QwRu8rcmsF6ObQJU4ZV2Xnj-Tgs_k",
    "pitch-docs": "1gY-46uMc2Pm46upxjwAAAaUjP2lniqDLyl-X5-cG2oM",
    "task-phases": "15bMzxfPuxtC8PUBrP5mMDm0Nu7PgtNnESuMRbw28d6g",
    "pqa-iqa-checklist": "1YuCaB6jnrvG3GpliHj1x7VDDA2dps33Ziot_-p3W4sA",
    "common-task-issues": "1fJLPO62GyxnriBOClirkfJoPS3ppw1M12QB5u0evfcE",
}
MEETINGS_QUERY = (
    "name contains 'Task Pitch Meeting' "
    "and name contains 'Notes by Gemini' "
    "and trashed=false"
)
DOCS_DIR = repo_root() / "llm/task_ideas"
TRANSCRIPTS_DIR = DOCS_DIR / "transcripts"


def cache_name(title: str) -> str:
    """`Task Pitch Meeting - Alex Folks - 2026/08/07 13:00 PDT - Notes by Gemini`
    becomes `2026-08-07-alex.local.md`."""
    date = re.search(r"(\d{4})/(\d{2})/(\d{2})", title)
    host = re.search(r"Meeting - (\w+)", title)
    if not date or not host:
        raise gdoc.GdocError(f"unexpected meeting title: {title}")
    return f"{'-'.join(date.groups())}-{host.group(1).lower()}.local.md"


def export_to(doc: str, out: Path) -> None:
    body = gdoc.export(doc)
    out.write_bytes(body)
    print(f"{out.name} ({len(body) // 1024} KiB)")


def main() -> None:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, doc in DOC_IDS.items():
        export_to(doc, DOCS_DIR / f"{name}.local.md")

    meetings = gdoc.search(MEETINGS_QUERY)
    for meeting in meetings:
        out = TRANSCRIPTS_DIR / cache_name(meeting.name)
        if out.exists() and meeting.modified.timestamp() <= out.stat().st_mtime:
            continue
        # One unreadable meeting shouldn't end the sweep.
        try:
            export_to(meeting.id, out)
        except gdoc.GdocError as e:
            print(e, file=sys.stderr)
    print(f"{len(meetings)} meetings in {TRANSCRIPTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except gdoc.GdocError as e:
        raise SystemExit(str(e)) from None
