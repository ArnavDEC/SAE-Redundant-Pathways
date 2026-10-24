#!/usr/bin/env python3
"""Fetch a Google Doc as markdown, using your gcloud credentials."""

import argparse
import re
import sys
from pathlib import Path

import gdoc


def doc_id(ref: str) -> str:
    """Accept either a full docs.google.com URL or a bare document ID."""
    match = re.search(r"/document/(?:u/\d+/)?d/([\w-]+)", ref)
    return match.group(1) if match else ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc", help="Google Doc URL or bare document ID")
    parser.add_argument("-o", "--out", help="write here instead of stdout")
    args = parser.parse_args()

    body = gdoc.export(doc_id(args.doc))
    if args.out:
        Path(args.out).write_bytes(body)
    else:
        sys.stdout.buffer.write(body)


if __name__ == "__main__":
    try:
        main()
    except gdoc.GdocError as e:
        raise SystemExit(str(e)) from None
