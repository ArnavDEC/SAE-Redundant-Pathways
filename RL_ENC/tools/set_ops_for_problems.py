#!/usr/bin/env python3
"""Set operations on problems*.txt files.

Usage:
    problem_sets.py union   -o OUT FILE [FILE ...]
    problem_sets.py diff    -o OUT FILE1 FILE2
    problem_sets.py intersect -o OUT FILE [FILE ...]
"""

import argparse
import sys
from pathlib import Path


def read_problems(path: str) -> list[str]:
    """Read a problems file, returning lines in order (stripped, no blanks)."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def write_problems(path: str, items: list[str]) -> None:
    Path(path).write_text("\n".join(items) + "\n" if items else "")
    print(f"Wrote {len(items)} items to {path}", file=sys.stderr)


def stable_union(files: list[str]) -> list[str]:
    """Union preserving first-seen order."""
    seen = set()
    result = []
    for f in files:
        for item in read_problems(f):
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def stable_diff(file_a: str, file_b: str) -> list[str]:
    """Items in A but not in B, preserving A's order."""
    exclude = set(read_problems(file_b))
    return [item for item in read_problems(file_a) if item not in exclude]


def stable_intersect(files: list[str]) -> list[str]:
    """Intersection preserving first file's order."""
    if not files:
        return []
    sets = [set(read_problems(f)) for f in files]
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return [item for item in read_problems(files[0]) if item in common]


def main() -> None:
    parser = argparse.ArgumentParser(description="Set operations on problems*.txt files")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_union = sub.add_parser("union", help="Set union of files")
    p_union.add_argument("-o", "--output", required=True, help="Output path")
    p_union.add_argument("files", nargs="+", help="Input files")

    p_diff = sub.add_parser("diff", help="Set difference (A - B)")
    p_diff.add_argument("-o", "--output", required=True, help="Output path")
    p_diff.add_argument("file_a", help="File A")
    p_diff.add_argument("file_b", help="File B (items to remove)")

    p_inter = sub.add_parser("intersect", help="Set intersection of files")
    p_inter.add_argument("-o", "--output", required=True, help="Output path")
    p_inter.add_argument("files", nargs="+", help="Input files")

    args = parser.parse_args()

    if args.cmd == "union":
        result = stable_union(args.files)
    elif args.cmd == "diff":
        result = stable_diff(args.file_a, args.file_b)
    elif args.cmd == "intersect":
        result = stable_intersect(args.files)

    write_problems(args.output, result)


if __name__ == "__main__":
    main()
