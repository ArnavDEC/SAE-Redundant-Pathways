#!/usr/bin/env python3
"""Pick the pytest tests to run for the files a PR changes."""

import argparse
import json
import os
import subprocess

# Tests under a matching directory run only when that directory itself changed.
EXPLORATORY_DIRS = frozenset(
    {
        "prototyping",
        "experiments",
        "dev_scripts",
        "golden_solution",
        "agent_workspace",
        "llm",
    }
)
EXPLORATORY_SUFFIX = "-validation"

# Nothing imports from these, so each is its own unit and changes stay local.
STANDALONE_DIRS = frozenset({"docs", "interviews", ".claude"})


def _git(*args: str) -> list[str]:
    """Output lines, with git's stderr left to reach the job log on failure."""
    return subprocess.check_output(["git", *args], text=True).splitlines()


def is_test(path: str) -> bool:
    """Mirrors pytest's default python_files."""
    name = path.rsplit("/", 1)[-1]
    return name.endswith("_test.py") or (
        name.startswith("test_") and name.endswith(".py")
    )


def markers_of(tracked: list[str]) -> frozenset[str]:
    """Directories holding a Dockerfile or pyproject.toml."""
    split_paths = (path.rpartition("/") for path in tracked)
    return frozenset(
        parent
        for parent, _, name in split_paths
        if parent and (name.startswith("Dockerfile") or name == "pyproject.toml")
    )


def unit_of(path: str, markers: frozenset[str]) -> str:
    """The directory pytest should run `path` from."""
    parts = path.split("/")
    if parts[0] in STANDALONE_DIRS:
        return parts[0]
    # Anything not inside a subdirectory of tasks/ or llm/ runs from the repo root.
    if parts[0] not in ("tasks", "llm") or len(parts) < 3:
        return "."
    for end in range(len(parts) - 1, 1, -1):
        candidate = "/".join(parts[:end])
        if candidate in markers:
            return candidate
    # No marker anywhere above: the tasks/ or llm/ subdirectory itself.
    return "/".join(parts[:2])


def _touches(unit: str, changed: list[str], markers: frozenset[str]) -> bool:
    if unit == ".":
        return any(unit_of(path, markers) == "." for path in changed)
    # Prefix match: a change under a nested unit (tasks/x/docker) counts for tasks/x.
    return any(path.startswith(f"{unit}/") for path in changed)


def gate_of(test: str, unit: str) -> str | None:
    """The outermost exploratory directory containing `test`, if any."""
    prefix = "" if unit == "." else f"{unit}/"
    parts = test.removeprefix(prefix).split("/")[:-1]
    for i, segment in enumerate(parts):
        if segment in EXPLORATORY_DIRS or segment.endswith(EXPLORATORY_SUFFIX):
            return prefix + "/".join(parts[: i + 1])
    return None


def plan(
    changed: list[str], all_tests: list[str], markers: frozenset[str]
) -> list[dict[str, str]]:
    """Matrix entries: the unit directory, and the test files to run inside it."""
    units: dict[str, list[str]] = {}
    for test in all_tests:
        unit = unit_of(test, markers)
        if not _touches(unit, changed, markers):
            continue
        gate = gate_of(test, unit)
        if gate is not None and not _touches(gate, changed, markers):
            continue
        units.setdefault(unit, []).append(test.removeprefix(f"{unit}/"))
    # One process per unit, so tasks' grader.py modules can't clobber each other.
    return [
        {"dir": unit, "files": " ".join(files)} for unit, files in sorted(units.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick the pytest tests to run for the files a PR changes."
    )
    parser.add_argument("base", help="base ref to diff against, e.g. origin/main")
    base = parser.parse_args().base
    os.chdir(_git("rev-parse", "--show-toplevel")[0])
    tracked = _git("ls-files")
    markers = markers_of(tracked)
    all_tests = [path for path in tracked if is_test(path)]
    changed = _git("diff", "--name-only", f"{base}...HEAD")
    print(json.dumps(plan(changed, all_tests, markers)))


if __name__ == "__main__":
    main()
