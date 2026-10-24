#!/usr/bin/env -S uv run
# PYTHON_ARGCOMPLETE_OK
"""Completion helper for run.sh.

This script provides argcomplete completions for run.sh, which is a bash
wrapper that calls `task_runner.py run`. Since run.sh implicitly adds the
'run' subcommand, this script exposes only the run subcommand's arguments.
This script is not meant to ever be run directly, as it doesn't do anything.
"""

import argcomplete
import argparse

from task_runner import add_run_arguments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tasks with Claude or Codex",
    )
    add_run_arguments(parser)
    argcomplete.autocomplete(parser)
    parser.parse_args()


if __name__ == "__main__":
    main()
