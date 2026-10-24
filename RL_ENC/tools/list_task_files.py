import argparse
import pathspec
import sys
from pathlib import Path
from common import dockerignore_patterns, repo_root

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    args = parser.parse_args()

    task_dir = repo_root() / "tasks" / args.task
    spec = pathspec.PathSpec.from_lines('gitwildmatch',
                                        dockerignore_patterns(task_dir))

    def print_recursive(current_dir: Path, rel_dir: Path) -> None:
        for item in current_dir.iterdir():
            rel_path = rel_dir / item.name
            if spec.match_file(str(rel_path)):
                continue
            if item.is_file():
                print(item)
            elif item.is_dir():
                print_recursive(item, rel_path)
    try:
        print_recursive(task_dir, Path('.'))
    # Handle this being piped to something that exits without reading all lines, like `less`.
    except BrokenPipeError:
        sys.exit(0)

if __name__ == "__main__":
    main()
