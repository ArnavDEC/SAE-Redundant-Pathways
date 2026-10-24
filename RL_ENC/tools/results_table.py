#!/usr/bin/env -S uv run

import argparse
import docker
import docker.errors
import json
from pathlib import Path

from tabulate import tabulate

from docker_helpers import get_container_file_contents

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runfile", type=Path)
    args = parser.parse_args()

    with args.runfile.open('r') as f:
        containers = [docker.from_env().containers.get(line.strip()) for line in f]

    table_entries: list[tuple[str, str, str]] = []
    for container in containers:
        taskname = get_container_file_contents(container, 
                                               Path("/taskname.txt")).decode('utf-8')
        try:
            grade_obj = json.loads(
                    get_container_file_contents(container,
                                                Path("/grader/grade.json"))
                    .decode('utf-8'))
            if grade_obj.get("timeout"):
                grade_str = "Timeout"
            else:
                grade_str = str(grade_obj["score"])

        except docker.errors.NotFound:
            if container.status == "exited":
                grade_str = "Grader Error"
            else:
                grade_str = "Running"
        except json.JSONDecodeError:
            grade_str = "Bad grade file"
        table_entries.append((taskname, container.short_id, grade_str))

    table_entries.sort()

    print(tabulate(table_entries,
                   headers=["Task", "Container ID", "Grade"],
                   tablefmt="fancy_grid"))
if __name__ == "__main__":
    main()
