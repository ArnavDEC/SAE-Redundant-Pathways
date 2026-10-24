#!/usr/bin/env python3
"""Client for the episode's constrained, root-owned experiment service."""

import argparse
import json
import socket
import sys

SOCKET_PATH = "/tmp/causal-feature-lab.sock"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-features")

    observe_parser = subparsers.add_parser("observe")
    observe_parser.add_argument(
        "--distribution", choices=("train", "decorrelated", "long"), required=True
    )
    observe_parser.add_argument("--count", type=int, default=32)

    factorial_parser = subparsers.add_parser(
        "factorial",
        help="compare target and second-concept presence in four matched conditions",
    )
    factorial_parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="paired inputs per condition (maximum 16)",
    )
    factorial_parser.add_argument(
        "--feature-ids",
        type=int,
        nargs="+",
        help="optionally report conditional joint ablation effects for these IDs",
    )

    ablate_parser = subparsers.add_parser("ablate")
    ablate_parser.add_argument("--feature-ids", type=int, nargs="+", required=True)
    ablate_parser.add_argument(
        "--distribution", choices=("train", "decorrelated", "long"), required=True
    )
    ablate_parser.add_argument("--count", type=int, default=32)

    steer_parser = subparsers.add_parser("steer")
    steer_parser.add_argument("--feature-ids", type=int, nargs="+", required=True)
    steer_parser.add_argument("--scale", type=float, required=True)
    steer_parser.add_argument("--count", type=int, default=32)

    patch_parser = subparsers.add_parser("patch")
    patch_parser.add_argument("--feature-ids", type=int, nargs="+", required=True)
    patch_parser.add_argument("--count", type=int, default=32)

    args = parser.parse_args()
    request = vars(args)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(15)
            client.connect(SOCKET_PATH)
            client.sendall((json.dumps(request) + "\n").encode())
            response = client.makefile("r").readline()
    except (OSError, TimeoutError) as exc:
        print(f"Experiment service unavailable: {exc}", file=sys.stderr)
        return 2

    if not response:
        print("Experiment service returned an empty response", file=sys.stderr)
        return 2
    result = json.loads(response)
    print(json.dumps(result, sort_keys=True))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
