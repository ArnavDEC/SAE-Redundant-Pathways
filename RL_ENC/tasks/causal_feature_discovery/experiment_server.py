#!/usr/bin/env python3
"""Root-only server implementing the black-box experiment API."""

import json
import socketserver
import sys
import threading
from pathlib import Path

sys.path.insert(0, "/grader")
from private_model import (  # noqa: E402
    Episode,
    MAX_CALLS,
    MAX_INPUTS,
    MAX_REPORTED_FEATURES,
)

SOCKET_PATH = Path("/tmp/causal-feature-lab.sock")


class ExperimentService:
    def __init__(self) -> None:
        episode_data = json.loads(Path("/grader/episode_seed.json").read_text())
        self.episode = Episode(episode_data["seed"])
        self.calls = 0
        self.lock = threading.Lock()

    def handle(self, request: dict[str, object]) -> dict[str, object]:
        command = request.get("command")
        if not isinstance(command, str):
            return {"error": "command must be a string"}
        if command == "list-features":
            return {"feature_ids": list(range(self.episode.feature_count))}

        if command not in {"observe", "ablate", "steer", "patch", "factorial"}:
            return {"error": "unknown experiment command"}

        count = request.get("count", 32)
        max_count = MAX_INPUTS // 4 if command == "factorial" else MAX_INPUTS
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= max_count:
            if command == "factorial":
                return {"error": f"count must be from 1 to {max_count} per condition"}
            return {"error": f"count must be an integer from 1 to {MAX_INPUTS}"}

        feature_ids = request.get("feature_ids")
        needs_feature_ids = command in {"ablate", "steer", "patch"}
        if needs_feature_ids or (command == "factorial" and feature_ids is not None):
            if (
                not isinstance(feature_ids, list)
                or not feature_ids
                or len(feature_ids) > MAX_REPORTED_FEATURES
                or any(
                    not isinstance(feature_id, int) or isinstance(feature_id, bool)
                    for feature_id in feature_ids
                )
            ):
                return {
                    "error": (
                        "feature_ids must contain 1 to "
                        f"{MAX_REPORTED_FEATURES} integer feature IDs"
                    )
                }
            if len(set(feature_ids)) != len(feature_ids):
                return {"error": "duplicate feature IDs are not allowed"}
            try:
                for feature_id in feature_ids:
                    self.episode._check_feature(feature_id)
            except ValueError as exc:
                return {"error": str(exc)}

        distribution = request.get("distribution")
        if command in {"observe", "ablate"}:
            if not isinstance(distribution, str):
                return {"error": "distribution is required"}
            try:
                self.episode._check_distribution(distribution)
            except ValueError as exc:
                return {"error": str(exc)}

        scale = request.get("scale")
        if command == "steer":
            if not isinstance(scale, (int, float)) or isinstance(scale, bool):
                return {"error": "scale must be numeric"}
            if not -5 <= scale <= 5:
                return {"error": "scale must be between -5 and 5"}

        with self.lock:
            if self.calls >= MAX_CALLS:
                return {"error": f"Experiment budget exhausted ({MAX_CALLS} calls)"}
            self.calls += 1
            nonce = self.calls

        try:
            if command == "observe":
                return self.episode.observe(distribution, count, nonce)
            if command == "factorial":
                return self.episode.factorial(count, nonce, feature_ids)
            if command == "ablate":
                return self.episode.ablate(feature_ids, distribution, count, nonce)
            if command == "steer":
                return self.episode.steer(feature_ids, float(scale), count, nonce)
            return self.episode.patch(feature_ids, count, nonce)
        except ValueError as exc:
            return {"error": str(exc)}


SERVICE = ExperimentService()


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(65537)
        if len(raw) > 65536:
            response = {"error": "request is too large"}
        else:
            try:
                request = json.loads(raw)
            except json.JSONDecodeError as exc:
                response = {"error": f"invalid JSON request: {exc.msg}"}
            else:
                if not isinstance(request, dict):
                    response = {"error": "request must be a JSON object"}
                else:
                    response = SERVICE.handle(request)
        self.wfile.write((json.dumps(response) + "\n").encode())


class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


if SOCKET_PATH.exists():
    SOCKET_PATH.unlink()
with ThreadedUnixServer(str(SOCKET_PATH), RequestHandler) as server:
    SOCKET_PATH.chmod(0o666)
    server.serve_forever()
