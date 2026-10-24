#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./tools/watch-run.sh [container_id]

Watch the rollout tmux session for a container.

Options:
  -h, --help    Show this help message
EOF
}

CONTAINER_ID=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [ -n "${CONTAINER_ID}" ]; then
        echo "Too many positional arguments. Provide at most one container id." >&2
        usage >&2
        exit 1
      fi
      CONTAINER_ID="$1"
      ;;
  esac
  shift
 done

if [ -z "${CONTAINER_ID}" ]; then
  CONTAINER_ID=$(docker ps -n1 -q)
fi
if [ -z "${CONTAINER_ID}" ]; then
  echo "No running containers found. Provide a container id." >&2
  exit 1
fi

RUNNING_STATE=$(docker inspect -f '{{.State.Running}}' "${CONTAINER_ID}" 2>/dev/null || echo "false")
if [ "${RUNNING_STATE}" != "true" ]; then
  echo "Container ${CONTAINER_ID} is not running."
  echo "Use ./tools/view-log.sh ${CONTAINER_ID} engine to inspect raw logs."
  echo "Use ./tools/view-clean-log.sh ${CONTAINER_ID} to inspect cleaned logs."
  echo "Use ./tools/view-clean-log.sh ${CONTAINER_ID} --save-clean to save cleaned logs."
  exit 0
fi

docker exec -it "${CONTAINER_ID}" tmux a -t rl-agent
