#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./tools/view-log.sh <container_id> <entrypoint|engine|grader>

Options:
  -h, --help    Show this help message

For cleaned engine logs, use:
  ./tools/view-clean-log.sh <container_id> [--save-clean]
EOF
}

if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
  usage
  exit 0
fi

if [ "$#" -ne 2 ]; then
  usage >&2
  exit 1
fi

CONTAINER_ID="$1"
LOG_NAME="$2"

valid_logs="entrypoint engine grader"
if [[ ! ${valid_logs} =~ (^|[[:space:]])${LOG_NAME}($|[[:space:]]) ]]; then
  echo "Invalid log name: valid logs are ${valid_logs}" >&2
  exit 1
fi

docker cp "${CONTAINER_ID}:/logs/${LOG_NAME}.log" - | tar -xO | less -R
