#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./tools/view-clean-log.sh <container_id> [--save-clean]

Render cleaned engine logs for a container.

Options:
  --save-clean  Save cleaned engine output to runs/<container_id>-engine.clean.log
  -h, --help    Show this help message
EOF
}

if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
  usage
  exit 0
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage >&2
  exit 1
fi

CONTAINER_ID="$1"
SAVE_CLEAN=0

if [ "$#" -eq 2 ]; then
  case "$2" in
    --save-clean)
      SAVE_CLEAN=1
      ;;
    *)
      echo "Unknown option: $2" >&2
      usage >&2
      exit 1
      ;;
  esac
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLEAN_LOG_PATH="${REPO_ROOT}/runs/${CONTAINER_ID}-engine.clean.log"

if [ "${SAVE_CLEAN}" = "1" ]; then
  mkdir -p "${REPO_ROOT}/runs"
  docker cp "${CONTAINER_ID}:/logs/engine.log" - \
    | tar -xO \
    | python3 "${SCRIPT_DIR}/clean_engine_log.py" > "${CLEAN_LOG_PATH}"
  less -R "${CLEAN_LOG_PATH}"
  echo "Wrote cleaned log to ${CLEAN_LOG_PATH}"
  exit 0
fi

docker cp "${CONTAINER_ID}:/logs/engine.log" - \
  | tar -xO \
  | python3 "${SCRIPT_DIR}/clean_engine_log.py" \
  | less -R
