#!/usr/bin/env bash

set -euo pipefail

NETWORK_MODE=false
CONTAINER_ID=""

usage() {
  echo "Usage: $0 [--network] <container_id>"
  echo ""
  echo "Options:"
  echo "  --network    Switch container from anthropic_only to bridge network"
  echo "               (restores original network on exit)"
  exit 1
}

# Parse arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --network)
      NETWORK_MODE=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      ;;
    *)
      if [ -n "$CONTAINER_ID" ]; then
        echo "Error: Multiple container IDs provided"
        usage
      fi
      CONTAINER_ID="$1"
      shift
      ;;
  esac
done

if [ -z "$CONTAINER_ID" ]; then
  echo "Error: No container ID provided"
  usage
fi

# Try to copy the debug file. If it fails, it's probably already there, so just
# ignore the error
docker cp -q /dev/null "$CONTAINER_ID":/tmp/DEBUG &> /dev/null || true

# Switch networks if requested
if [ "$NETWORK_MODE" = true ]; then
  echo "Switching container to bridge network..."
  docker network disconnect anthropic_only "$CONTAINER_ID" 2>/dev/null || true
  docker network connect bridge "$CONTAINER_ID"
fi

docker start "$CONTAINER_ID"
docker exec -it "$CONTAINER_ID" /bin/bash

# Restore original network if we changed it
if [ "$NETWORK_MODE" = true ]; then
  echo "Restoring container to anthropic_only network..."
  docker network disconnect bridge "$CONTAINER_ID" 2>/dev/null || true
  docker network connect anthropic_only "$CONTAINER_ID"
fi

docker stop "$CONTAINER_ID" &
