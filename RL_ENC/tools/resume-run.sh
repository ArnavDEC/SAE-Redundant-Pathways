#!/usr/bin/env bash

set -euo pipefail

DEFAULT_MESSAGE="Continue what you were working on."

usage() {
  cat <<'EOF'
Usage: ./tools/resume-run.sh [--interactive] <container_id> [message] [-- <extra claude args>]

Resume a finished run's container as a `claude --continue` session, continuing
the agent's most recent conversation in /workdir. Watch it with
./tools/watch-run.sh <container_id>.

Sends a message and lets the agent work until it stops, just like a normal run,
except that the result isn't graded. (The container is still stopped.)

  message         Prompt to resume with. Ignored with --interactive.
                  Default: "Continue what you were working on."

Anything after `--` is passed through to `claude`.

Options:
  -i, --interactive  Attach and drive it yourself.
                     Won't stop the container when Claude stops.
  -h, --help         Show this help message

The grader is deliberately not re-run. To trigger it manually, run:
  ./tools/task_runner.py regrade <container_id>
A resumed grade is not directly comparable to a normal grade: the agent's inputs
are different, the regrade path skips the entrypoint's pre-grading lockdown, and
grade.json keeps the original run's agent timings.
EOF
}

INTERACTIVE=false
CONTAINER_ID=""
MESSAGE=""
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -i|--interactive) INTERACTIVE=true ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)
      if [ -z "$CONTAINER_ID" ]; then
        CONTAINER_ID="$1"
      elif [ -z "$MESSAGE" ]; then
        MESSAGE="$1"
      else
        echo "Provide at most one container id and one message." >&2; exit 1
      fi ;;
  esac
  shift
done

if [ -z "$CONTAINER_ID" ]; then
  echo "Error: no container id provided" >&2
  usage >&2
  exit 1
fi

CMD="claude --dangerously-skip-permissions --continue"
if $INTERACTIVE; then
  # Suppress the autoexit Stop hook for this invocation only.
  CMD+=" --settings $(printf '%q' '{"disableAllHooks": true}')"
else
  CMD+=" $(printf '%q' "${MESSAGE:-$DEFAULT_MESSAGE}")"
fi
for a in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do CMD+=" $(printf '%q' "$a")"; done

# Plant /tmp/DEBUG so the entrypoint hangs instead of re-running task + grader.
# A real empty file, not the `docker cp /dev/null` debug-container.sh uses: that
# lands a device node, which a later cp won't overwrite.
MARKER=$(mktemp)
trap 'rm -f "$MARKER"' EXIT
docker cp -q "$MARKER" "$CONTAINER_ID":/tmp/DEBUG

docker start "$CONTAINER_ID" >/dev/null

# Reattach rather than fail if a previous resume is still live.
if ! docker exec "$CONTAINER_ID" tmux has-session -t rl-agent 2>/dev/null; then
  # engine-resume.log, not engine.log: never clobber the original run's log.
  docker exec "$CONTAINER_ID" /run-agent-tmux.sh /logs/engine-resume.log "$CMD"
fi

if $INTERACTIVE; then
  docker exec -it "$CONTAINER_ID" tmux attach -t rl-agent || true
else
  echo "Resuming; watch with: ./tools/watch-run.sh $CONTAINER_ID"
  # Errors if Claude exited and took the tmux server -- what we're waiting for.
  docker exec "$CONTAINER_ID" tmux wait-for rl-agent-done || true
fi

# A live session means we detached: leave the container up.
if docker exec "$CONTAINER_ID" tmux has-session -t rl-agent 2>/dev/null; then
  echo "Detached; container left running. Reattach: $0 --interactive $CONTAINER_ID"
  echo "Stop it when done:  docker stop $CONTAINER_ID"
else
  echo "Claude exited. Re-grade with: ./tools/task_runner.py regrade $CONTAINER_ID"
  docker stop "$CONTAINER_ID" >/dev/null &
fi
