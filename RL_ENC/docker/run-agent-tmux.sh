#!/bin/bash
# Start AGENT_CMD in the detached, root-owned tmux session named "rl-agent"; use
# ./tools/watch-run.sh to attach. When the agent exits, the session captures the
# pane to LOGFILE and signals "rl-agent-done".

USER_NAME="${USER_NAME:-model}"

# The session runs as root, so `sudo -E` would otherwise hand the agent HOME=/root,
# where `claude --continue` finds no session history.
export HOME="/home/${USER_NAME}"

LOGFILE="$1"
AGENT_CMD="$2"

# tmux doesn't inherit env vars, so source the TPU config inside the session;
# `sudo -E` preserves it across the drop to ${USER_NAME}.
tmux new-session -s rl-agent -d \
  "source /etc/tpu-env.sh && cd /workdir && (sudo -E -u ${USER_NAME} ${AGENT_CMD}) ; \
  (tmux capture-pane -e -pS - > ${LOGFILE}); tmux wait-for -S rl-agent-done"
