#!/usr/bin/env bash
# Unified entrypoint for task execution and grading
# Runs agent as non-root user, then grader as root (only if agent succeeds)

# If we've been signalled we're in debug mode, just hang forever so that the
# user can attach a shell.
#
# It feels like there should be a cleaner way to accomplish this, but there
# really isn't: `exec` only works on running containers; `start` runs
# the existing entrypoint and then stops the container, but doesn't let you
# change the entrypoint; `run`, which let's you set the entry point, can't
# resume an existing container.
# cf https://stackoverflow.com/questions/32353055/how-to-start-a-stopped-docker-container-with-a-different-command
if [ -e /tmp/DEBUG ]; then
  tail -f /dev/null
fi

# ============================================================================
# CONFIGURATION
# ============================================================================

ENGINE="${ENGINE:-claude}"
USER_NAME="${USER_NAME:-model}"

# Log all orchestration to entrypoint.log
exec > >(tee /logs/entrypoint.log) 2>&1

echo "============================================================================"
echo "Engine: $ENGINE | Timeout: ${TASK_TIMEOUT}s | User: $USER_NAME | Disallowed tools: ${DISALLOWED_TOOLS:-(none)}"
echo "============================================================================"

# ============================================================================
# AGENT EXECUTION PHASE (as non-root user)
# ============================================================================

echo ""
echo "==> Starting task execution as user '$USER_NAME'..."

# Explicitly set HOME to the agent user's directory for the agent phase.
# The base image sets ENV HOME=/home/${USER_NAME} for build-time tooling,
# but we re-assert it here so the entrypoint's security boundary is
# obvious from reading this script alone.  It is reset to /root before
# grading (see PRE-GRADING LOCKDOWN below).
USER_HOME="/home/${USER_NAME}"
export HOME="$USER_HOME"

# Record agent start time
python3 -c "
import json
from datetime import datetime
with open('/grader/timing.json', 'w') as f:
    json.dump({'agent_start_time': datetime.utcnow().isoformat()}, f)
"

task_py="/TASK.py"; [ -f "$task_py" ] || task_py="/task.py"
if [ -f "$task_py" ]; then
    python3 "$task_py" ${@:2}
fi

# Validate task file exists (accept TASK.txt or task.txt)
task_txt="/TASK.txt"; [ -f "$task_txt" ] || task_txt="/task.txt"
if [ ! -f "$task_txt" ]; then
    echo "ERROR: Task file not found at /task.txt or /TASK.txt"
    exit 1
fi

# If there is any hardware that the agent should use that's env and not task
# dependent, tell the agent to use it.
echo $HARDWARE_INFO >> "$task_txt"

chmod o+w /logs
# Define the agent command based on engine
if [ "$ENGINE" = "codex" ]; then
    AGENT_CMD="bash -lc 'codex exec --skip-git-repo-check --yolo --experimental-json - < $task_txt'"
elif [ "$ENGINE" = "claude" ]; then
    AGENT_CMD="claude --dangerously-skip-permissions --model \"${AGENT_MODEL:-claude-opus-4-8}\" --effort max --verbose${DISALLOWED_TOOLS:+ --disallowed-tools \"$DISALLOWED_TOOLS\"} < $task_txt"
else
    echo "ERROR: Unknown engine '$ENGINE'. Must be 'claude' or 'codex'"
    exit 1
fi
echo "Debug: AGENT_CMD=$AGENT_CMD"

# Execute agent with timeout as non-root user
/run-agent-tmux.sh /logs/engine.log "${AGENT_CMD}"
echo "Started new tmux session in container called rl-agent; attach to it to view progress"
timeout ${TASK_TIMEOUT} tmux wait-for rl-agent-done
TIMEOUT_RESULT=$?
echo "Finished waiting for session"

# Kill any processes the agent left running (e.g. background jobs, spawned
# subprocesses) before the grader starts. This prevents the agent from
# communicating with a helper process during grading or writing additional
# files to the filesystem after the grader begins its cleanup sweep.
echo "Killing any lingering processes owned by '${USER_NAME}'..."
# STOP-then-KILL: stopped procs can't fork, so fork-churn racing the kill
# converges instead of regrowing between scan passes (mirrors shim.py).
for i in $(seq 20); do
    pkill -STOP -u "${USER_NAME}" 2>/dev/null
    STATS=$(ps -u "${USER_NAME}" -o stat= 2>/dev/null)
    # T stopped; Z/D/t can't fork (D has STOP pending, stops before userspace)
    # printf because echo "" gets caught by the grep
    if ! printf '%s' "$STATS" | grep -qv '^[TZDt]'; then
        break
    fi
    [ "$i" = 20 ] && KILL_EVASION=1
done
pkill -KILL -u "${USER_NAME}" 2>/dev/null || true
echo "Lingering processes killed"

# Record agent end time
python3 -c "
import json
from datetime import datetime
with open('/grader/timing.json') as f:
    timing = json.load(f)
timing['agent_end_time'] = datetime.utcnow().isoformat()
with open('/grader/timing.json', 'w') as f:
    json.dump(timing, f)
"

# If we timed out waiting for the agent, this will log the progress so far
# before moving on.
if [ ! -f /logs/engine.log ]; then
  tmux capture-pane -pS - -t rl-agent > /logs/engine.log
fi

# Check for timeout
if [ "$TIMEOUT_RESULT" = "124" ]; then
    echo ""
    echo "========================================================================"
    echo "!!! AGENT EXECUTION TIMEOUT REACHED !!!"
    echo "!!! Task did not complete within ${TASK_TIMEOUT} seconds !!!"
    echo "========================================================================"
    echo ""
    # Write timeout marker for results_table.py to detect, including timing data
    python3 -c "
import json
with open('/grader/timing.json') as f:
    timing = json.load(f)
grade = {
    'score': 0.0,
    'feedback': 'TIMEOUT: Agent did not complete within ${TASK_TIMEOUT} seconds',
    'timeout': True,
    **timing
}
with open('/grader/grade.json', 'w') as f:
    json.dump(grade, f, indent=2)
"
fi

# Agent processes survived the STOP loop: the grading environment is
# compromised, so skip the grader and record a deserved zero (mirrors
# shim.py's kill_evasion_detected path).
if [ "${KILL_EVASION:-0}" = "1" ]; then
    echo "!!! KILL EVASION DETECTED: agent processes still runnable after 20 STOP passes !!!"
    jq --arg stats "$STATS" \
       '. + {score: 0.0, feedback: "KILL EVASION: agent processes survived pre-grading kill loop", kill_evasion_detected: true, final_proc_stats: $stats}' \
       /grader/timing.json > /grader/grade.json
    exit 0
fi
# ============================================================================
# PRE-GRADING LOCKDOWN
# ============================================================================

# Reset HOME to root's default so the grader does not inherit the agent's
# home directory (where the agent could have planted config files).
export HOME=/root

# Clear /tmp so agent can't mess with grader I/O
# Obsolecence of /tmp contents at point of grader start was verified before this change
rm -rf /tmp/* /tmp/.[!.]* 2>/dev/null || true

# ============================================================================
# GRADING PHASE (as root)
# ============================================================================
{
    echo ""
    echo "==> Starting grading as root..."

    # Check if GRADER.py exists
    if [ ! -f "/grader/GRADER.py" ] && [ ! -f "/grader/grader.py" ]; then
        echo "ERROR: GRADER.py/grader.py not found in /grader/"
        exit 1
    fi

    # Run grader and write result directly to grade.json
    cd /workdir
    python3 /grader/run_grader.py ${@:2}
    GRADER_EXIT_CODE=$?

    if [ "$GRADER_EXIT_CODE" = "0" ]; then
        echo "Grading completed successfully"
    else
        echo "GRADER FAILED: Grading exited with code $GRADER_EXIT_CODE"
    fi

} | tee /logs/grader.log 2>&1

# ============================================================================
# COMPLETION
# ============================================================================

# Final summary to stdout
echo ""
echo "============================================================================"
echo "Complete - Check logs for details"
echo "============================================================================"

# Exit with grader's status
exit $GRADER_EXIT_CODE
