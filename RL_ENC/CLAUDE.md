# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a secure task execution and grading system designed to test AI programming assistants (Claude, Codex) on various programming challenges. It uses **permission-based isolation** within a single Docker container: the agent runs as a non-root user and cannot read the grader code (owned by root with 700 permissions), while grading runs as root only after the agent completes.

## Common Commands

### Running Tasks

```bash
# Run a single task (default: Claude)
./run.sh partition

# Run with Codex
./run.sh --engine codex partition

# Run all tasks
./run.sh --all

# Run with custom parallelism (default: 4)
./run.sh -j 8 --all

# Run multiple tasks
./run.sh simple_math string_reverse

# Run a task multiple times
./run.sh -n 5 simple_math

# Run with custom timeout (default: 3600s)
./run.sh --timeout 7200 long_task
```

### Debugging and Monitoring

```bash
# Watch agent work in real-time (visual/interactive, for humans only)
./tools/watch-run.sh <container_id>

# View grading results
./tools/view-grade.sh <container_id>

# View logs (entrypoint, engine, or grader)
./tools/view-log.sh <container_id> grader
./tools/view-log.sh <container_id> engine
./tools/view-log.sh <container_id> entrypoint

# Debug container state after run
./tools/debug-container.sh <container_id>

# Re-grade with updated grader code
./tools/task_runner.py regrade <container_id>

# View pass rates for a batch run
./tools/run-pass-rate.sh runs/<run_id>.txt

# Display results summary table
./tools/results_table.py
```

### Task Development

When creating a new task:
1. Create directory: `tasks/{task_name}/`
2. Add `task.txt` with task requirements (piped to agent), OR `task.py` to dynamically generate task.txt (task_runner.py runs this before building)
3. Create `Dockerfile` that:
   - Installs Claude/Codex CLI
   - Copies `grader` to `/grader`
   - Copies `task.py` to `/task.py` (or `task.txt` to `/task.txt` if static file)
   - Sets up `/workdir` directory
   - Optionally copies any starter files into `/workdir`
   - Uses `docker/entrypoint.sh` as entrypoint
4. Add `grader/grader.py` with `grade()` function inside `grader/` subdirectory
5. Optional: Add `grader_data/` subdirectory with test data for grader

## Architecture

### Task Execution Flow
1. **Image Building**: `task_runner.py` builds task-specific Docker image `rl-task-{name}` from `tasks/{name}/Dockerfile`
2. **Container Startup**: Single container runs `docker/entrypoint.sh` as root
3. **Agent Execution Phase** (runs as non-root user):
   - Entrypoint drops to the non-root user via `sudo -E -u ${USER_NAME}`
   - Pipes `/task.txt` to Claude/Codex CLI in `/workdir`
   - Agent output logged to `/logs/engine.log`
   - Times out after `TASK_TIMEOUT` seconds (default: 3600)
4. **Grading Phase** (runs as root, only if agent succeeded):
   - Imports `/grader/grader.py` and calls `grade()`
   - Writes result to `/grader/grade.json`
   - Output logged to `/logs/grader.log`

### Security Isolation Model
**Permission-based isolation within a single container:**
- Agent runs as non-root user and **cannot read** `/grader` (root-owned, 700 permissions)
- Grader code is present in the container but inaccessible to the agent
- Grader only runs as root after agent completes
- Network access is restricted to the Anthropic API only; the agent cannot download packages or fetch external data at runtime, so dependencies must be pre-installed in the task Dockerfile

### Key Files
- **`run.sh`**: Thin wrapper that forwards arguments to `tools/task_runner.py`
- **`tools/task_runner.py`**: Main orchestration script - builds images, runs containers, manages parallel execution
- **`docker/entrypoint.sh`**: In-container execution script - runs agent as non-root, then grader as root
- **`tasks/{name}/Dockerfile`**: Task-specific image definition (copies grader, task.txt, sets up `/workdir`)
- **`tasks/{name}/task.txt`**: Task description piped to agent
- **`tasks/{name}/grader/grader.py`**: Grading logic with `grade()` function

### Authentication

`task_runner.py` reads `ANTHROPIC_API_KEY` from your shell environment,
validates it, and writes `docker/container-internal-claude.json` (gitignored)
with the key fingerprint on every run. Keep `ANTHROPIC_API_KEY` exported in
your shell — no manual file copying needed.

For Codex (`--engine codex`): authenticate the Codex CLI on the host (creates
`~/.codex/`); the Dockerfile copies that directory into the image.

### Task Structure Examples
- **Simple Python**: Single file solutions — `simple_math` (write solution.py)
- **Algorithm**: Data structures — `partition` (the partition problem)
- **String manipulation**: `string_reverse`

### Grader Requirements

Graders must implement a `grade()` function that:
- Takes no parameters (grader runs from within the container with access to `/workdir`)
- Returns dict with at least `{"score": float}` (0.0 to 1.0), can include additional fields like `"feedback"`
- Handles missing/malformed solutions gracefully (return score=0, not crash)

The grader execution (`docker/grader/run_grader.py`):
- Imports grader module and calls `grade()`
- Catches exceptions and writes error to `grade.json` with score=0
- Adds timestamp to result
- Writes formatted grade to `/grader/grade.json`

### Dockerfile Security Guidelines

The agent is adversarial — assume it will try to tamper with anything writable.
Whatever state the container ends up in is what the grader sees, so anything
the agent can modify becomes a grader input.

**Do not `chown ~/.cache` after `uv pip install`.** `uv` hardlinks packages
from its cache into `/usr/local/lib/python*/site-packages/`. If you then
`chown -R ${USERNAME} ~/.cache`, those hardlinks make site-packages files
agent-owned, allowing the agent to overwrite e.g. `torch/__init__.py` with
code that runs when the grader imports it. If the agent needs cache access,
either chown before installing packages, or chown only the specific
subdirectory needed (e.g. `.cache/huggingface`).

**HuggingFace cache lives at `/opt/huggingface`.** The base image sets
`HF_HOME=/opt/huggingface` (root-owned, agent-readable, 755). Anything
downloaded via `snapshot_download`, `hf`, or `from_pretrained` during image
build lands there as root and the agent cannot tamper with it. Don't `chown`
or `chmod` that path in your Dockerfile — just run downloads as root and
leave it alone.

**If your grader executes agent-written code** (e.g. importing an
`architecture.py` the agent wrote), run that import in a subprocess with
dropped privileges. Otherwise the agent's code can monkeypatch modules
(`torch`, `json`, `builtins`) inside the grader process and influence the
score. Spawn `subprocess.run(["sudo", "-u", "nobody", ...])` or equivalent,
have it write the result to disk, and read it back.

### Container Directory Structure
```
/entrypoint.sh         # Orchestration script (runs agent, then grader)
/task.txt              # Task description (contents piped to agent at start, not readable as file)
/{taskname}.txt        # Task name file
/workdir/              # Agent's working directory (writable by non-root user)
/grader/               # Grader code (root-owned, 700 permissions - agent CANNOT read)
  grader.py            # Grading logic
  grader_data/         # Optional test data
  grade.json           # Output (written after grading)
/logs/                 # Execution logs
  entrypoint.log       # Orchestration log
  engine.log           # Agent CLI output
  grader.log           # Grading output
```

### Helper Scripts

The `tools/` directory contains helper scripts for interacting with running and completed tasks:
- `task_runner.py`: Main orchestrator - builds images, runs containers, manages parallel execution
- `docker_helpers.py`: Docker utility functions
- `results_table.py`: Display results summary table
- `watch-run.sh`: Watch agent work in real-time
- `view-grade.sh`: View grading results from a container
- `view-log.sh`: View container logs (entrypoint, engine, or grader)
- `debug-container.sh`: Debug container state after run
- `run-pass-rate.sh`: Calculate pass rates from batch runs
