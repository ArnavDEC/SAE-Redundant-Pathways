# RL Environment

A task execution and grading system for developing RL environments for LLMs.

## Setup

For Ubuntu machines:

```bash
# 1. System dependencies (Docker, npm, claude-code CLI). Reads/writes ~/.bashrc.
#    Adds you to the `docker` group, so restart your terminal afterwards
#    (or run `newgrp docker`) before any docker command will work.
./install_system_dependencies.sh

# 2. Python environment
uv sync
source .venv/bin/activate

# 3. Environment variables. Copy the template, then open .env in any editor
#    and paste your ANTHROPIC_API_KEY (provided in your invitation email).
#    Source it before running tasks.
cp .env.example .env
# ...edit .env to paste your key...
source .env
```

`HF_TOKEN` is optional — only set it if your task downloads gated models
(e.g. Llama, Gemma) during image build. See `.env.example` for details.

You will also need passwordless `sudo` (or willingness to type your password
once per session) — `run.sh` configures iptables rules on the Docker bridge
to restrict the agent's network access to the Anthropic API only.

## Quick Start

Start by running an existing example task to test your setup.

```bash
# Run a single task
./run.sh simple-math

# View the rollout from <containerid> (output by previous command)
./tools/view-log.sh <containerid> engine
```

## Dependencies

- Docker Engine 20.10.14 or higher (for `--build-context` support)
- Python 3.9 or higher
  - Python packages from the pyproject.toml; install with `uv sync` or `pip install .`
- Claude Code

## Creating your own task

1. **Create a new directory** in the tasks folder with your task name
2. **Create a Dockerfile** to set up your task environment. It should be called
   "Dockerfile" and be in the top level of your task directory. Start with an
   existing Dockerfile like the one from the `simple-math` task, and modify it to
   include any dependencies you need. DO NOT modify any of the lines above the
   `TASK SPECIFIC CONTENT` comment. If you want any files to be accessible to the
   agent, copy them into the `/workdir/` directory in your Dockerfile.
3. **Create a task.txt** or task.py file for setting up your task prompt. If
   you use a task.txt, you'll just write a plain text prompt to that file that
   the agent will be given at the beginning of the task; you will commit this
   file. If you want a dynamically generated prompt, you can write a task.py
   script which will be called to generate your task.txt; it should produce
   something reasonable by default with no arguments. Either file should go in the
   top level of your task directory.
4. **Write a grader.py script** inside a `grader` subdirectory of your task
   directory. The grader directory and all its contents will be copied into the
   environment, but will be only accessible to the grader and not the agent when
   it is doing the task. It must provide a `grade()` function (takes no
   parameters) that runs with root privileges in the container and has access to
   `/workdir`. The function must return a dictionary with at least a "score"
   field holding a float from 0 to 1, but can also have any other fields to hold
   useful information, which we generally put under a field called "feedback".
5. **Note:** ./run.sh installs a network isolation mechanism for all the tasks.
   Accordingly, this means that you have to pre-install all of the files you
   want in the agent's environment in your task Dockerfile. This is a design
   choice we'd like to enforce; we've found that this overall reduces the
   complexity of task development.

## Running/Debugging your task

1. **Start a single rollout** with `./run.sh <taskname>`. It will build the
   docker image and start the agent with your prompt. This script is a
   convenience wrapper around `./tools/task_runner.py`; use `--help` to see more
   flags.
2. **You can watch the agent work** in the Claude UI by running
   `./tools/watch-run.sh <containerid>` in another terminal; the run.sh script
   will tell you what container id to use.
3. **You can view logs** to see the agent's output (what
   ./tools/watch-run.sh shows you during the run) with `./tools/view-log.sh
   <containerid> engine`, find any grader errors with `./tools/view-log.sh
   <containerid> grader`, or look at the output of the entrypoint script with
   `./tools/view-log.sh <containerid> entrypoint`.
4. **View the grade** from the task once it is finished with `./tools/view-grade.sh
   <containerid>`. This command may fail with a docker HTTP error if your
   grader script failed to run to completion. Or, you may see unexpected feedback
   or grading value.
5. **You can also investigate the full state of the container** after the run with
   `./tools/debug-container.sh <containerid>`
6. **Re-run the grader** after making updates to it and fixing bugs by running
   `./tools/task_runner.py regrade <containerid>`.
7. **Do a "run" of multiple rollouts** once you're satisfied with the result of
   a single rollout, to measure overall pass rate. You'll start your run with
   `./run.sh <taskname> -n10`. You may want to use the `-j` flag to set the
   number of simultaneous rollouts (default 4).
8. **View the results of a run** with `./tools/run-pass-rate.sh
   runs/<runname>.txt`. Every run is given a unique filename in the `runs`
   folder when you have more than one rollout. The running interface will tell you
   the name of your run; you can also name your run something specific by
   passing `--run-name <name>` to the run script. You can also get a table summary
   of the result, similar to the one printed while the run is going, by running
   `./tools/results_table.py`

## Glossary

- **Rollout** - A single run of a task with an LLM agent
- **Container ID** - A unique string identifying a container where a rollout occurs
- **Run** - A set of rollouts done together

## Project Structure

```
rl-env/
├── run.sh                       # Main entry point (thin wrapper)
├── docker/
│   ├── entrypoint.sh           # In-container execution script
│   ├── autoexit-claude-settings.json  # Claude Code settings
│   └── grader/
│       └── run_grader.py       # Grader execution script
├── tools/                       # Helper scripts and orchestration
│   ├── task_runner.py          # Main orchestrator (builds images, runs containers)
│   ├── docker_helpers.py       # Docker utility functions
│   ├── results_table.py        # Display results summary table
│   ├── watch-run.sh            # Watch agent work in real-time
│   ├── view-grade.sh           # View grading results
│   ├── view-log.sh             # View container logs
│   ├── debug-container.sh      # Debug container state
│   └── run-pass-rate.sh        # Calculate pass rates from runs
├── tasks/
│   └── [task_name]/
│       ├── Dockerfile          # Task-specific image definition
│       ├── task.txt            # Task description (or task.py for dynamic)
│       ├── task.py             # Optional: Dynamic task generation
│       └── grader/
│           ├── grader.py       # Evaluation logic (required)
│           └── grader_data/    # Optional: Test data for grader
└── runs/                        # Run metadata and batch tracking
    └── [run_id].txt            # List of container IDs for a batch run
```

## Container Directory Structure

When a task container is running, the internal filesystem is organized as follows:

```
/
├── entrypoint.sh               # Orchestration script (runs agent, then grader)
├── task.txt                    # Task description (piped to agent at start)
├── taskname.txt                # Task name (created by task_runner, not Dockerfile)
├── workdir/                    # Agent's working directory (only location agent can read/write)
├── grader/                     # Grader code (root-owned, 700 permissions - agent CANNOT read)
│   ├── grader.py              # Grading logic
│   ├── grader_data/           # Optional test data
│   └── grade.json             # Output (written after grading)
└── logs/                       # Execution logs (created at runtime by entrypoint)
    ├── entrypoint.log         # Orchestration log
    ├── engine.log             # Agent CLI output (may not exist if agent hasn't started)
    └── grader.log             # Grading output (may not exist if grading hasn't started)
```

### Tmux Compatibility
The runner and the `./tools/watch-run.sh` script use tmux to provide a view into
the Claude UI that you can attach to. If you are doing your work inside a tmux
session, though, this can cause some issues. Unfortunately, since the outer tmux session and
the inner one are separated by the docker container boundary, tmux won't give
you the normal warning about nesting tmux sessions when you run the
`./tools/watch-run.sh` script, so beware.

If you're doing your work in a tmux session, you have several options for
working with this script:
1. Leave tmux before running the `./tools/watch-run.sh` script.
2. Change your tmux prefix key to something other than the default
   (https://stackoverflow.com/questions/11557076/if-i-set-key-bind-of-c-b-to-c-a-in-tmux-how-can-i-move-the-cursor-to-the-beginni)
3. Use C-b C-b as your prefix: C-b C-b (C-b twice) in a tmux where the prefix
   is the default C-b will send a C-b to the underlying process. That will get
   interpreted as a C-b by the inner tmux, so you can send commands to it.
4. Set C-b in the container: the most complicated option, you would have to
   copy in a config file with a different prefix in your task Dockerfile.
