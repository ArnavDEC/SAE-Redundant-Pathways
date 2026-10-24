
import argparse
import difflib
import fnmatch
import os
import re
import sys
import platform
import grp
import pwd
import shutil
import subprocess
import tempfile
import json
import json5
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from collections.abc import Callable
from typing import IO, Any, NewType, Protocol, TypeVar, cast

from pathlib import Path


class ProgressReader:
    """Wrap a readable for a streaming upload; reports (sent, total) to a callback per read."""
    def __init__(self, stream: IO[bytes], total: int, callback: Callable[[int, int], None]) -> None:
        self._stream, self._total, self._sent, self._cb = stream, total, 0, callback

    def __len__(self) -> int:
        # requests uses len() for Content-Length, avoiding chunked encoding
        return self._total

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        self._sent += len(chunk)
        if chunk:
            self._cb(self._sent, self._total)
        return chunk


def upload_bar(label: str) -> Callable[[int, int], None]:
    """A (sent, total) callback that draws a one-line upload progress bar to a tty."""
    def draw(sent: int, total: int) -> None:
        if not sys.stderr.isatty():
            return
        pct = sent / total * 100
        filled = int(pct // 2)
        sys.stderr.write(f"\r{label} [{'#' * filled}{'.' * (50 - filled)}] {pct:5.1f}%")
        sys.stderr.flush()
        if sent >= total:
            sys.stderr.write("\n")
    return draw


def check_docker_permissions() -> None:
    """Check that user has appropriate Docker permissions."""
    # Check if running as root
    if os.getuid() == 0:
        print("WARNING: Running as root. This is not recommended.", file=sys.stderr)
        print("Consider running as a non-root user in the docker group instead.", file=sys.stderr)
        print("", file=sys.stderr)
        return

    # macOS: Docker Desktop handles permissions automatically; skip docker group check
    if platform.system() == 'Darwin':
        return

    # Check if user is in docker group
    try:
        docker_group = grp.getgrnam('docker')
        user_groups = os.getgroups()

        if docker_group.gr_gid not in user_groups:
            current_user = pwd.getpwuid(os.getuid()).pw_name
            print("ERROR: Current user is not in the 'docker' group.", file=sys.stderr)
            print("Please add yourself to the docker group with:", file=sys.stderr)
            print(f"  sudo usermod -aG docker {current_user}", file=sys.stderr)
            print("Then log out and back in for the changes to take effect.", file=sys.stderr)
            print("See https://docs.docker.com/engine/install/linux-postinstall/"
                  "for details", file=sys.stderr)
            sys.exit(1)
    except KeyError:
        print("Error: 'docker' group not found on system.", file=sys.stderr)
        print("Please create the group with `sudo groupadd docker` "
              "and add yourself with `sudo usermod -AG docker {current_user}`. "
              "Then log out and back in for the changes to take effect.",
              file=sys.stderr)
        print("See https://docs.docker.com/engine/install/linux-postinstall/"
              "for details", file=sys.stderr)
        sys.exit(1)

def repo_root() -> Path:
    """Root of the checkout containing cwd, else the one containing this file."""
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / ".git").exists() or (d / ".jj").exists():
            return d
    return Path(__file__).resolve().parent.parent


def dockerignore_patterns(task_dir: Path) -> list[str]:
    """Global tasks/.dockerignore patterns, then the task's own so its `!` rules win."""
    # Not task_dir.parent: nested tasks (tasks/sprint/foo) still take tasks/.dockerignore.
    paths = [repo_root() / "tasks" / ".dockerignore", task_dir / ".dockerignore"]
    return [line
            for p in paths
            if p.exists()
            for line in p.read_text().splitlines()]


def build_in_task_dir(build_cmd: list[str], task_dir: Path) -> None:
    """Mirror the task's configured repo files, then docker build in task_dir
    with the merged .dockerignore staged in place."""
    for source, dest in load_task_config(task_dir).mirrored_repo_files.items():
        root = repo_root()
        if not (root / source).is_file():
            raise FileNotFoundError(
                f"mirrored_repo_files source {source!r} in {task_dir}'s config "
                f"is not a file under the repo root {root}")
        target = task_dir / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / source, target)
    merged = "\n".join(dockerignore_patterns(task_dir))
    local = task_dir / ".dockerignore"
    backup = task_dir / ".dockerignore.backup"
    if local.exists():
        local.rename(backup)
    local.write_text(merged)
    try:
        subprocess.run(build_cmd, cwd=task_dir, stdout=subprocess.DEVNULL, check=True)
    finally:
        local.unlink()
        if backup.exists():
            backup.rename(local)


GITHUB_REPO = "dmodel-ai/rl-env"

CommitId = NewType("CommitId", str)  # full sha of a (snapshot) commit


class RepoError(Exception):
    """Git/jj machinery failure with a user-facing message; CLI boundaries
    translate it into a clean exit."""

# Fixed identity and date so an identical (HEAD, tree) pair snapshots to an
# identical commit sha on every machine — re-pushing the same state is a no-op.
_SNAPSHOT_IDENTITY = {f"GIT_{role}_{field}": value
                      for role in ("AUTHOR", "COMMITTER")
                      for field, value in [("NAME", "dmodel run snapshot"),
                                           ("EMAIL", "runs@dmodel.ai"),
                                           ("DATE", "2000-01-01T00:00:00+0000")]}


def _run_out(cmd: list[str], **kwargs: Any) -> str:
    """stdout of cmd, stripped; RepoError naming the command and stderr on failure."""
    try:
        proc: subprocess.CompletedProcess[str] = subprocess.run(
            cmd, capture_output=True, text=True, check=True, **kwargs)
        return proc.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RepoError(f"`{' '.join(cmd)}` exited {e.returncode}:\n"
                        f"{e.stderr.strip()}") from e


def git_cmd(root: Path) -> list[str]:
    """A git invocation pinned to the repo, for git and (non-colocated) jj checkouts."""
    if (root / ".git").exists():
        return ["git", "-C", str(root)]
    if (root / ".jj").exists():
        return ["git", "--git-dir", _run_out(["jj", "git", "root"], cwd=root)]
    raise RepoError(f"{root} is neither a git nor a jj checkout")


def _github_repo(url: str) -> str | None:
    """The owner/name a remote URL points at, if it's a GitHub URL in any form."""
    m = re.fullmatch(r"(?:git@github\.com:|ssh://git@github\.com/|"
                     r"https://github\.com/)(.+?)(?:\.git)?/?", url)
    return m.group(1) if m else None


def find_github_remote() -> str:
    """The git remote pointing at the canonical GitHub repo; raises if there is none."""
    git = git_cmd(repo_root())
    for name in _run_out(git + ["remote"]).split():
        if _github_repo(_run_out(git + ["remote", "get-url", name])) == GITHUB_REPO:
            return name
    raise RepoError(f"no git remote points at github.com/{GITHUB_REPO}; add one "
                    f"(e.g. `git remote add origin git@github.com:{GITHUB_REPO}.git`) "
                    "so the run ref can be pushed, or pass --skip-pin")


def snapshot_commit(root: Path) -> CommitId:
    """Full sha of a commit capturing the tracked files of the working tree
    (untracked files stay local — the ref is pushed to GitHub); never touches
    HEAD or the real index.
    """
    try:
        # jj: working-copy commit is already a git commit
        if (root / ".jj").exists():
            return CommitId(_run_out(["jj", "log", "-r", "@", "--no-graph",
                                      "-T", "commit_id"], cwd=root))

        # git: synthetic commit via a copy of the index
        git = git_cmd(root)
        head = _run_out(git + ["rev-parse", "HEAD"])
        index = _run_out(git + ["rev-parse", "--path-format=absolute",
                                "--git-path", "index"])
        with tempfile.NamedTemporaryFile() as tmp_index:
            # seeding from the real index (not HEAD) keeps staged new files
            shutil.copyfile(index, tmp_index.name)
            env = {**os.environ, "GIT_INDEX_FILE": tmp_index.name}
            _run_out(git + ["add", "-u"], env=env)
            tree = _run_out(git + ["write-tree"], env=env)
        if tree == _run_out(git + ["rev-parse", "HEAD^{tree}"]):
            return CommitId(head)  # clean tree: the run is exactly HEAD
        return CommitId(_run_out(git + ["commit-tree", tree, "-p", head,
                                        "-m", "run snapshot"],
                                 env={**os.environ, **_SNAPSHOT_IDENTITY}))
    except RepoError as e:
        raise RepoError(f"couldn't snapshot the working tree: {e}") from e


def confirm(question: str) -> bool:
    """Banner-framed y/N prompt, all on stderr so piped stdout stays clean;
    False, without printing, when stdin isn't a tty."""
    if not sys.stdin.isatty():
        return False
    print(f"\n{'=' * 60}\n{question}\n{'=' * 60}\n[y/N] ", end="",
          file=sys.stderr, flush=True)
    return input().strip().lower() in ("y", "yes")


def push_run_ref(remote: str, sha: CommitId) -> str | None:
    """Push snapshot commit sha to refs/runs/<sha>; return the ref, or `None`
    on push failure (warns and prompts whether to continue unpinned)."""
    ref = f"refs/runs/{sha}"
    cmd = git_cmd(repo_root()) + ["push", remote, f"{sha}:{ref}"]
    push = subprocess.run(cmd, capture_output=True, text=True)
    if push.returncode != 0:
        print(f"WARNING: failed to push run ref {ref} to {remote}:\n"
              f"{push.stderr.rstrip()}\n"
              f"Push it manually with: {' '.join(cmd)}", file=sys.stderr)
        if not confirm("Continue without the run pin?"):
            raise RepoError("not continuing without the run pin; push it "
                            "manually (command above) or rerun with --skip-pin")
        return None
    print(f"Pushed run ref {ref} to {remote}")
    return ref


# Alternate tool names -> Claude Code CLI tool names. Claude has no tmux tool; its
# Bash tool plays the role of both bash and tmux (see disallowed_tools_claude).
TOOL_NAME_ALIASES: dict[str, str] = {
    "bash": "Bash",
    "tmux": "Bash",
    "str_replace_editor": "Edit",
}

# Always disallowed tools
ALWAYS_DISALLOWED_TOOLS: set[str] = {
    # These tools are always blocked in the claude code version of
    # the env because they allow it to circumvent the disallowedTools mechanism
    "Edit(**/.claude/**)",        
    "Write(**/.claude/**)",

    # Blocked so that the agent doesn't stall itself
    "AskUserQuestion"
}

DEFAULT_SUBMISSION_SIZE_LIMIT = 1 << 30  # 1 GiB; the remote runner's own default is 16 MiB


class ArtifactSpec(BaseModel):
    """A task's declared artifact: which files to record, and how to (re)build them."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    globs: list[str]
    build: str | None = None


# Deprecated CONFIG keys: used to be top-level keys, now folded into `remote_problem`
DEPRECATED_REMOTE_PROBLEM_KEYS = {"grader_timeout": "grading_timeout_seconds", "tool_timeout": "tool_timeout_seconds"}


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requires_network: bool = Field(False, alias="network")
    requires_accelerator: bool = False
    remote_problem: dict[str, Any] = {}  # Remote-runner problem fields, passed through verbatim and ignored by run.sh.
    remote_job: dict[str, Any] = {}  # Remote-runner job fields, likewise; one job covers every task submitted together.
    no_problem_prefix: bool = False
    disallowed_tools: set[str] = set()  # plus tmux, if CONFIG unsets enable_tmux
    agent_api_access: bool = False
    artifacts: dict[str, ArtifactSpec] = {}
    # Repo files copied into the task dir before every build, as {source: dest}.
    mirrored_repo_files: dict[str, str] = {}
    # Grade in the agent's own container, as tasks always have. Declaring
    # submission_mirror instead grades in a fresh one; setting both is an error.
    single_container: bool = False
    # Directories handed from the agent's container to the grader's, as
    # {source: destination}. The remote runner copies the regular files directly inside
    # each source (not nested subdirectories).
    submission_mirror: dict[str, str] | None = None
    submission_size_limit_bytes: int = DEFAULT_SUBMISSION_SIZE_LIMIT
    # Fresh grading container fields (remote-runner agentic_grader_params), passed verbatim; needs submission_mirror.
    remote_grader: dict[str, Any] = {}

    @field_validator("artifacts")
    @classmethod
    def _keys_name_a_ref_file(cls, artifacts: dict[str, ArtifactSpec]) -> dict[str, ArtifactSpec]:
        # sidecar suffixes would collide with another key's lock/cache files
        # there, which the repo .gitignore silently keeps out of git
        for key in artifacts:
            if not key.strip(".") or "/" in key \
                    or key.endswith((".cache", ".fetching", ".lock")):
                raise ValueError(f"artifact key {key!r} must be a plain filename "
                                 "not ending in .cache/.fetching/.lock; it names "
                                 "the ref file under .ARTIFACT_REFS/")
        return artifacts

    @model_validator(mode="before")
    @classmethod
    def _expand_sugar_keys(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Apply the CONFIG keys that are sugar for another field."""
        config = dict(config)
        disallowed_tools = list(config.get("disallowed_tools", []))
        if not config.pop("enable_tmux", True):
            disallowed_tools.append("tmux")
        config["disallowed_tools"] = disallowed_tools
        remote_problem = config.get("remote_problem", {})
        if not isinstance(remote_problem, dict):
            raise ValueError(f"CONFIG remote_problem must be an object, not {remote_problem!r}")
        remote_problem = dict(remote_problem)
        for old, new in DEPRECATED_REMOTE_PROBLEM_KEYS.items():
            value = config.pop(old, None)
            if value is None:
                continue
            if new in remote_problem:
                raise ValueError(f"CONFIG sets both {old} and remote_problem.{new}; "
                                 "keep the latter")
            print(f"WARNING: CONFIG key {old} is deprecated; write it as "
                  f"remote_problem.{new} instead", file=sys.stderr)
            remote_problem[new] = value
        config["remote_problem"] = remote_problem
        return config

    def disallowed_tools_claude(self) -> set[str]:
        """Return disallowed_tools translated to Claude Code CLI names."""
        disallowed_tools = set(self.disallowed_tools)

        # Claude Code maps `bash` and `tmux` to `Bash`. Only block `Bash` here if both are disabled.
        if not {"bash", "tmux"} <= disallowed_tools:
            disallowed_tools -= {"bash", "tmux"}
        return {TOOL_NAME_ALIASES.get(t, t) for t in disallowed_tools} | ALWAYS_DISALLOWED_TOOLS


def load_task_config(task_dir: Path) -> TaskConfig:
    json_path = special_file(task_dir / "CONFIG.json")
    jsonc_path = special_file(task_dir / "CONFIG.jsonc")
    if json_path.exists() and jsonc_path.exists():
        raise FileExistsError(
            f"Both CONFIG.json and CONFIG.jsonc exist in {task_dir}; pick one")
    config_path = json_path if json_path.exists() else jsonc_path
    if config_path.exists():
        with config_path.open() as f:
            return TaskConfig.model_validate(json5.load(f))
    return TaskConfig.model_validate({})

@dataclass
class Task:
    name: str
    problem_id: str | None
    config: TaskConfig | None
    params: dict[str, Any] | None

    def __init__(self, name: str, problem_id: str | None = None, *,
                 config: TaskConfig | None = None,
                 params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.problem_id = problem_id
        self.config = config
        self.params = params

    @classmethod
    def parse(cls, task_spec: str) -> "Task":
        if ":" not in task_spec:
            return Task(task_spec, None)
        else:
            parts = task_spec.split(":")
            return Task(parts[0], parts[1])
    def problem_name(self) -> str:
        if self.config and self.config.no_problem_prefix and self.problem_id:
            return self.problem_id
        return self.name.split("/")[-1] + \
                (f"-{self.problem_id}" if self.problem_id is not None else "")
    def load_config_and_params(self, root_dir: Path) -> None:
        self.config = load_task_config(root_dir / "tasks" / self.name)
        params_path = special_file(root_dir / "tasks" / self.name / "PARAMS.json")
        if params_path.exists():
            with params_path.open() as f:
                self.params = json.load(f)
        else:
            self.params = {}

    def __str__(self) -> str:
        if self.problem_id is None:
            return self.name
        else:
            return self.name + ":" + self.problem_id

def task_completer(**kwargs: object) -> list[str]:
    """Complete task names from tasks/ directory for argcomplete."""
    import sys
    root = repo_root()
    tasks_dir = root / "tasks"
    if not tasks_dir.exists():
        print(f"Warning: task autocompletion failed - {tasks_dir} does not exist",
              file=sys.stderr)
        return []
    valid_tasks = []
    # Walk recursively to find tasks in nested directories (e.g. linearprobe/probing)
    for task_pattern in ("TASK.txt", "TASK.py", "task.txt", "task.py"):
        for task_file in tasks_dir.rglob(task_pattern):
            task_name = str(task_file.parent.relative_to(tasks_dir))
            if task_name not in valid_tasks:
                valid_tasks.append(task_name)
    return sorted(valid_tasks)


def task_hint(name: str) -> str:
    """A did-you-mean suffix when name is close to an existing task's."""
    close = difflib.get_close_matches(name, task_completer(), 1)
    return f" (did you mean {close[0]!r}?)" if close else ""


class _CompletableAction(Protocol):
    # argparse.Action has no typed `completer`; argcomplete reads it at runtime.
    completer: Callable[..., list[str]]


def set_task_completer(action: argparse.Action) -> None:
    cast(_CompletableAction, action).completer = task_completer


def expand_and_validate_tasks(tasks: list[Task], root_dir: Path,
                              problems_filename: str = "PROBLEMS.txt") -> list[Task]:
    expanded_tasks = []
    task_groups = {task_name: [task for task in tasks if task.name == task_name]
                   for task_name in {task.name for task in tasks}}
    for task_name, tasks in task_groups.items():
        task_path = root_dir / "tasks" / task_name
        assert task_path.exists(), f"Couldn't find directory for task {task_name}"
        assert special_file(task_path / "TASK.py").exists() or special_file(task_path / "TASK.txt").exists(), \
                f"No TASK.py or TASK.txt in {str(task_path)}, not a valid task directory"
        problems_path = special_file(task_path / problems_filename)
        if problems_path.exists():
            with open(problems_path) as f:
                # Allow comments started with the # character
                valid_problems = [line.split('#')[0].strip() for line in f
                                  if line.split('#')[0].strip()]
        else:
            valid_problems = None
        for task in tasks:
            if valid_problems is None:
                if task.problem_id is not None:
                    print(f"Problem {task.name} doesn't have a problems.txt, "
                          f"so you can't pass problem names! Add a problems.txt "
                          f"or just pass the raw task as {task.name} (no colon).")
                    exit(1)
                expanded_tasks.append(task)
            else:
                if task.problem_id is None:
                    print(f"Problem {task.name} needs to be passed a "
                          f"problem id! Use {task.name}:<problem-id> instead, "
                          f"or {task.name}:all to run all problems.")
                    exit(1)
                elif task.problem_id == "all":
                    expanded_tasks.extend([Task(task.name, problem_id)
                                           for problem_id in valid_problems])
                elif "*" in task.problem_id or "?" in task.problem_id:
                    # Pattern matching with fnmatch (Unix-style wildcards)
                    matched = [p for p in valid_problems
                               if fnmatch.fnmatch(p, task.problem_id)]
                    if not matched:
                        print(f"Pattern '{task.problem_id}' matched no problems in {task.name}.")
                        print(f"Available problems: {valid_problems}")
                        exit(1)
                    expanded_tasks.extend([Task(task.name, problem_id)
                                           for problem_id in matched])
                else:
                    assert task.problem_id in valid_problems, \
                            f"Couldn't locate problem {task.problem_id} in problems file"
                    expanded_tasks.append(task)
    return expanded_tasks

@dataclass
class HardwareInfo:
    has_tpu: bool
    has_gpu: bool
    gpu_details: str | None = None

    def info_str(self) -> str | None:
        """Generate hardware info string based on available hardware."""
        if self.has_tpu:
            return """
A v5e TPU (Tensor Processing Unit) is available in this environment.
Standard PyTorch does NOT automatically use TPU - you need `torch_xla`.
The tpu devices are /dev/vfio, not /dev/accel.
Be aware that naive torch_xla code is often inefficient
and triggers too many XLA graph recompilations."""
        if self.has_gpu:
            if self.gpu_details:
                return f"""A GPU is available in this environment: {self.gpu_details}
You should use it for any compute-intensive tasks that can benefit from hardware acceleration."""
            return """A GPU is available in this environment. You should use it for
any compute-intensive tasks that can benefit from hardware acceleration."""
        return None

T = TypeVar('T')
def unwrap(val: T | None) -> T:
    assert val is not None
    return val

def get_latest_opus_str() -> str:
    """Returns the model string of the latest Opus version.

    Assumes that ANTHROPIC_API_KEY is already set in the env as per BWS workflow.

    Raises: RuntimeError if no Opus model is found in the model list.
    """
    import anthropic

    client = anthropic.Anthropic()

    # client.models.list returns model strings newest -> oldest
    #   catching first opus occurence: latest opus
    # expect Opus to be one of the latest released models: grab 5 models at a time
    for model in client.models.list(limit=5):
        model_lower = model.id.lower()
        if "opus" in model_lower:
            return model.id
    raise RuntimeError("No Opus model found in Anthropic models list.")

def special_file(file_path: Path) -> Path:
    # Allow for backwards compatibility by returning the path with the
    #    lowercase special file name if it exists
    lower_path = Path(file_path).with_stem(Path(file_path).stem.lower())
    if Path(file_path).exists():
        return Path(file_path)
    else:
        return lower_path
