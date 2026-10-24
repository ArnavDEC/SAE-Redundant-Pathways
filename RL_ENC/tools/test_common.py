from pathlib import Path

import pytest
from pydantic import ValidationError

import common
from common import TaskConfig, build_in_task_dir, load_task_config


def test_files_are_mirrored_before_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "docker" / "grader_lib").mkdir(parents=True)
    (tmp_path / "docker" / "grader_lib" / "safe_read.py").write_text("# shared\n")
    task_dir = tmp_path / "tasks" / "demo"
    task_dir.mkdir(parents=True)
    monkeypatch.setattr(common, "repo_root", lambda: tmp_path)
    (task_dir / "CONFIG.json").write_text(
        '{"mirrored_repo_files": {"docker/grader_lib/safe_read.py": "reference/x.py"}}'
    )
    # Relative task_dir: the copy must not resolve its destination against the build cwd.
    monkeypatch.chdir(task_dir.parent)
    # reference/ doesn't exist yet, so this also covers creating the destination dir.
    build_in_task_dir(["test", "-e", "reference/x.py"], Path(task_dir.name))


def test_missing_mirror_source_names_the_config_entry(tmp_path: Path) -> None:
    (tmp_path / "CONFIG.json").write_text('{"mirrored_repo_files": {"gone.py": "x.py"}}')
    with pytest.raises(FileNotFoundError, match="'gone.py'"):
        build_in_task_dir(["true"], tmp_path)


def test_build_runs_without_any_mirrored_files(tmp_path: Path) -> None:
    build_in_task_dir(["true"], tmp_path)


def test_a_misspelled_config_key_is_not_silently_ignored() -> None:
    with pytest.raises(ValidationError, match="nework"):
        TaskConfig.model_validate({"nework": True})


def test_an_artifact_key_must_name_a_ref_file() -> None:
    with pytest.raises(ValidationError, match="mounts.lock"):
        TaskConfig.model_validate({"artifacts": {"mounts.lock": {"globs": ["M/**"]}}})


def test_spelling_a_deprecated_key_both_ways_is_an_error() -> None:
    with pytest.raises(ValidationError, match="grader_timeout"):
        TaskConfig.model_validate(
            {"grader_timeout": 60, "remote_problem": {"grading_timeout_seconds": 90}})


def test_a_deprecated_key_warns_and_still_applies(
    capsys: pytest.CaptureFixture[str]
) -> None:
    config = TaskConfig.model_validate({"grader_timeout": 60})
    assert config.remote_problem == {"grading_timeout_seconds": 60}
    assert "grader_timeout is deprecated" in capsys.readouterr().err


def test_a_null_remote_problem_is_rejected() -> None:
    with pytest.raises(ValidationError, match="remote_problem must be an object"):
        TaskConfig.model_validate({"remote_problem": None})


def test_tmux_is_opt_out() -> None:
    assert "tmux" not in TaskConfig.model_validate({}).disallowed_tools
    without_tmux = TaskConfig.model_validate({"enable_tmux": False})
    assert "tmux" in without_tmux.disallowed_tools


def test_template_task_mirrors_sources_that_exist() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_task_config(root / "tasks" / "template-task")
    for source in config.mirrored_repo_files:
        assert (root / source).exists(), (
            f"template-task mirrors {source!r}, which the repo lacks"
        )
