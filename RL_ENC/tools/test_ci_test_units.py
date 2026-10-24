from ci_test_units import gate_of, is_test, markers_of, plan, unit_of

# Copied from the real repo.
MARKERS = frozenset(
    {
        "tasks/leace",
        "tasks/prune",
        "tasks/neuron-descriptions-v2",
        "tasks/transportable-edit",
        "tasks/transportable-edit/docker",
        "tasks/sprint/neuron-sae-hybrid-task",
        "llm/jeffs_tools",
    }
)


def test_markers_hold_a_dockerfile_or_pyproject() -> None:
    assert markers_of(
        [
            "pyproject.toml",
            "tasks/leace/pyproject.toml",
            "tasks/prune/Dockerfile",
            "tasks/transportable-edit/docker/Dockerfile.base",
            "tasks/leace/grader/grader.py",
            "tasks/leace/tests/pyproject.toml.bak",
        ]
    ) == {"tasks/leace", "tasks/prune", "tasks/transportable-edit/docker"}


def test_is_test_matches_both_globs() -> None:
    assert is_test("tasks/leace/tests/test_logit_signs.py")
    assert is_test("tasks/sandbagging-circuit-localization/e2e_test.py")
    assert not is_test("tasks/fluent-dreaming/tests/conftest.py")
    assert not is_test("tasks/prune/grader_data/test_cases.json")


def test_unit_is_the_nearest_marker_directory() -> None:
    assert unit_of("tasks/leace/tests/test_logit_signs.py", MARKERS) == "tasks/leace"
    assert unit_of("tasks/prune/grader/test_grader.py", MARKERS) == "tasks/prune"


def test_the_deepest_marker_directory_wins() -> None:
    docker = "tasks/transportable-edit/docker"
    assert unit_of(f"{docker}/tests/test_entrypoint.py", MARKERS) == docker


def test_nested_task_groups_do_not_collapse() -> None:
    # tasks/sprint only groups tasks; collapsing to it would put two flat
    # grader.py modules in one process.
    hybrid = "tasks/sprint/neuron-sae-hybrid-task"
    assert unit_of(f"{hybrid}/grader/test_grader.py", MARKERS) == hybrid


def test_unmarked_task_falls_back_to_the_second_level() -> None:
    assert (
        unit_of("tasks/binding-types/test_model_binding.py", MARKERS)
        == "tasks/binding-types"
    )
    assert (
        unit_of("llm/adams_tools/runner/test_runner_tools.py", MARKERS)
        == "llm/adams_tools"
    )


def test_standalone_directories_do_not_reach_the_repo_root() -> None:
    assert unit_of("docs/ARTIFACTS.md", MARKERS) == "docs"
    assert plan(["docs/ARTIFACTS.md"], ["tools/test_ci_test_units.py"], MARKERS) == []


def test_infra_belongs_to_the_repo_root() -> None:
    assert unit_of("tools/ci_test_units.py", MARKERS) == "."
    assert unit_of("docker/grader/run_grader.py", MARKERS) == "."
    assert unit_of("tasks/conftest.py", MARKERS) == "."


def test_exploratory_directories_are_gated() -> None:
    assert (
        gate_of("tasks/prune/llm/dev_scripts/test_x.py", "tasks/prune")
        == "tasks/prune/llm"
    )
    nd = "tasks/neuron-descriptions-v2"
    assert (
        gate_of(f"{nd}/ss35m-validation/test_ss35m_validation.py", nd)
        == f"{nd}/ss35m-validation"
    )
    assert gate_of("tasks/leace/tests/test_logit_signs.py", "tasks/leace") is None
    # llm/jeffs_tools is its own unit, so the exploratory `llm` segment never
    # appears in the relative path.
    assert (
        gate_of("llm/jeffs_tools/tests/test_clean_transcript.py", "llm/jeffs_tools")
        is None
    )


def test_plan_runs_only_the_units_that_changed() -> None:
    tests = ["tasks/leace/tests/test_a.py", "tasks/prune/grader/test_b.py"]
    assert plan(["tasks/leace/Dockerfile"], tests, MARKERS) == [
        {"dir": "tasks/leace", "files": "tests/test_a.py"}
    ]


def test_plan_is_empty_when_no_unit_with_tests_changed() -> None:
    # unit-tests.yml skips the matrix by string-comparing this, serialized, to '[]'.
    assert (
        plan(["tasks/prune/grader.py"], ["tasks/leace/tests/test_a.py"], MARKERS) == []
    )


def test_plan_gates_exploratory_tests() -> None:
    tests = ["tasks/prune/llm/dev_scripts/test_x.py"]
    assert plan(["tasks/prune/grader.py"], tests, MARKERS) == []
    assert plan(["tasks/prune/llm/dev_scripts/run.py"], tests, MARKERS) == [
        {"dir": "tasks/prune", "files": "llm/dev_scripts/test_x.py"}
    ]


def test_plan_sorts_units() -> None:
    tests = ["tasks/prune/grader/test_b.py", "tasks/leace/tests/test_a.py"]
    changed = ["tasks/prune/grader.py", "tasks/leace/grader.py"]
    assert [entry["dir"] for entry in plan(changed, tests, MARKERS)] == [
        "tasks/leace",
        "tasks/prune",
    ]


def test_marker_directory_nested_in_a_task_still_runs_the_task_tests() -> None:
    # tasks/transportable-edit/docker holds a Dockerfile, so it is a unit of its
    # own with no tests; the task's tests must not fall through the gap.
    tests = ["tasks/transportable-edit/grader/tests/test_grader_scoring.py"]
    assert plan(["tasks/transportable-edit/docker/Dockerfile"], tests, MARKERS) == [
        {
            "dir": "tasks/transportable-edit",
            "files": "grader/tests/test_grader_scoring.py",
        }
    ]


def test_plan_runs_infra_tests_when_infra_changes() -> None:
    tests = ["tools/test_ci_test_units.py", "tasks/leace/tests/test_a.py"]
    assert plan(["tools/task_runner.py"], tests, MARKERS) == [
        {"dir": ".", "files": "tools/test_ci_test_units.py"}
    ]
