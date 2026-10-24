"""Properties of the artifact ref grammar and of the tree record."""

import json
import os
import random
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from hypothesis import assume, given, strategies as st

from artifacts import _ref_str, _resolve, _summary_lines, _text_lines, \
        ContentSha, TreeEntry, tree_record

# Keys name .ARTIFACT_REFS/<key> files; task ids are their tasks/ subpaths.
NAMES = st.text(alphabet="abxyz-_0", min_size=1, max_size=4)
TASK_IDS = st.lists(NAMES, min_size=1, max_size=2).map("/".join)
# The grammar's own punctuation, plus names, is what a ref is built from.
REFS = st.text(alphabet="abxyz-_0#.,/", max_size=8)


@contextmanager
def repo(tasks: dict[str, list[str]], cwd: str = "") -> Iterator[Path]:
    """A checkout whose tasks declare the given artifact keys, entered at cwd."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        (root / ".git").mkdir()  # what repo_root() looks for
        for task_id, keys in tasks.items():
            (root / "tasks" / task_id).mkdir(parents=True)
            (root / "tasks" / task_id / "CONFIG.json").write_text(json.dumps(
                {"artifacts": {k: {"globs": [f"{k}/**"]} for k in keys}}))
        old = Path.cwd()
        os.chdir(root / "tasks" / cwd if cwd else root)
        try:
            yield root
        finally:
            os.chdir(old)


@st.composite
def _keys_and_selection(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    keys = draw(st.lists(NAMES, min_size=1, max_size=4, unique=True))
    return keys, draw(st.lists(st.sampled_from(keys), min_size=1, unique=True))


@given(task_id=TASK_IDS, keys=st.lists(NAMES, min_size=1, max_size=4, unique=True))
def test_a_suggested_ref_resolves_to_what_it_was_rendered_from(
        task_id: str, keys: list[str]) -> None:
    # _ref_str writes the refs the CLI's own messages tell users to run.
    with repo({task_id: keys}) as root:
        task_dir = root / "tasks" / task_id
        for key in keys:
            assert _resolve(_ref_str(task_dir, key), None)[:2] == (task_dir, [key])


@given(task_id=TASK_IDS, selection=_keys_and_selection())
def test_a_comma_list_selects_exactly_the_keys_it_names_in_order(
        task_id: str, selection: tuple[list[str], list[str]]) -> None:
    keys, sel = selection
    with repo({task_id: keys}) as root:
        assert _resolve(f"{task_id}#{','.join(sel)}", None)[1] == sel


@given(task_id=TASK_IDS, keys=st.lists(NAMES, min_size=1, max_size=4, unique=True))
def test_dotdot_selects_every_declared_key(task_id: str, keys: list[str]) -> None:
    with repo({task_id: keys}, cwd=task_id) as root:
        assert _resolve(f"{task_id}#..", None)[1] == sorted(keys)
        assert _resolve("..", None)[1] == sorted(keys)  # bare: never a task id


@given(task_id=TASK_IDS, selection=_keys_and_selection())
def test_the_cwd_spellings_of_a_selector_all_agree(
        task_id: str, selection: tuple[list[str], list[str]]) -> None:
    keys, sel = selection
    joined = ",".join(sel)
    with repo({task_id: keys}, cwd=task_id) as root:
        # the bare spelling reads as a task id first, so it only agrees for
        # selectors that name nothing under tasks/ (this drops "." too)
        assume(not (root / "tasks" / joined).is_dir())
        spellings = [_resolve(ref, None)
                     for ref in (f"#{joined}", f".#{joined}", f"./#{joined}", joined)]
        assert spellings.count(spellings[0]) == len(spellings)


@given(name=NAMES, key=NAMES)
def test_path_pins_the_bare_key_reading_of_a_ref_that_names_a_task(
        name: str, key: str) -> None:
    assume(name != key and "host" not in (name, key))
    with repo({name: [key], "host": [name]}) as root:
        # bare: `name` names a task, so that task's only artifact is the ref
        assert _resolve(name, None)[:2] == (root / "tasks" / name, [key])
        # --path: the same string reads as a key of the task being pointed at
        host = root / "tasks" / "host"
        assert _resolve(name, str(host))[:2] == (host, [name])


@given(ref=REFS, pin_path=st.booleans(), task_id=TASK_IDS,
       keys=st.lists(NAMES, min_size=1, max_size=4, unique=True))
def test_a_resolved_ref_names_declared_keys_of_a_task_under_tasks(
        ref: str, pin_path: bool, task_id: str, keys: list[str]) -> None:
    with repo({task_id: keys}, cwd=task_id) as root:
        try:  # every other ref is the CLI's to reject, not to misresolve
            task_dir, sel, artifacts = _resolve(ref, str(Path.cwd()) if pin_path else None)
        except SystemExit:
            return
        assert task_dir.resolve().is_relative_to(root / "tasks")
        assert sel and set(sel) <= set(artifacts)


@given(files=st.dictionaries(NAMES, st.binary(max_size=8), min_size=1, max_size=4),
       shuffle=st.randoms(use_true_random=False))
def test_the_tree_record_is_independent_of_the_order_files_come_in(
        files: dict[str, bytes], shuffle: random.Random) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task_dir = Path(tmp)
        for name, data in files.items():
            (task_dir / name).write_bytes(data)
        order = list(files)
        shuffle.shuffle(order)
        assert tree_record(task_dir, order)[1:] == tree_record(task_dir, sorted(files))[1:]


@given(data=st.binary())
def test_diffable_content_survives_being_split_into_lines(data: bytes) -> None:
    lines = _text_lines(data)
    assert lines is None or "".join(lines).encode() == data


def _tree(contents: dict[str, str]) -> dict[str, TreeEntry]:
    """The `_summary_lines` view of a tree: each path and the sha of its content."""
    return {p: TreeEntry(p, ContentSha(sha), len(sha), 0o644)
            for p, sha in contents.items()}


@given(a=st.dictionaries(NAMES, NAMES, max_size=4),
       b=st.dictionaries(NAMES, NAMES, max_size=4))
def test_every_changed_path_gets_exactly_one_summary_line(
        a: dict[str, str], b: dict[str, str]) -> None:
    lines = _summary_lines(_tree(a), _tree(b))
    changed = {p for p in a.keys() | b.keys() if a.get(p) != b.get(p)}
    # a rename names both of its paths on one line, every other change one path
    assert sum(2 if line.startswith("R ") else 1 for line in lines) == len(changed)


def test_the_summary_of_a_rename_reads_the_way_jj_prints_one() -> None:
    assert _summary_lines(_tree({"d/a.txt": "same"}), _tree({"d/b.txt": "same"})) \
        == ["R d/{a.txt => b.txt}"]
    assert _summary_lines(_tree({"a.txt": "same"}), _tree({"d/b.txt": "same"})) \
        == ["R {a.txt => d/b.txt}"]
