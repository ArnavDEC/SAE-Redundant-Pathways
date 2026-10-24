"""Pure helpers for the readonly-volume mount feature: content-hashing a
mount subdir and resolving which subdirs/bases a problem needs from MOUNTS_CONFIG.jsonc."""

import os
import json5
import fnmatch
import hashlib
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from typing import NamedTuple

from humanfriendly import format_size
from tqdm import tqdm

from common import repo_root

HASH_LEN = 16  # hex chars of a content hash used to name/key a volume
HASH_WORKERS = 8  # parallel file readers; ~8 saturates disk, more just thrashes it


class Symlink(NamedTuple):
    rel: str
    target: str


class Inode(NamedTuple):
    rel: str
    key: tuple[int, int]  # (st_dev, st_ino)


def hash_subdirs(subdirs: list[Path]) -> dict[Path, str]:
    """Content hash per subdir: sha256 over its sorted (relpath, content-id) entries."""
    files: dict[Path, list[Symlink | Inode]] = {}
    inode_size: dict[tuple[int, int], int] = {}
    inode_rep: dict[tuple[int, int], Path] = {}
    for sd in subdirs:
        entries: list[Symlink | Inode] = []
        for root, _dirs, names in os.walk(sd, followlinks=False):
            for n in names:
                p = Path(root) / n
                rel = str(p.relative_to(sd))
                if p.is_symlink():
                    # record a symlink by target, don't follow: the blob it points at is
                    # hashed via blobs/, so following would read every weight shard twice
                    entries.append(Symlink(rel, os.readlink(p)))
                else:
                    st = p.lstat()
                    key = (st.st_dev, st.st_ino)
                    entries.append(Inode(rel, key))
                    inode_size.setdefault(key, st.st_size)   # dedup hardlinks: read each inode once
                    inode_rep.setdefault(key, p)
        files[sd] = entries

    total = sum(inode_size.values())
    print(f"[hash] {len(subdirs)} subdir(s), {len(inode_size)} unique files, "
          f"{format_size(total)} to read", flush=True)

    def hash_one(item: tuple[tuple[int, int], Path]) -> tuple[tuple[int, int], str]:
        key, p = item
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                progress.update(len(chunk))
        return key, h.hexdigest()

    digest: dict[tuple[int, int], str] = {}
    with tqdm(total=total, unit="B", unit_scale=True, desc="[hash]") as progress, \
            ThreadPoolExecutor(max_workers=min(HASH_WORKERS, os.cpu_count() or 1)) as ex:
        for key, d in ex.map(hash_one, inode_rep.items()):
            digest[key] = d

    out = {}
    for sd, entries in files.items():
        manifest = "\n".join(sorted(
            f"{e.rel}\tL\t{e.target}" if isinstance(e, Symlink)
            else f"{e.rel}\tI\t{digest[e.key]}"
            for e in entries
        ))
        out[sd] = hashlib.sha256(manifest.encode()).hexdigest()[:HASH_LEN]
    return out


def _task_dir(task: str) -> Path:
    return repo_root() / "tasks" / task


def _mounts_config_path(task: str) -> Path | None:
    """The task's MOUNTS_CONFIG.json/.jsonc, or None if it has neither."""
    task_dir = _task_dir(task)
    found = [p for p in (task_dir / "MOUNTS_CONFIG.json", task_dir / "MOUNTS_CONFIG.jsonc")
             if p.exists()]
    if len(found) > 1:
        raise FileExistsError(f"Both MOUNTS_CONFIG.json and .jsonc exist in {task_dir}; pick one")
    return found[0] if found else None


def mounts_root(task: str) -> Path | None:
    """The task's MOUNTS/ dir (None if no mounts). Error if the mounts are missing from disk."""
    root = _task_dir(task) / "MOUNTS"
    if root.is_dir():
        return root
    cfg = _mounts_config_path(task)
    if cfg is not None:
        raise FileNotFoundError(
            f"{cfg} declares mounts but {root} does not exist; "
            "build it (the task's prepare_mounts, or its artifact) before running")
    return None


def problem_subdirs(task: str, problem_id: str | None) -> dict[str, str]:
    """{top-level subdir in MOUNTS/ -> its base under /opt} that this problem needs.

    Each subdir is packed into its own squashfs and mounted at /opt/<base>/<subdir>. By
    default a problem gets every top-level dir in MOUNTS/ at /opt/<subdir>; an optional
    tasks/<task>/MOUNTS_CONFIG.jsonc groups dirs under shared bases and limits which subdirs
    reach which problems:
      {"mount_base": {base: [subdir, ...]}, "include": {problem_id glob: [subdir, ...]}}.
    Every subdir named in either map must be a top-level dir in MOUNTS/; a subdir under no
    base defaults to "" (mounted at /opt/<subdir>)."""
    root = mounts_root(task)
    if root is None:
        return {}
    tops = {p.name for p in root.iterdir() if p.is_dir()}
    cfg_path = _mounts_config_path(task)
    if cfg_path is None:
        return {s: "" for s in tops}
    cfg = json5.loads(cfg_path.read_text())
    base_of = {d: base for base, dirs in cfg["mount_base"].items() for d in dirs}
    # "*" also covers the variant-less case (no PROBLEMS.txt -> problem_id is None);
    # any other glob needs a concrete problem id to match.
    subs = {s for pat, subdirs in cfg["include"].items()
            if (pat == "*" if problem_id is None else fnmatch.fnmatch(problem_id, pat))
            for s in subdirs}
    unknown = (subs | set(base_of)) - tops
    if unknown:
        raise ValueError(f"{cfg_path}: names subdirs absent from MOUNTS/: {sorted(unknown)}")
    return {s: base_of.get(s, "") for s in subs}
