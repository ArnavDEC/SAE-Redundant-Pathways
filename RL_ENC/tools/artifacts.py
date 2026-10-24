#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Content-addressed records of the non-source-controlled files tasks are built from.

Store layout:
  by-content/<sha256>  zstd-compressed file blobs, named by the sha256 of the
                       raw content (compression is invisible to the addresses)
  by-tree/<sha256>     tree records: plain canonical JSON [{path, sha256, size, mode}, ...]

A task declares artifacts in its CONFIG.json:
  {"artifacts": {"<key>": {"globs": ["model_cache/**"], "build": "prepare.sh"}}}
and binds to a specific record via the source-controlled file .ARTIFACT_REFS/<key>
(content "by-tree/<sha>\\n"), written by `record` and consumed by the runners.
.ARTIFACT_REFS/<key>.cache (gitignored) is a local stat cache so fetches of an
unchanged tree skip rehashing it.

Symlinks are rejected by `record` — docker COPY would bake the link, not its
content, into the image. Hardlinks are fine: each is recorded as its own entry.
"""

import argcomplete
import argparse
import contextlib
import difflib
import fcntl
import functools
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NewType, TextIO, TypeVar

import pathspec
import zstandard
from pydantic import ValidationError
from tqdm import tqdm

from common import ArtifactSpec, RepoError, Task, confirm, git_cmd, \
        load_task_config, repo_root, task_hint, unwrap

# Bucket lives in GCP project gen-lang-client-0029335129 ("research"); a path
# prefix in the URL scopes all keys under it (used for scratch/test runs).
ARTIFACT_STORE_URL = os.environ.get("RL_ENV_ARTIFACT_STORE",
                                    "gs://dmodel-rl-env-artifacts")
REFS_DIR = ".ARTIFACT_REFS"
HASH_WORKERS = 8  # parallel file readers; ~8 saturates disk, more just thrashes it
TRANSFER_WORKERS = 32  # GCS streams are latency-bound; disk tuning doesn't apply
_ABORT = threading.Event()  # failed map: chunk loops stop instead of finishing multi-GB work
_T = TypeVar("_T")
MAX_TEXT_DIFF = 1 << 20  # diff prints larger files as a one-line note
_CLI = Path(__file__).resolve()  # suggested commands then work from any cwd


class ArtifactError(Exception):
    """Record/fetch failure with a user-facing message; the CLI and runner
    boundaries print it and exit instead of unwinding a traceback."""


# Both are bare hex, so only the type says which namespace a value names;
# mixing them up would otherwise surface only as a runtime store miss.
TreeSha = NewType("TreeSha", str)        # sha256 of a tree record (by-tree/<sha>)
ContentSha = NewType("ContentSha", str)  # sha256 of raw file bytes (by-content/<sha>)


@dataclass(frozen=True)
class TreeEntry:
    """One recorded file: task-dir-relative path, content sha256, size, and
    normalized (755/644) mode."""
    path: str
    sha256: ContentSha
    size: int
    mode: int


class Store:
    def __init__(self, url: str = ARTIFACT_STORE_URL) -> None:
        from google.cloud import storage  # type: ignore[import-untyped]  # lazy: artifact-free runs never need GCS auth
        bucket, _, prefix = url.removeprefix("gs://").partition("/")
        self.prefix = f"{prefix}/" if prefix else ""
        self.bucket = storage.Client().bucket(bucket)

    def exists(self, name: str) -> bool:
        return bool(self.bucket.blob(self.prefix + name).exists())

    def upload_blob(self, sha256: ContentSha, path: Path,
                    tick: Callable[[int], object] | None = None) -> None:
        """Compress path into by-content/<sha256>, hashing the raw bytes as
        they stream: a mid-record write to path would otherwise poison the
        immutable blob forever. Any exception aborts the upload cleanly."""
        h = hashlib.sha256()
        blob = self.bucket.blob(self.prefix + "by-content/" + sha256)
        with open(path, "rb") as src, blob.open("wb") as dst:
            # level 10: record-time cost only, decompression speed is unaffected
            with zstandard.ZstdCompressor(level=10).stream_writer(
                    dst, closefd=False) as comp:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    if _ABORT.is_set():
                        raise KeyboardInterrupt
                    h.update(chunk)
                    comp.write(chunk)
                    if tick:
                        tick(len(chunk))
            if h.hexdigest() != sha256:
                raise ArtifactError(f"{path} changed while being recorded; "
                                    "rerun record once nothing is writing to it")

    def upload_bytes(self, name: str, data: bytes) -> None:
        self.bucket.blob(self.prefix + name).upload_from_string(data)

    def download_bytes(self, name: str) -> bytes:
        from google.api_core.exceptions import NotFound  # lazy, as in __init__
        try:
            data: bytes = self.bucket.blob(self.prefix + name).download_as_bytes()
            return data
        except NotFound:
            raise _missing(name) from None

    def download_blob(self, sha256: ContentSha, path: Path,
                      tick: Callable[[int], object] | None = None) -> ContentSha:
        """Decompress by-content/<sha256> into path, returning the sha256 of
        the bytes written — hashing the stream spares callers rereading
        multi-GB files to verify."""
        from google.api_core.exceptions import NotFound  # lazy, as in __init__
        h = hashlib.sha256()
        try:
            with self.bucket.blob(self.prefix + "by-content/" + sha256).open("rb") as src, \
                    open(path, "wb") as dst:
                reader = zstandard.ZstdDecompressor().stream_reader(src)
                for chunk in iter(lambda: reader.read(1 << 20), b""):
                    if _ABORT.is_set():
                        raise KeyboardInterrupt
                    h.update(chunk)
                    dst.write(chunk)
                    if tick:
                        tick(len(chunk))
        except NotFound:
            raise _missing("by-content/" + sha256) from None
        return ContentSha(h.hexdigest())


def _missing(name: str) -> ArtifactError:
    return ArtifactError(f"{name} isn't in the artifact store "
                         f"({ARTIFACT_STORE_URL}); was it recorded to a "
                         "different store? (RL_ENV_ARTIFACT_STORE overrides "
                         "the default)")


def _corrupt(name: str) -> ArtifactError:
    return ArtifactError(f"{name}: content doesn't hash to its name; the "
                         "artifact store is corrupt or tampered with")


@functools.cache
def open_store() -> Store:
    """Cached so lazy callers share one client no matter how often they call."""
    return Store()


@contextlib.contextmanager
def _store_boundary() -> Iterator[None]:
    """CLI/runner boundary: translate store and repo failures into clean exits."""
    from google.api_core.exceptions import GoogleAPICallError  # lazy, as in Store
    from google.auth.exceptions import GoogleAuthError
    from requests.exceptions import RequestException
    try:
        yield
    except KeyboardInterrupt:  # a bare traceback reads as a crash, not a pause
        raise SystemExit("interrupted; rerun the same command to resume") from None
    except GoogleAuthError as e:  # missing/expired ADC, at construction or first RPC
        raise SystemExit("no valid credentials for the artifact store; run:\n"
                         "  gcloud auth application-default login") from e
    except (GoogleAPICallError, RequestException) as e:  # 403s, 5xx, flaky network
        raise SystemExit(f"artifact store request failed: {e}") from e
    except (ArtifactError, RepoError) as e:
        raise SystemExit(str(e)) from None
    except OSError as e:  # disk full, permissions: the message, minus the traceback
        raise SystemExit(str(e)) from e


def _sha256_file(path: Path, tick: Callable[[int], object] | None = None) -> ContentSha:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            if _ABORT.is_set():
                raise KeyboardInterrupt
            h.update(chunk)
            if tick:
                tick(len(chunk))
    return ContentSha(h.hexdigest())


def _abort_fast(ex: ThreadPoolExecutor, results: Iterable[_T]) -> list[_T]:
    """Drain an executor map; on any failure (^C included) flag the chunk
    loops to stop and drop queued work, so the executor's join returns in
    about one chunk instead of after the in-flight multi-GB transfers."""
    _ABORT.clear()
    try:
        return list(results)
    except BaseException:
        _ABORT.set()
        ex.shutdown(cancel_futures=True)
        raise


def matched_files(task_dir: Path, globs: list[str]) -> tuple[list[str], list[str]]:
    """Task-dir-relative (files, symlinked dirs) matching the gitignore-style
    globs, both sorted. Symlinked dirs aren't traversed, so anything behind one
    is invisible to a record — callers reject them."""
    spec = pathspec.PathSpec.from_lines("gitwildmatch", globs)
    files, dir_links = [], []
    for root, dirs, names in os.walk(task_dir):
        dirs[:] = [d for d in dirs if d != REFS_DIR]  # never record the refs/cache
        for d in dirs:
            rel = str((Path(root) / d).relative_to(task_dir))
            # the trailing "/" makes gitwildmatch treat the path as a directory
            if (Path(root) / d).is_symlink() and spec.match_file(rel + "/"):
                dir_links.append(rel)
        for name in names:
            rel = str((Path(root) / name).relative_to(task_dir))
            if spec.match_file(rel):
                files.append(rel)
    return sorted(files), sorted(dir_links)


def tree_record(task_dir: Path, files: list[str],
                sha: Callable[[Path], ContentSha] = _sha256_file,
                ) -> tuple[list[TreeEntry], bytes, TreeSha]:
    """The files' entries, their canonical JSON record, and its sha256 (the tree sha)."""
    def entry(rel: str) -> TreeEntry:
        path = task_dir / rel
        st = path.stat()
        # mode is normalized to 755/644 so the recorder's umask can't skew the sha
        return TreeEntry(rel, sha(path), st.st_size,
                         0o755 if st.st_mode & 0o100 else 0o644)
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as ex:
        entries = _abort_fast(ex, ex.map(entry, sorted(files)))
    # sort_keys makes the field order canonical, so the tree sha is stable.
    data = json.dumps([asdict(e) for e in entries], sort_keys=True,
                      separators=(",", ":")).encode()
    return entries, data, TreeSha(hashlib.sha256(data).hexdigest())


def _claim_writer(marker: Path, ref: str) -> TextIO:
    """The artifact's writer lock: at most one fetch or record at a time.
    It's a flock on <marker>.lock, so the OS releases it however the holder
    exits; the pid written inside only feeds the already-running message."""
    marker.parent.mkdir(exist_ok=True)
    fd = os.open(f"{marker}.lock", os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pid = os.read(fd, 32).decode("ascii", "replace").strip() or "?"
        os.close(fd)
        raise ArtifactError(f"another fetch or record of {ref} is already "
                            f"running (pid {pid})") from None
    lock = os.fdopen(fd, "r+")
    lock.truncate(0)
    lock.write(str(os.getpid()))
    lock.flush()
    return lock


def _fetch_tree(store: Store, tree_sha: TreeSha) -> list[TreeEntry]:
    data = store.download_bytes(f"by-tree/{tree_sha}")
    if hashlib.sha256(data).hexdigest() != tree_sha:
        raise _corrupt(f"by-tree/{tree_sha}")
    return [TreeEntry(**e) for e in json.loads(data)]


def _blob_bytes(store: Store, sha256: ContentSha) -> bytes:
    try:
        data = zstandard.ZstdDecompressor().stream_reader(
            io.BytesIO(store.download_bytes("by-content/" + sha256))).read()
    except zstandard.ZstdError:  # truncated or garbage stream: treat as corrupt
        data = None
    if data is None or hashlib.sha256(data).hexdigest() != sha256:
        raise _corrupt("by-content/" + sha256)
    return data


def _ref_str(task_dir: Path, key: str) -> str:
    """This artifact's CLI ref ('<task-id>#<key>'), for status prefixes and
    suggested commands; main() scopes every task dir to tasks/."""
    return f"{task_dir.resolve().relative_to(repo_root() / 'tasks')}#{key}"


def _stat_cache(cache_path: Path) -> dict[str, Any]:
    try:  # the cache is disposable; missing or corrupt just means rehash
        cache: dict[str, Any] = json.loads(cache_path.read_text())
        return cache
    except (OSError, ValueError):
        return {}


def _cached_sha(cache: dict[str, Any], task_dir: Path,
                tick: Callable[[int], object] | None = None,
                tick_skipped: Callable[[int], object] | None = None,
                ) -> Callable[[Path], ContentSha]:
    """sha256 via the stat cache, memoizing misses into `cache` so an update
    fetch hashes each file once. For progress bars: tick gets reread bytes per
    chunk, tick_skipped gets whole-file sizes on cache hits."""
    def sha(path: Path) -> ContentSha:
        rel = str(path.relative_to(task_dir))
        c: dict[str, Any] | None = cache.get(rel)
        st = path.stat()
        if not (c and c["size"] == st.st_size and c["mtime_ns"] == st.st_mtime_ns):
            c = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                 "sha256": _sha256_file(path, tick)}
            cache[rel] = c
        elif tick_skipped:
            tick_skipped(st.st_size)
        return ContentSha(c["sha256"])
    return sha


def _write_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_text("")  # probe: "now" on the file clock
    now = cache_path.stat().st_mtime_ns

    # drop entries whose mtime equals now"; same-size same-tick writes
    # pass the stat match without a rehash (git's racy-index smudge)
    cache_path.write_text(json.dumps(
        {k: v for k, v in cache.items() if v["mtime_ns"] < now}))


def _n_files(n: int) -> str:
    return f"{n} file{'' if n == 1 else 's'}"


class _SkipBar(tqdm):  # type: ignore[type-arg]
    """Byte bar where skipped bytes (tick_skipped) fill green from the right,
    so a warm run reads as mostly-green at a glance."""
    _WIDTH = 20  # fixed: {bar2} bypasses tqdm's width fitting; keep <80 cols

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.skipped = 0
        kwargs["bar_format"] = ("{l_bar}{bar2}| {n_fmt}/{total_fmt} "
                                "[{elapsed}<{remaining}, {rate_fmt}]")
        super().__init__(*args, **kwargs)

    def tick_skipped(self, n: int) -> None:
        self.update(n)  # before skipped += n, so skipped never outruns n
        self.skipped += n

    @property
    def format_dict(self) -> MutableMapping[str, Any]:
        skipped = self.skipped  # snapshot before n: keeps done >= 0 under threads
        d = super().format_dict
        total = d["total"] or 1  # all-empty tree: avoid 0/0
        skip = round(self._WIDTH * skipped / total)
        done = round(self._WIDTH * (d["n"] - skipped) / total)
        d["bar2"] = ("█" * done + " " * (self._WIDTH - done - skip)
                     + "\x1b[32m" + "█" * skip + "\x1b[0m")
        return d


def _hashed(ref: str, task_dir: Path, files: list[str], cache: dict[str, Any],
            ) -> tuple[list[TreeEntry], bytes, TreeSha]:
    """tree_record with a pre-line and byte bar: hashing a cold multi-GB tree
    takes minutes and would otherwise look hung. Cache hits tick whole files,
    so a warm pass flashes straight to done."""
    total = sum((task_dir / f).stat().st_size for f in files)
    print(f"[artifacts] {ref}: hashing {_n_files(len(files))} "
          f"({total / 1e6:,.1f} MB)", file=sys.stderr)
    with _SkipBar(total=total, unit="B", unit_scale=True, desc="[hash]",
                  disable=None) as bar:
        return tree_record(task_dir, files,
                           _cached_sha(cache, task_dir, bar.update, bar.tick_skipped))


def _recorded_sha(task_dir: Path, key: str) -> TreeSha:
    ref_path = task_dir / REFS_DIR / key
    if not ref_path.exists():
        raise ArtifactError(f"artifact {key!r} has no record ref at "
                            f"{ref_path}; pull the commit that recorded it, "
                            "or build and upload one with:\n"
                            f"  {_CLI} record {_ref_str(task_dir, key)}")
    return TreeSha(ref_path.read_text().strip().removeprefix("by-tree/"))


def list_records(task_dir: Path, key: str, commits: bool) -> None:
    """Print each distinct recorded tree, newest first (by first pin). Purely
    local: the pin history is the git history of the ref file."""
    root = repo_root()
    ref_path = task_dir / REFS_DIR / key
    rel = str(ref_path.relative_to(root))
    git = git_cmd(root)  # hoisted: the jj branch shells out per call
    # d: skip deletes, which have no blob to show; full-history + first-parent
    # diffs: don't simplify away pins that landed via merge commits
    log = subprocess.run(
        git + ["log", "--reverse", "--full-history", "--diff-filter=d",
               "--diff-merges=first-parent", "--no-patch",
               "--format=%h %as", "--", rel],
        cwd=root, capture_output=True, text=True, check=True)
    first: dict[str, str] = {}  # tree ref -> "hash date" that first pinned it
    for line in log.stdout.splitlines():
        tree = subprocess.run(
            git + ["show", f"{line.split()[0]}:{rel}"],
            cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        first.setdefault(tree, line)
    pinned = ref_path.read_text().strip() if ref_path.exists() else ""
    if pinned:
        first.setdefault(pinned, "uncommitted")
    if not first:
        _recorded_sha(task_dir, key)  # no ref file and no history: raises
    for tree, c in reversed(first.items()):
        print(tree + (f"  {c}" if commits else "")
              + ("  (pinned)" if tree == pinned else ""))


def plan(tasks: list[Task], root_dir: Path
         ) -> list[tuple[Path, str, ArtifactSpec, TreeSha]]:
    """(task_dir, key, spec, tree_sha) for every artifact the tasks declare,
    deduped by task name so a task expanded into many problems fetches once.
    Reading every ref up front means a missing record fails before any
    downloads start."""
    items = []
    for task in {t.name: t for t in tasks}.values():  # one entry per task name
        task_dir = root_dir / "tasks" / task.name
        for key, spec in unwrap(task.config).artifacts.items():
            items.append((task_dir, key, spec, _recorded_sha(task_dir, key)))
    return items


def fetch(store: Callable[[], Store], task_dir: Path, key: str, spec: ArtifactSpec,
          tree_sha: TreeSha, clear_stale: bool = False) -> None:
    """Make local files exactly match the recorded tree, downloading mismatches
    and deleting files the record doesn't contain.

    The store is opened lazily: an up-to-date tree needs no GCS access (or
    credentials) at all. Unrecorded local state is overwritten after a prompt,
    or unconditionally when `clear_stale` is `True`."""
    ref = _ref_str(task_dir, key)
    # the writer lock spans the whole fetch, so fetches and records never
    # interleave; a leftover marker means the write phase was interrupted
    # and the mismatched disk state is debris
    marker = task_dir / REFS_DIR / f"{key}.fetching"
    _lock = _claim_writer(marker, ref)  # released when fetch exits
    # only an empty marker is fetch debris: record's ('record') or garbage
    # content still gets the overwrite prompt below
    resuming = marker.is_file() and marker.read_text() == ""
    cache_path = task_dir / REFS_DIR / f"{key}.cache"
    cache = _stat_cache(cache_path)
    cached_sha = _cached_sha(cache, task_dir)

    disk, dir_links = matched_files(task_dir, spec.globs)
    if dir_links:
        raise ArtifactError(f"{ref}: symlinked dir(s) in the tree "
                            f"({', '.join(dir_links)}); records never hold "
                            "symlinks — remove them and refetch")
    # hashing follows symlinks and records never hold them, so a symlinked
    # tree matches no record; treat it as unrecorded local state
    links = [f for f in disk if (task_dir / f).is_symlink()]
    disk_is_recorded = False
    disk_entries: list[TreeEntry] = []
    disk_sha: TreeSha | None = None
    if disk and not links:
        disk_entries, _, disk_sha = _hashed(ref, task_dir, disk, cache)
        _write_cache(cache_path, cache)  # persist the hash work now: any exit keeps it
        if disk_sha == tree_sha:
            for e in disk_entries:  # mode normalization (600→644) can hide drift; re-pin
                (task_dir / e.path).chmod(e.mode)
            marker.unlink(missing_ok=True)
            print(f"[artifacts] {ref}: up to date")
            return
        disk_is_recorded = store().exists(f"by-tree/{disk_sha}")
    try:
        entries = _fetch_tree(store(), tree_sha)
    except ArtifactError as exc:  # a bare store name doesn't say which artifact
        raise ArtifactError(f"{ref}: {exc}") from None
    globs = pathspec.PathSpec.from_lines("gitwildmatch", spec.globs)
    if outside := sorted(e.path for e in entries
                         if not globs.match_file(e.path)):
        raise ArtifactError(
            f"artifact {key!r}: the declared globs no longer match "
            f"{len(outside)} recorded file(s), e.g. {outside[0]}; record the "
            f"artifact with the new globs:\n  {_CLI} record {ref}")
    drift = ""  # how disk differs from the target record, for the messages below
    if disk and links:
        drift = f"symlink(s) on disk, e.g. {links[0]}; records never hold them"
    elif disk and not disk_is_recorded:
        rec = {e.path: (e.sha256, e.mode) for e in entries}
        dsk = {e.path: (e.sha256, e.mode) for e in disk_entries}
        names = sorted(p for p in rec.keys() | dsk.keys()
                       if rec.get(p) != dsk.get(p))
        drift = f"{len(names)} differing file(s), e.g. {', '.join(names[:3])}"
    if disk_is_recorded:
        print(f"[artifacts] {ref}: updating from recorded "
              f"by-tree/{disk_sha} to by-tree/{tree_sha}")
    elif disk and resuming:
        print(f"[artifacts] {ref}: resuming interrupted fetch ({drift})")
    elif disk:
        # no diff suggestion for symlinks: diff refuses them too
        inspect = ("" if links else
                   f"; inspect with:\n  {_CLI} diff {ref}")
        print(f"WARNING: {ref}: the files on disk don't match "
              f"any recorded tree ({drift}){inspect}\n"
              "fetching would overwrite or delete work you may have "
              "meant to record with:\n"
              f"  {_CLI} record {ref}",
              file=sys.stderr)
        if not (clear_stale or confirm("Overwrite them from the record?")):
            raise ArtifactError("not overwriting; record the local files or "
                                "rerun with --clear-stale-artifacts")
    for e in entries:
        if Path(e.path).is_absolute() or ".." in Path(e.path).parts:
            raise ArtifactError(f"by-tree/{tree_sha}: refusing to write outside the "
                                f"task dir: {e.path!r}")
        if e.mode not in (0o644, 0o755):
            raise ArtifactError(f"by-tree/{tree_sha}: mode {e.mode:o} for "
                                f"{e.path!r}; records only hold 644/755")
    # a symlinked dir between task_dir and an entry (even one the globs
    # don't name, e.g. data/ under a data/models/** glob) would route the
    # writes below outside the tree — mkdir and open both follow it
    if linked_parents := sorted({str(p) for e in entries
                                 for p in Path(e.path).parents
                                 if p != Path(".")
                                 and (task_dir / p).is_symlink()}):
        raise ArtifactError(f"{ref}: recorded paths sit behind "
                            f"symlinked dir(s) ({', '.join(linked_parents)}), "
                            "which would route writes outside the task dir; "
                            "remove them and refetch")
    # the write phase starts here; the marker is written only now so a
    # kill during hashing or at the prompt can't flag untouched disk as
    # debris
    marker.write_text("")

    def fetch_entry(e: TreeEntry, tick: Callable[[int], object] | None = None,
                    ) -> bool:
        path = task_dir / e.path
        # a symlink never matches: the record holds regular files only
        if path.is_file() and not path.is_symlink() \
                and cached_sha(path) == e.sha256:
            path.chmod(e.mode)  # content can match while the exec bit drifted
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)  # don't write through a stale symlink or r/o file
        except OSError as err:  # e.g. a local dir where the record holds a file
            raise ArtifactError(
                f"artifact {key!r}: can't replace {path}: {err}") from err
        try:
            ok = store().download_blob(e.sha256, path, tick) == e.sha256
        except zstandard.ZstdError:
            ok = False
        if not ok:
            path.unlink()
            raise _corrupt("by-content/" + e.sha256)
        path.chmod(e.mode)
        st = path.stat()  # our own verified write: fingerprint it for the cache
        cache[e.path] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                         "sha256": e.sha256}
        return True

    # the pre-line covers CI logs; the bar (tty only) covers multi-GB waits
    total = sum(e.size for e in entries)
    print(f"[artifacts] {ref}: fetching {_n_files(len(entries))} "
          f"({total / 1e6:,.1f} MB)")
    with ThreadPoolExecutor(max_workers=TRANSFER_WORKERS) as ex, \
            _SkipBar(total=total, unit="B", unit_scale=True, desc="[fetch]",
                     disable=None) as bar:
        def ticked(e: TreeEntry) -> bool:
            if not fetch_entry(e, bar.update):  # downloads tick per chunk
                bar.tick_skipped(e.size)  # already-matching files tick whole
                return False
            return True
        try:
            fetched = sum(_abort_fast(ex, ex.map(ticked, entries)))
        except BaseException:
            _write_cache(cache_path, cache)  # keep the hashing work for the resume
            raise
    extras = sorted(set(disk) - {e.path for e in entries})
    for extra in extras:  # recoverable or consented to above; nothing is accidentally lost
        try:
            (task_dir / extra).unlink()
        except OSError as err:
            raise ArtifactError(
                f"{ref}: can't delete {extra}: {err}") from err
    # prune dirs the deletions emptied: docker COPY of a leftover empty dir
    # would differ from a fresh clone
    for d in sorted({p for x in extras for p in Path(x).parents if p != Path(".")},
                    key=lambda p: len(p.parts), reverse=True):
        try:
            (task_dir / d).rmdir()
        except OSError:  # not empty: it still holds recorded files
            pass
    _write_cache(cache_path, cache)
    marker.unlink(missing_ok=True)
    print(f"[artifacts] {ref}: {fetched}/{_n_files(len(entries))} "
          "downloaded" + (f", {len(extras)} deleted" if extras else ""))


def fetch_task_artifacts(tasks: list[Task], root_dir: Path,
                         clear_stale: bool = False) -> None:
    """Runner entry point: fetch every declared artifact before any image build.
    Artifact-free runs (and fully up-to-date ones) never touch GCS auth."""
    with _store_boundary():
        for task_dir, key, spec, tree_sha in plan(tasks, root_dir):
            fetch(open_store, task_dir, key, spec, tree_sha, clear_stale)


def _text_lines(data: bytes) -> list[str] | None:
    """Unified-diff input lines, or None for binary (NUL or non-UTF-8) content."""
    try:
        return None if b"\0" in data else data.decode().splitlines(keepends=True)
    except UnicodeDecodeError:
        return None


def _rename_path(src: str, dst: str) -> str:
    """jj's rename spelling: the shared directory prefix, then `{old => new}`."""
    pre = os.path.commonpath([src, dst])
    pre = f"{pre}/" if pre else ""
    return f"{pre}{{{src[len(pre):]} => {dst[len(pre):]}}}"


def _summary_lines(from_entries: dict[str, TreeEntry],
                   to_entries: dict[str, TreeEntry]) -> list[str]:
    """`jj diff -s` form: an A/M/D line per changed path, and R for a delete
    and an add that share content, ordered by the path each starts from."""
    gone: dict[ContentSha, list[str]] = {}  # deleted paths, lowest popped first
    for p in sorted(from_entries.keys() - to_entries.keys(), reverse=True):
        gone.setdefault(from_entries[p].sha256, []).append(p)
    keyed = [(p, f"M {p}") for p in from_entries.keys() & to_entries.keys()
             if from_entries[p] != to_entries[p]]
    for p in sorted(to_entries.keys() - from_entries.keys()):
        if srcs := gone.get(to_entries[p].sha256):
            src = srcs.pop()
            keyed.append((src, f"R {_rename_path(src, p)}"))
        else:
            keyed.append((p, f"A {p}"))
    keyed += [(p, f"D {p}") for ps in gone.values() for p in ps]
    return [line for _, line in sorted(keyed)]


def diff(store: Callable[[], Store], task_dir: Path, key: str,
         spec: ArtifactSpec, from_tree: str | None, to_tree: str | None,
         only: list[str], summary: bool) -> None:
    """Git-style diff between two artifact trees; a None ref means the files
    on disk. Binary (and >MAX_TEXT_DIFF) content prints as a one-line note;
    `summary` prints only the per-file A/M/D/R lines."""
    no_diff = f"[artifacts] {_ref_str(task_dir, key)}: no differences"
    disk_map: dict[str, TreeEntry] = {}
    disk_sha: TreeSha | None = None  # only read when a side's ref is None, which fills these
    if from_tree is None or to_tree is None:
        disk, dir_links = matched_files(task_dir, spec.globs)
        if links := [f for f in disk if (task_dir / f).is_symlink()] + dir_links:
            raise ArtifactError(f"artifact {key!r}: symlink(s) on disk "
                                f"({', '.join(links[:3])}); records never hold "
                                "symlinks, so there's no tree to compare — "
                                "remove them first")
        cache = _stat_cache(task_dir / REFS_DIR / f"{key}.cache")
        entries, _, disk_sha = _hashed(_ref_str(task_dir, key), task_dir, disk,
                                       cache)
        disk_map = {e.path: e for e in entries}

    def side_sha(tree: str | None) -> TreeSha:
        if tree:
            return TreeSha(tree.removeprefix("by-tree/"))
        assert disk_sha is not None  # a None side always fills it above
        return disk_sha

    if side_sha(from_tree) == side_sha(to_tree):
        print(no_diff)
        return

    def side(tree: str | None) -> tuple[dict[str, TreeEntry],
                                       Callable[[TreeEntry], bytes]]:
        if tree is None:
            return disk_map, lambda e: (task_dir / e.path).read_bytes()
        return {e.path: e for e in _fetch_tree(store(), side_sha(tree))}, \
            lambda e: _blob_bytes(store(), e.sha256)

    a, content_a = side(from_tree)
    b, content_b = side(to_tree)
    if only:
        only_spec = pathspec.PathSpec.from_lines("gitwildmatch", only)
        a = {p: e for p, e in a.items() if only_spec.match_file(p)}
        b = {p: e for p, e in b.items() if only_spec.match_file(p)}
    if summary:
        if lines := _summary_lines(a, b):
            print("\n".join(lines))
        else:
            print(no_diff)
        return
    changed = False
    for p in sorted(a.keys() | b.keys()):
        ea, eb = a.get(p), b.get(p)
        if ea == eb:
            continue
        changed = True
        print(f"diff --artifact a/{p} b/{p}")
        a_name = f"a/{p}" if ea else "/dev/null"
        b_name = f"b/{p}" if eb else "/dev/null"
        if ea and eb and ea.mode != eb.mode:
            print(f"old mode {ea.mode:o}\nnew mode {eb.mode:o}")
        if ea and eb and ea.sha256 == eb.sha256:
            continue  # mode-only change
        if max(e.size for e in (ea, eb) if e) > MAX_TEXT_DIFF:
            print(f"Files {a_name} and {b_name} differ (too large to diff)")
            continue
        lines_a = _text_lines(content_a(ea)) if ea else []
        lines_b = _text_lines(content_b(eb)) if eb else []
        if lines_a is None or lines_b is None:
            print(f"Binary files {a_name} and {b_name} differ")
            continue
        for line in difflib.unified_diff(
                lines_a, lines_b, fromfile=a_name, tofile=b_name):
            sys.stdout.write(line if line.endswith("\n")
                             else line + "\n\\ No newline at end of file\n")
    if not changed:
        print(no_diff)


def _require_ignored(task_dir: Path, key: str, files: list[str]) -> None:
    """Artifact files must be gitignored and untracked, or they'd double-record
    into commits and run snapshots (jj auto-tracks anything not ignored)."""
    root = repo_root()
    rels = [str((task_dir / f).resolve().relative_to(root)) for f in files]
    task_rel = str(task_dir.resolve().relative_to(root))
    # jj works where git's index commands don't (non-colocated checkouts,
    # workspaces), and auto-tracks anything unignored on snapshot (which `file
    # list` triggers), so for jj the tracked check subsumes the ignore check.
    jj = (root / ".jj").exists()
    # -z/NUL framing on the git side: quotePath would otherwise C-escape
    # non-ASCII names and they'd never intersect rels (jj prints them raw)
    cmd = (["jj", "file", "list", "--", task_rel] if jj
           else git_cmd(root) + ["ls-files", "-z", "--", task_rel])
    listed = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    # "Refused to snapshot" is jj skipping an unignored over-size file (exit 0!);
    # its hint says to gitignore them, which is exactly this check's job
    if listed.returncode != 0 or "Refused to snapshot" in listed.stderr:
        raise ArtifactError(f"artifact {key!r}: `{' '.join(cmd)}`:\n"
                            f"{listed.stderr.strip()}")
    paths = listed.stdout.splitlines() if jj else listed.stdout.split("\0")
    if tracked := sorted(set(paths) & set(rels)):
        fix = "gitignore and `jj file untrack`" if jj else "`git rm --cached`"
        raise ArtifactError(f"artifact {key!r}: {len(tracked)} matched file(s) are "
                            f"tracked, e.g. {tracked[0]}; {fix} them — artifacts "
                            "live in the record, not source control")
    if jj:
        return
    chk = subprocess.run(git_cmd(root) + ["check-ignore", "-z", "--stdin"],
                         input="\0".join(rels),
                         cwd=root, capture_output=True, text=True)
    if chk.returncode not in (0, 1):  # 1 just means "none ignored"
        raise ArtifactError(f"git check-ignore failed: {chk.stderr.strip()}")
    if unignored := sorted(set(rels) - set(chk.stdout.split("\0"))):
        raise ArtifactError(f"artifact {key!r}: {len(unignored)} matched file(s) "
                            f"aren't gitignored, e.g. {unignored[0]}; add them to "
                            f"{task_dir / '.gitignore'}")


def record(store: Store, task_dir: Path, key: str,
           artifacts: dict[str, ArtifactSpec], build: bool | None = None) -> TreeSha:
    """Build (if a build script is declared), upload, and write the record ref.
    Takes the task's whole artifact dict so overlapping declarations are
    caught here, where the post-build file set is known.
    build=`None`: prompt before rebuilding over unrecorded edits.
    build=`True`/`False`: rebuild unconditionally / skip the build script."""
    spec = artifacts[key]
    marker = task_dir / REFS_DIR / f"{key}.fetching"
    _lock = _claim_writer(marker, _ref_str(task_dir, key))  # released when record exits
    marker.write_text("record")  # a leftover prompts at the next fetch, never resumes
    # an unnoticed rebuild can silently destroy hand edits: when disk doesn't
    # match the pinned record, ask before running the build over it
    if spec.build and build is None and (task_dir / REFS_DIR / key).exists():
        disk, _ = matched_files(task_dir, spec.globs)
        linked = any((task_dir / f).is_symlink() for f in disk)
        disk_sha = ""
        if disk and not linked:
            _, _, disk_sha = _hashed(
                _ref_str(task_dir, key), task_dir, disk,
                _stat_cache(task_dir / REFS_DIR / f"{key}.cache"))
        if disk and (linked or disk_sha != _recorded_sha(task_dir, key)):
            print(f"WARNING: {_ref_str(task_dir, key)}: the files on disk don't "
                  f"match the pinned record; running {spec.build!r} may "
                  "overwrite hand edits" + ("" if linked else
                      f"; inspect with:\n  {_CLI} diff {_ref_str(task_dir, key)}"),
                  file=sys.stderr)
            if not confirm(f"Run {spec.build!r} anyway?"):
                raise ArtifactError(
                    f"artifact {key!r}: not rebuilding over unrecorded edits; "
                    "rerun with --force-build (rebuild, discarding them) or "
                    "--no-build (record the files as they are)")
    if spec.build and build is not False:
        print(f"[artifacts] {_ref_str(task_dir, key)}: running {spec.build!r}")
        try:
            subprocess.run(["./" + spec.build], cwd=task_dir, check=True)
        except OSError as err:  # missing, not executable, or no shebang
            raise ArtifactError(f"artifact {key!r}: build {spec.build!r} must be "
                                f"an executable script in {task_dir}: {err}")
        except subprocess.CalledProcessError as err:
            raise ArtifactError(f"artifact {key!r}: {spec.build} exited "
                                f"{err.returncode}")
    files, dir_links = matched_files(task_dir, spec.globs)
    links = [f for f in files if (task_dir / f).is_symlink()] + dir_links
    if links:
        raise ArtifactError(f"artifact {key!r}: can't record symlinks (docker COPY "
                            f"keeps the link, not its content): {', '.join(links)}")
    if not files:
        raise ArtifactError(f"artifact {key!r}: nothing matches globs {spec.globs} "
                            f"in {task_dir}")
    for other, ospec in artifacts.items():
        if other != key and (both := set(files)
                             & set(matched_files(task_dir, ospec.globs)[0])):
            raise ArtifactError(
                f"artifact {key!r}: {len(both)} file(s) also match artifact "
                f"{other!r}'s globs, e.g. {sorted(both)[0]}; overlapping "
                "artifacts would fight over the same paths on fetch")
    _require_ignored(task_dir, key, files)
    cache: dict[str, Any] = {}  # start empty: record trusts no prior hashes
    ref = _ref_str(task_dir, key)
    entries, data, tree_sha = _hashed(ref, task_dir, files, cache)
    for e in entries:  # pin disk modes to the normalized recorded ones
        (task_dir / e.path).chmod(e.mode)
    total = sum(e.size for e in entries)
    print(f"[artifacts] {ref}: uploading {_n_files(len(entries))} "
          f"({total / 1e6:,.1f} MB; existing blobs skip)")

    def upload(e: TreeEntry, tick: Callable[[int], object]) -> bool:
        if store.exists("by-content/" + e.sha256):
            return False
        store.upload_blob(e.sha256, task_dir / e.path, tick)
        return True

    with ThreadPoolExecutor(max_workers=TRANSFER_WORKERS) as ex, \
            _SkipBar(total=total, unit="B", unit_scale=True, desc="[upload]",
                     disable=None) as bar:
        def ticked(e: TreeEntry) -> None:
            if not upload(e, bar.update):  # uploads tick per chunk
                bar.tick_skipped(e.size)  # skipped blobs tick whole
        _abort_fast(ex, ex.map(ticked, entries))
    store.upload_bytes(f"by-tree/{tree_sha}", data)
    ref_path = task_dir / REFS_DIR / key
    ref_path.parent.mkdir(exist_ok=True)
    ref_path.write_text(f"by-tree/{tree_sha}\n")
    marker.unlink()  # disk now is this record: no interrupted writes to flag
    _write_cache(task_dir / REFS_DIR / f"{key}.cache", cache)
    print(f"[artifacts] recorded {_ref_str(task_dir, key)}: "
          f"{_n_files(len(files))}, "
          f"by-tree/{tree_sha}")
    print(f"[artifacts] commit {ref_path} to bind the task to this record")
    return tree_sha


def _resolve(ref: str, path: str | None
             ) -> tuple[Path, list[str], dict[str, ArtifactSpec]]:
    """Ref grammar -> (task dir, selected keys, the task's declared artifacts)."""
    # '.#<sel>', './#<sel>', or bare '.' pin the task to cwd/--path: same as
    # a leading '#', which shells would eat as a comment
    if ref == "." or ref.startswith((".#", "./#")):
        ref = ref[ref.index("#"):] if "#" in ref else ""

    # Task names may contain "/", so "#" is what distinguishes the ref forms;
    # an explicit --path pins a "#"-less ref to the bare-'<sel>' reading.
    # '..' is a selector, not a task: 'tasks/..' is a real dir and would match.
    if "#" in ref:
        task_id, _, sel = ref.partition("#")
        if task_id:
            task_dir = repo_root() / "tasks" / task_id
            if not task_dir.is_dir():
                raise SystemExit(f"no such task: {task_id}{task_hint(task_id)}")
        else:  # bare '#<sel>': like '<sel>', the task comes from cwd/--path
            task_dir = Path(path or ".").resolve()
    elif ref and ref != ".." and path is None and (repo_root() / "tasks" / ref).is_dir():
        sel, task_dir = "", repo_root() / "tasks" / ref
    else:
        sel, task_dir = ref, Path(path or ".").resolve()
        if not task_dir.is_dir():
            raise SystemExit(f"--path {path}: no such directory")

    # every verb wants a task dir: refs suggest '<task-id>#<sel>' commands,
    # and record needs the repo's gitignore machinery
    if not task_dir.resolve().is_relative_to(repo_root() / "tasks"):
        raise SystemExit(f"{task_dir} isn't a task dir under "
                         f"{repo_root() / 'tasks'}; run from one or name it: "
                         "'<task-id>#<sel>'"
                         + (task_hint(ref) if ref else ""))

    # a "#"-less ref that names no task falls through to the bare-selector
    # reading; if that fails too (e.g. a typo'd task), name both interpretations
    typo = (f"no task {sel!r} under {repo_root() / 'tasks'}{task_hint(sel)}, and "
            if sel and path is None and "#" not in ref else "")

    try:
        artifacts = load_task_config(task_dir).artifacts
    except (ValidationError, FileExistsError) as e:  # CONFIG.json problems
        raise SystemExit(f"bad CONFIG in {task_dir}: {e}") from e

    if not artifacts:
        config = "CONFIG.jsonc" if (task_dir / "CONFIG.jsonc").exists() else "CONFIG.json"
        raise SystemExit(
            f"{typo}{task_dir} declares no artifacts"
            + (f"; declare one in {task_dir / config}, e.g.\n"
               '''  {"artifacts": {"data": {"globs": ["data/**"], "build": "build_data.sh"}}}'''
               if (task_dir / "TASK.txt").exists() or (task_dir / "TASK.py").exists()
               else "; run from a task dir, or name one: '<task-id>#<sel>'"))

    if sel == "..":  # the explicit ask for every declared artifact
        keys = sorted(artifacts)
    elif not sel:  # a task with one artifact doesn't need the key spelled out
        if len(artifacts) != 1:
            refs = [f"'{_ref_str(task_dir, k)}'" for k in sorted(artifacts)]
            raise SystemExit(  # full refs: a bare '#<sel>' is a shell comment
                f"{task_dir} declares {len(artifacts)} artifacts; pick one: "
                + (" or ".join(refs) if len(refs) == 2 else f"e.g. {refs[0]}")
                + f", or all with '{_ref_str(task_dir, '..')}'")
        keys = sorted(artifacts)
    else:
        keys = sel.split(",")
        for k in keys:
            if k not in artifacts:
                raise SystemExit(f"{typo}{task_dir} declares no artifact {k!r} "
                                 f"(declared: {sorted(artifacts)})")
    return task_dir, keys, artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,  # keep the store-layout table's line breaks
        epilog=f"store: {ARTIFACT_STORE_URL} (override via RL_ENV_ARTIFACT_STORE)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ref_args = argparse.ArgumentParser(add_help=False)  # shared by every verb
    ref_args.add_argument("ref", nargs="?", default="",
                          help="'<task-id>#<sel>', or '#<sel>'/'.#<sel>' for "
                          "the --path/cwd task; <sel> is a key, 'k1,k2', or "
                          "'..' (all); bare '<task-id>' means its only "
                          "artifact, bare '<sel>' reads from --path/cwd")
    ref_args.add_argument("-p", "--path",
                          help="task dir for bare-'<sel>' refs (default: cwd)")
    sub = parser.add_subparsers(dest="command", required=True)
    rec_help = ("build (if declared), upload, and write .ARTIFACT_REFS/<key> "
                "(commit it to pin the task)")
    rec_p = sub.add_parser("record", parents=[ref_args],
                           help=rec_help, description=rec_help)
    rec_group = rec_p.add_mutually_exclusive_group()
    rec_group.add_argument("--force-build", action="store_true",
                           help="run the build script even over unrecorded edits")
    rec_group.add_argument("--no-build", action="store_true",
                           help="skip the build script; record files as they are")
    fetch_p = sub.add_parser("fetch", parents=[ref_args],
                             help="make local files exactly match the record",
                             description="make local files exactly match the "
                             "record, downloading mismatches and deleting "
                             "files the record doesn't contain")
    fetch_p.add_argument("--clear-stale-artifacts", action="store_true",
                         help="overwrite local files that don't match any "
                         "recorded tree without prompting")
    diff_help = "git-style diff between artifact trees or disk"
    diff_p = sub.add_parser("diff", parents=[ref_args],
                            help=diff_help, description=diff_help)
    diff_p.add_argument("--from", dest="from_tree", metavar="BY_TREE",
                        help="diff from this by-tree/<sha> (default: the "
                        "pinned record, or disk when --to is given)")
    diff_p.add_argument("--to", dest="to_tree", metavar="BY_TREE",
                        help="diff to this by-tree/<sha> (default: the files "
                        "on disk)")
    diff_p.add_argument("-f", "--file", action="append", default=[],
                        metavar="GLOB",
                        help="limit diff to matching paths (repeatable)")
    diff_p.add_argument("-s", "--summary", action="store_true",
                        help="list each changed path as A/M/D/R, not its content")
    list_help = "list an artifact's recorded trees, newest first (by first pin)"
    list_p = sub.add_parser("list", parents=[ref_args],
                            help=list_help, description=list_help)
    list_p.add_argument("--commits", action="store_true",
                        help="also show the commit each record was first pinned in")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    task_dir, keys, artifacts = _resolve(args.ref, args.path)

    if args.command == "diff" and len(keys) > 1 and (args.from_tree or args.to_tree):
        raise SystemExit("--from/--to name a single record; use a one-key ref")

    with _store_boundary():
        for key in keys:
            if args.command == "record":
                build = (True if args.force_build else
                         False if args.no_build else None)
                record(open_store(), task_dir, key, artifacts, build)
            elif args.command == "list":
                if len(keys) > 1:
                    print(f"{_ref_str(task_dir, key)}:")
                list_records(task_dir, key, args.commits)
            elif args.command == "diff":
                from_tree = args.from_tree or (
                    None if args.to_tree else _recorded_sha(task_dir, key))
                diff(open_store, task_dir, key, artifacts[key],
                     from_tree, args.to_tree or None, args.file, args.summary)
            else:
                fetch(open_store, task_dir, key, artifacts[key],
                      _recorded_sha(task_dir, key), args.clear_stale_artifacts)


if __name__ == "__main__":
    main()
