"""Safe reader for agent-written files: a grader runs as root over the agent-controlled
/workdir, where a bare open() could follow a planted symlink or block on a FIFO."""

import errno
import os
import stat

# Infra faults with no agent-reachable trigger are discarded rather than scored 0.
# Everything else scores 0, even things which could also be infra errors.
_INFRA_ERRNOS = frozenset(
    {
        errno.EIO,  # storage fault
        errno.ESTALE,  # handle invalidated under us
        errno.EMFILE,  # the grader leaked its own fds
        errno.EACCES,  # an LSM denial; root bypasses file modes
    }
)

# Judgment calls--can be either infra errors or agent attacks. Scored 0 for now.
_CONTESTED_ERRNOS = frozenset({errno.ENOMEM, errno.ENFILE})

# Reachable only by planting something at the path, so seeing one means the agent tried.
_SABOTAGE_ERRNOS = frozenset(
    {
        errno.ELOOP,  # a symlink, refused by O_NOFOLLOW
        errno.ENXIO,  # a socket bound at the path
        errno.EAGAIN,  # a write lease held by a surviving agent process
    }
)

# The agent left nothing readable at the path: the ordinary way to score 0, not an attack.
_MISTAKE_ERRNOS = frozenset(
    {
        errno.ENOENT,  # nothing there
        errno.ENOTDIR,  # a component is a file, e.g. results/ written as a file
    }
)


def read_submission_bytes(path: str, max_bytes: int) -> bytes | None:
    """Bytes of the file at `path`, or None if it is missing or not a regular file within
    `max_bytes` (rejects symlinks, FIFOs, sockets, devices, directories, and oversize files).

    `path` is agent-controlled, so failing to open it scores 0 unless the errno says the
    fault was the infrastructure's."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno in _INFRA_ERRNOS:
            raise
        if exc.errno in _SABOTAGE_ERRNOS:
            cause = "likely agent sabotage"
        elif exc.errno in _CONTESTED_ERRNOS:
            cause = "exhaustion, scored as agent-caused"
        elif exc.errno in _MISTAKE_ERRNOS:
            cause = "no submission at the path"
        else:
            cause = "UNEXPECTED, cause unknown"
        print(f"safe_read: cannot open {path}: {exc} ({cause})")
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            print(f"safe_read: {path} is not a regular file")
            return None
        if info.st_size > max_bytes:
            print(f"safe_read: {path} is {info.st_size} > max_bytes={max_bytes}")
            return None
        with open(fd, "rb", closefd=False) as handle:
            return handle.read(max_bytes)
    finally:
        os.close(fd)


def read_submission(path: str, max_bytes: int) -> str | None:
    """Text of the file at `path`, or None if it fails `read_submission_bytes`."""
    data = read_submission_bytes(path, max_bytes)
    return None if data is None else data.decode("utf-8", "replace")
