"""Nonblocking process lock for a report's complete read/update lifetime."""

from contextlib import contextmanager
import errno
import os
from pathlib import Path


class ReportBusy(RuntimeError):
    pass


@contextmanager
def report_lock(path):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock file: unlinking it lets a new writer lock a different inode.
    with target.with_name(target.name + ".lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise ReportBusy("another evaluation owns this report") from error
            raise
        try:
            yield
        finally:
            release()
