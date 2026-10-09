"""Best-effort cross-platform CLI run lock, released by the OS on process exit.

Uses POSIX flock or Windows msvcrt byte-range lock. Only CLI commands are
serialized; directly calling the Python API from multiple processes is unsafe.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


class RunBusyError(RuntimeError):
    pass


@contextmanager
def locked_run(run: Path):
    if not run.is_dir():
        raise RunBusyError(f'Run directory does not exist: {run}')
    path = run / '.lerp.lock'
    if path.is_symlink():
        raise RunBusyError(f'Unsafe symlinked run lock: {path}')
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'\x00')
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RunBusyError('Another Lerp process is using this run') from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RunBusyError('Another Lerp process is using this run') from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
