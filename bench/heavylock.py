"""Adapter to the anniversary project's shared heavy lock (``media/.heavy.lock``).

The lock protocol lives in the film project's ``tools/heavy.py``. We *reuse* that module by path rather than
copying its protocol, so benchmark jobs and film renders exclude each other. If the module cannot be found
(public synthetic use, another machine) a local lock under this repo's ignored ``out/`` is used instead and
the backend is recorded in the manifest.

Usage::

    with HeavyLock("bench baseline t060", chunk=0) as lock:
        ...                      # lock.on_step(i) may release/re-acquire between chunks

``acquire``/``release`` are explicit so an ``on_step`` gate can release periodically. A stale lock is never
broken here; the film module's own policy (dead holder + age) applies.
"""
from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path

DEFAULT_FILM_TOOLS = Path.home() / "Documents" / "annniversary-video" / "tools"
FILM_TOOLS_ENV = "MLXSTAB_FILM_TOOLS"     # only this one path override is read; no other env is exported
LOCAL_LOCK = Path(__file__).resolve().parent.parent / "out" / ".heavy.lock"


def film_tools_dir():
    return Path(os.environ.get(FILM_TOOLS_ENV, DEFAULT_FILM_TOOLS))


def _load_film_heavy():
    tools = film_tools_dir()
    path = tools / "heavy.py"
    if not path.exists():
        return None, tools
    spec = importlib.util.spec_from_file_location("_mlxstab_film_heavy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, tools


@contextmanager
def _local_heavy(lock_path: Path, what="", poll=10):
    """Same protocol as the film lock, rooted at this repo's ignored out/ directory."""
    import subprocess
    import time

    def holder():
        try:
            return (lock_path / "pid").read_text().split("\n", 1)
        except Exception:
            return None

    while True:
        try:
            lock_path.mkdir(parents=True, exist_ok=False)
            (lock_path / "pid").write_text(f"{os.getpid()}\n{what}")
            break
        except FileExistsError:
            try:
                os.kill(int(holder()[0]), 0)
                alive = True
            except Exception:
                alive = False
            if not alive and time.time() - lock_path.stat().st_mtime > 1200:
                subprocess.run(["rm", "-rf", str(lock_path)])
                continue
            time.sleep(poll)
    try:
        yield
    finally:
        if holder() and holder()[0] == str(os.getpid()):
            subprocess.run(["rm", "-rf", str(lock_path)])


class HeavyLock:
    """Reusable, explicitly acquirable lock. ``chunk>0`` releases/re-acquires at on_step multiples.

    ``pause_s`` is the gap left for waiting jobs between chunks (the shared-project etiquette is ~30 frames
    per chunk with a ~12 s gap). Acquisition wait and pause time are accounted separately from processing.
    """

    def __init__(self, what="", enabled=True, chunk=0, poll=10, pause_s=0.0, sleep=None):
        import time as _time
        self.what = what
        self.enabled = enabled
        self.chunk = int(chunk or 0)
        self.poll = poll
        self.pause_s = float(pause_s or 0.0)
        self.sleep = sleep or _time.sleep
        self.backend = "disabled"
        self.path = None
        self._cm = None
        self.acquire_wait_s = 0.0
        self.pause_total_s = 0.0
        self.releases = 0
        if enabled:
            mod, tools = _load_film_heavy()
            if mod is not None:
                self.backend = "film"
                self.path = str(mod.LOCK)
                self._factory = lambda: mod.heavy(self.what, self.poll)
            else:
                self.backend = "local"
                self.path = str(LOCAL_LOCK)
                self._factory = lambda: _local_heavy(LOCAL_LOCK, self.what, self.poll)

    # -- explicit lifecycle -----------------------------------------------------------------
    def acquire(self):
        if not self.enabled or self._cm is not None:
            return
        import time as _time
        t0 = _time.perf_counter()
        self._cm = self._factory()
        self._cm.__enter__()
        self.acquire_wait_s += _time.perf_counter() - t0

    def release(self):
        if self._cm is None:
            return
        cm, self._cm = self._cm, None
        cm.__exit__(None, None, None)
        self.releases += 1

    def on_step(self, step):
        """Gate for ``stabilize``/``metrics.measure`` ``on_step``: periodically yield between chunks."""
        if self.chunk and step and step % self.chunk == 0:
            self.release()
            import time as _time
            t0 = _time.perf_counter()
            if self.pause_s > 0:
                self.sleep(self.pause_s)
            self.pause_total_s += _time.perf_counter() - t0
            self.acquire()

    def status(self):
        return {"backend": self.backend, "path": self.path, "chunk": self.chunk, "pause_s": self.pause_s,
                "held": self._cm is not None, "releases": self.releases,
                "acquire_wait_s": self.acquire_wait_s, "pause_total_s": self.pause_total_s}

    # -- context manager --------------------------------------------------------------------
    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False
