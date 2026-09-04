"""Cheap system-memory sampling for pipeline logging.

One snapshot returns the Task-Manager headline number (system RAM percent
and used/total GB) plus a cheap breakdown of what this process tree is
using. Designed to be called every 1-2 seconds: the cost is one
``virtual_memory()`` call plus one recursive children walk (~8 ms on a
typical Windows box), dominated by the child enumeration.

:class:`MemoryMonitor` wraps that in a background thread with independent
tick, log and warn rates, and is what the workers and the gui options drive.
From a notebook::

    from mbo_utilities import start_memory_monitor, stop_memory_monitor
    m = start_memory_monitor(tick_s=0.5, log_s=30, warn_pct=90)
    ...
    m.peak_gb; stop_memory_monitor()
"""

from __future__ import annotations

from typing import Any

_GB = 1024 ** 3


def mem_snapshot(proc: Any | None = None) -> dict[str, Any]:
    """System RAM headline plus this process tree's usage.

    Keys:
        sys_pct   system memory in use, matches Task Manager's percentage
        used_gb   system memory in use (total - available)
        total_gb  total physical memory
        proc_gb   summed RSS of this worker and its children
        nproc     number of processes in that tree
        top       (pid, name, gb) of the largest single process, or None
    """
    import psutil

    vm = psutil.virtual_memory()
    snap: dict[str, Any] = {
        "sys_pct": vm.percent,
        "used_gb": (vm.total - vm.available) / _GB,
        "total_gb": vm.total / _GB,
    }

    p = proc or psutil.Process()
    try:
        procs = [p] + p.children(recursive=True)
    except psutil.Error:
        procs = [p]

    rss = 0
    top = None
    top_rss = -1
    for c in procs:
        try:
            m = c.memory_info().rss
        except psutil.Error:
            continue
        rss += m
        if m > top_rss:
            top_rss, top = m, c

    snap["proc_gb"] = rss / _GB
    snap["nproc"] = len(procs)
    if top is not None:
        try:
            name = top.name()
        except psutil.Error:
            name = "?"
        snap["top"] = (top.pid, name, top_rss / _GB)
    else:
        snap["top"] = None
    return snap


def format_mem_line(snap: dict[str, Any]) -> str:
    """One-line human form of a snapshot for the task log."""
    line = (
        f"mem {snap['sys_pct']:.1f}% "
        f"{snap['used_gb']:.1f}/{snap['total_gb']:.1f} GB"
    )
    if "proc_gb" in snap:
        line += f" | pipeline {snap['proc_gb']:.2f} GB/{snap.get('nproc', 0)} proc"
        top = snap.get("top")
        if top:
            line += f", top {top[0]} {top[1]} {top[2]:.2f} GB"
    return line


class MemoryMonitor:
    """Background memory sampler with independent tick, log and warn rates.

    Sampling and logging are deliberately separate: a fast tick is what makes
    a ``warn_pct`` check useful (it catches a spike before the OOM killer
    does) but nobody wants that rate in the task log, so ``log_s`` throttles
    the log lines independently. Every tick still lands in the csv when
    ``csv_path`` is given — that's the record you read after a crash.

    Parameters
    ----------
    tick_s : float
        Seconds between snapshots. Clamped to >= 0.05.
    log_s : float or None
        Seconds between log lines. ``None`` logs every tick; 0 disables
        periodic log lines entirely (warnings and peaks still get through).
    warn_pct : float
        System-memory percent at or above which a WARNING is emitted. 0
        disables the check.
    warn_cooldown_s : float
        Minimum seconds between repeated warnings, so a sustained high-memory
        stretch does not flood the log at the tick rate.
    csv_path : str or Path or None
        Append every tick as a row here. Header is written for a new file.
    logger : logging.Logger or None
        Where log lines go. Defaults to the package ``mbo`` logger, which
        already prints to the console.
    log_peaks : bool
        Log at INFO whenever process-tree memory sets a new peak (over
        ``peak_delta_gb``), regardless of ``log_s``.
    on_sample, on_warn : callable or None
        Called with the snapshot dict each tick / each warning. Exceptions
        raised by a callback are swallowed so the monitor thread survives.
    """

    def __init__(
        self,
        tick_s: float = 2.0,
        log_s: float | None = None,
        warn_pct: float = 0.0,
        warn_cooldown_s: float = 30.0,
        csv_path: Any = None,
        logger: Any = None,
        proc: Any = None,
        log_peaks: bool = True,
        peak_delta_gb: float = 0.1,
        on_sample: Any = None,
        on_warn: Any = None,
    ) -> None:
        self.tick_s = max(0.05, float(tick_s))
        self.log_s = None if log_s is None else max(0.0, float(log_s))
        self.warn_pct = float(warn_pct or 0.0)
        self.warn_cooldown_s = max(0.0, float(warn_cooldown_s))
        self.csv_path = csv_path
        if logger is None:
            # the "mbo" root is the logger that carries a console handler
            # (log.get() sub-loggers don't propagate), so a monitor started
            # from a notebook prints without any logging setup
            import logging

            from mbo_utilities import log as _log  # noqa: F401 - installs the handler

            logger = logging.getLogger("mbo")
        self.logger = logger
        self.proc = proc
        self.log_peaks = bool(log_peaks)
        self.peak_delta_gb = float(peak_delta_gb)
        self.on_sample = on_sample
        self.on_warn = on_warn

        self.last: dict[str, Any] | None = None
        self.peak_gb: float = 0.0
        self._thread = None
        self._stop = None

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        log_s = "every tick" if self.log_s is None else f"{self.log_s:g}s"
        warn = f", warn {self.warn_pct:g}%" if self.warn_pct else ""
        return (
            f"MemoryMonitor({state}, tick {self.tick_s:g}s, log {log_s}{warn})"
        )

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "MemoryMonitor":
        """Start the daemon sampling thread. No-op when already running."""
        import threading

        if self.running:
            return self
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="mbo-mem-monitor", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to finish the current tick and exit."""
        import threading

        if self._stop is not None:
            self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=timeout)
        self._thread = None

    def __enter__(self) -> "MemoryMonitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- sampling ----------------------------------------------------------

    def sample(self) -> dict[str, Any]:
        """Take one snapshot now, updating ``last`` and ``peak_gb``."""
        s = mem_snapshot(self.proc)
        self.last = s
        self.peak_gb = max(self.peak_gb, s.get("proc_gb", 0.0))
        return s

    def _run(self) -> None:
        import time

        f = self._open_csv()
        start = time.time()
        last_log = 0.0
        last_warn = 0.0
        first = True
        peak = 0.0
        try:
            while not self._stop.wait(self.tick_s):
                try:
                    s = self.sample()
                except Exception as e:
                    # a snapshot that fails once fails every tick (no psutil,
                    # denied process access) — say so and stop instead of
                    # spinning silently.
                    self.logger.warning("memory monitor stopping: %s", e)
                    return
                now = time.time()

                if f is not None:
                    self._write_row(f, s, now - start)
                _call(self.on_sample, s)

                if self.warn_pct and s["sys_pct"] >= self.warn_pct:
                    if now - last_warn >= self.warn_cooldown_s:
                        last_warn = now
                        self.logger.warning(
                            "memory high (>= %.0f%%): %s",
                            self.warn_pct,
                            format_mem_line(s),
                        )
                        _call(self.on_warn, s)

                pg = s.get("proc_gb", 0.0)
                new_peak = self.log_peaks and (first or pg > peak + self.peak_delta_gb)
                due = self.log_s is None or (
                    self.log_s > 0 and now - last_log >= self.log_s
                )
                if first or new_peak or due:
                    peak = max(peak, pg)
                    first = False
                    last_log = now
                    self.logger.info(format_mem_line(s))
                else:
                    self.logger.debug(format_mem_line(s))
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass

    def _open_csv(self):
        if not self.csv_path:
            return None
        try:
            from pathlib import Path

            p = Path(self.csv_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            f = open(p, "a", encoding="utf-8")
            if f.tell() == 0:
                f.write(
                    "t_iso,elapsed_s,sys_pct,used_gb,total_gb,"
                    "proc_gb,nproc,top_pid,top_name,top_gb\n"
                )
            return f
        except Exception:
            return None

    @staticmethod
    def _write_row(f, s: dict[str, Any], elapsed: float) -> None:
        """One csv row, flushed — the last sample must survive an OOM kill."""
        import time

        top = s.get("top")
        tp, tn, tgb = (
            (top[0], str(top[1]).replace(",", " "), top[2]) if top else ("", "", 0.0)
        )
        try:
            f.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{elapsed:.1f},"
                f"{s['sys_pct']:.1f},{s['used_gb']:.3f},{s['total_gb']:.3f},"
                f"{s.get('proc_gb', 0.0):.3f},{s.get('nproc', 0)},{tp},{tn},{tgb:.3f}\n"
            )
            f.flush()
        except Exception:
            pass


def _call(fn: Any, s: dict[str, Any]) -> None:
    """Run a user callback without letting it kill the monitor thread."""
    if fn is None:
        return
    try:
        fn(s)
    except Exception:
        pass


_ACTIVE: "MemoryMonitor | None" = None


def start_memory_monitor(**kwargs: Any) -> MemoryMonitor:
    """Start (or restart) the process-wide monitor. Handy from IPython::

        from mbo_utilities import start_memory_monitor
        m = start_memory_monitor(tick_s=0.5, log_s=10, warn_pct=90)
        ...
        m.last, m.peak_gb

    Takes the same keywords as :class:`MemoryMonitor`. Any monitor started by
    a previous call is stopped first, so this is safe to re-run in a cell.
    """
    global _ACTIVE
    stop_memory_monitor()
    _ACTIVE = MemoryMonitor(**kwargs).start()
    return _ACTIVE


def stop_memory_monitor() -> None:
    """Stop the process-wide monitor started by ``start_memory_monitor``."""
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.stop()
        _ACTIVE = None


def memory_monitor() -> "MemoryMonitor | None":
    """The process-wide monitor, or None when none is running."""
    return _ACTIVE
