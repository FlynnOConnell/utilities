"""
worker entry point for background subprocess tasks.

this module is invoked via:
    python -m mbo_utilities.gui._worker <task_type> <args_json>

it runs independently of the gui and can survive gui closure.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from mbo_utilities.gui.tasks import TASKS

# max minutes with no progress change before worker self-terminates
MAX_STALL_MINUTES = 120

# Windows Job Object handle, kept alive for the worker's lifetime so the
# job is not closed (and its members killed) prematurely. See _contain_in_job.
_JOB_HANDLE = None


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Setup logging for worker process.

    routes through mbo_utilities.log so the package's single non-propagating
    'mbo' handler is the only stdout sink — prevents the duplicate-line
    pattern caused by adding a second handler on a propagating child logger.
    """
    from mbo_utilities import log

    logger = log.get("worker")
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # apply the unified format to the package handler too
    for h in logger.handlers + logging.getLogger("mbo").handlers:
        h.setFormatter(fmt)

    if log_file:
        try:
            from logging.handlers import RotatingFileHandler

            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # cap a single task's log so a chatty run can't balloon to 100s of MB
            fh = RotatingFileHandler(
                log_path, mode="a", maxBytes=5_000_000, backupCount=2, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Failed to setup file logging: {e}", file=sys.stderr)

    return logger


def _update_status(pid: int, status: str, message: str | None = None, details: str | dict | None = None, uuid: str | None = None):
    """Ensure process status is reported to sidecar file."""
    try:
        # Match location used by TaskMonitor and ProcessManager
        from mbo_utilities.preferences import get_mbo_dirs
        log_dir = get_mbo_dirs()["logs"]
        if uuid:
            sidecar = log_dir / f"progress_{uuid}.json"
        else:
            sidecar = log_dir / f"progress_{pid}.json"

        current_data = {}
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    current_data = json.load(f)
            except Exception:
                pass

        # If already set to final state by task, respect it
        # (unless we are reporting error which overrides success-in-progress)
        if status == "completed" and current_data.get("status") == "completed":
            return

        data = current_data.copy()
        data["pid"] = pid
        data["uuid"] = uuid
        data["timestamp"] = time.time()
        data["status"] = status

        if message:
            data["message"] = message
        if details:
            data["details"] = details

        if status == "completed":
            data["progress"] = 1.0

        # Write atomically to avoid race conditions with readers
        tmp_file = sidecar.with_suffix(".tmp")
        with open(tmp_file, "w") as f:
            json.dump(data, f)
        _atomic_replace(tmp_file, sidecar)

    except Exception as e:
        print(f"Failed to update status sidecar: {e}", file=sys.stderr)


def _atomic_replace(src: Path, dst: Path, attempts: int = 10, delay: float = 0.05):
    """Path.replace with retry for windows sharing-violation races.

    the gui opens the sidecar for read with default share flags, which on
    windows denies rename until the handle closes. a short retry loop
    clears the contention without blocking the worker meaningfully.
    """
    last_err: Exception | None = None
    for _ in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
    # final attempt raises so caller can log/surface the real failure
    if last_err is not None:
        raise last_err


def _start_watchdog(uuid: str | None, logger: logging.Logger, log_file: str | None = None):
    """daemon thread that kills this process if progress stalls."""
    def _watchdog():
        from mbo_utilities.preferences import get_mbo_dirs
        log_dir = get_mbo_dirs()["logs"]
        last_progress = 0.0
        last_log_mtime = 0.0
        last_change = time.time()
        stall_seconds = MAX_STALL_MINUTES * 60

        while True:
            time.sleep(60)
            try:
                if uuid:
                    sidecar = log_dir / f"progress_{uuid}.json"
                else:
                    sidecar = log_dir / f"progress_{os.getpid()}.json"

                if sidecar.exists():
                    with open(sidecar) as f:
                        data = json.load(f)
                    progress = data.get("progress", 0.0)
                    status = data.get("status", "running")

                    # done, no need to watch
                    if status in ("completed", "error"):
                        return

                    if progress != last_progress:
                        last_progress = progress
                        last_change = time.time()
            except Exception:
                pass

            # Liveness fallback: the worker's stdout/stderr and logger are
            # all redirected to log_file. Long pipeline tasks (correct_stack,
            # multi_fuse) run a single blocking call that emits per-step log
            # lines but never moves the progress number. An advancing log
            # mtime means the task is still working, so only a process that
            # is both progress-frozen and log-silent for the full window is
            # treated as hung.
            try:
                if log_file:
                    m = os.path.getmtime(log_file)
                    if m > last_log_mtime:
                        last_log_mtime = m
                        last_change = time.time()
            except OSError:
                pass

            if time.time() - last_change > stall_seconds:
                msg = f"watchdog: no progress for {MAX_STALL_MINUTES} min, terminating"
                logger.error(msg)
                _update_status(os.getpid(), "error", message=msg, uuid=uuid)
                os._exit(1)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()


def _resolve_mem_settings() -> tuple[float, float, float]:
    """(tick_s, log_s, warn_pct) for this worker's memory monitor.

    Tick 0 means off. Env vars win so an HPC submission can dial the rates
    without touching the gui preferences:

      MBO_MEM_LOG_INTERVAL  sample (tick) interval, seconds
      MBO_MEM_LOG_EVERY     seconds between log lines (0 = every tick)
      MBO_MEM_WARN_PCT      warn at/above this system-memory percent (0 = off)
    """
    def _env(name: str) -> float | None:
        raw = os.environ.get(name)
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    tick = _env("MBO_MEM_LOG_INTERVAL")
    log_s = _env("MBO_MEM_LOG_EVERY")
    warn = _env("MBO_MEM_WARN_PCT")

    if tick is None or log_s is None or warn is None:
        try:
            from mbo_utilities.preferences import (
                get_mem_monitor,
                get_mem_monitor_interval,
                get_mem_log_interval,
                get_mem_warn_pct,
            )
            if tick is None:
                tick = float(get_mem_monitor_interval()) if get_mem_monitor() else 0.0
            if log_s is None:
                log_s = float(get_mem_log_interval())
            if warn is None:
                warn = float(get_mem_warn_pct())
        except Exception:
            pass

    return (tick or 0.0), (log_s if log_s is not None else 0.0), (warn or 0.0)


def _start_mem_monitor(uuid: str | None, logger: logging.Logger):
    """Sample system + process-tree memory to logs/mem_<uuid>.csv.

    Every tick is a csv row (flushed, so the last sample survives a hard OOM
    kill); log lines are throttled to their own rate, with new pipeline-memory
    peaks and high-memory warnings still getting through immediately.
    """
    tick, log_s, warn_pct = _resolve_mem_settings()
    if tick <= 0:
        return None

    from mbo_utilities._sysmem import MemoryMonitor
    from mbo_utilities.preferences import get_mbo_dirs

    log_dir = get_mbo_dirs()["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    return MemoryMonitor(
        tick_s=tick,
        log_s=log_s,
        warn_pct=warn_pct,
        csv_path=log_dir / f"mem_{uuid or os.getpid()}.csv",
        logger=logger,
    ).start()


def _contain_in_job():
    """Windows: assign this worker to a Job Object that terminates all
    members when the job handle closes.

    Processes a task spawns (ProcessPoolExecutor / multiprocessing Manager /
    suite2p plane workers) inherit the job, so they are reaped when the worker
    exits for any reason — clean exit, crash, taskkill, or the watchdog's
    os._exit. The job is created and held by the worker (not the gui), so it
    does not affect the worker's own survival of gui closure.

    Returns the job handle (caller must keep it alive) or None when
    unavailable: non-Windows, or the process is already in a job that forbids
    nesting (pre-Windows 8). Best-effort — never raises.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IO(ctypes.Structure):
            _fields_ = [
                (n, ctypes.c_ulonglong)
                for n in (
                    "ReadOperationCount", "WriteOperationCount",
                    "OtherOperationCount", "ReadTransferCount",
                    "WriteTransferCount", "OtherTransferCount",
                )
            ]

        class _EXT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC),
                ("IoInfo", _IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _EXT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        ):
            k32.CloseHandle(job)
            return None
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            k32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


class _CRCollapseWriter(io.TextIOBase):
    """Collapse carriage-return progress redraws to one log line per bar.

    tqdm (masknmf, imwrite) and pytorch-lightning ignore TQDM_DISABLE and
    emit every refresh; with stdio redirected to the log file each ``\\r``
    redraw lands as its own line, flooding the log. Only the text after
    the last ``\\r`` of each completed line survives, so a bar logs once,
    at its final state.
    """

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        out = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if "\r" in line:
                # keep the final redraw whether the writer leads or
                # trails with \r
                segs = [s for s in line.split("\r") if s.strip()]
                line = segs[-1] if segs else ""
            if line.strip():
                out.append(line)
        if out:
            self._stream.write("\n".join(out) + "\n")
            self._stream.flush()
        # cap a bar that redraws forever without ever emitting a newline
        if len(self._buf) > 65536 and "\r" in self._buf:
            self._buf = self._buf.rsplit("\r", 1)[-1]
        return len(text)

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


def main():
    """Main entry point for worker subprocess."""
    # force line buffering so print() output appears in logs immediately,
    # AND force utf-8 stdio so unicode prints from lsp (e.g. `→` U+2192
    # in run_plane_bin) don't crash on Windows where the default file
    # encoding for redirected stdio is cp1252.
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

    # disable tqdm dynamic display for file output (no terminal = no \r updates)
    os.environ["TQDM_DISABLE"] = "1"

    # not every library honors TQDM_DISABLE (masknmf's tqdm bars,
    # lightning's progress bar): squash their redraws at the stream level
    if sys.stdout is not None:
        sys.stdout = _CRCollapseWriter(sys.stdout)
    if sys.stderr is not None:
        sys.stderr = _CRCollapseWriter(sys.stderr)

    # early print so we can see the process started even if logging fails
    print(f"Worker starting (pid={os.getpid()})", file=sys.stderr, flush=True)

    if len(sys.argv) < 3:
        print("Usage: python -m mbo_utilities.gui._worker <task_type> <args_json>", file=sys.stderr)
        sys.exit(1)

    task_type = sys.argv[1]
    args_source = sys.argv[2]

    # args can arrive as a file path (new: avoids windows command-line
    # length limit) or as inline JSON (legacy / small payloads). detect
    # by checking if the string points to an existing file.
    args_file = None
    try:
        candidate = Path(args_source)
        if candidate.is_file() and candidate.suffix == ".json":
            args_file = candidate
            with open(args_file, "r", encoding="utf-8") as f:
                args = json.load(f)
        else:
            args = json.loads(args_source)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Failed to load args: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # clean up the args file immediately after loading so it
        # doesn't linger. the file is unique per task (UUID-named)
        # and no other process reads it.
        if args_file is not None:
            try:
                args_file.unlink(missing_ok=True)
            except OSError:
                pass

    # setup logging
    log_file = args.get("_log_file")
    uuid = args.get("_uuid")
    logger = setup_logging(log_file)

    logger.info(f"Worker started: task={task_type}, pid={os.getpid()}")

    # contain task-spawned children (pools, Manager, plane workers) so they
    # are reaped when this worker exits for any reason, including os._exit.
    global _JOB_HANDLE
    _JOB_HANDLE = _contain_in_job()
    if sys.platform == "win32":
        logger.debug(f"job containment: {'on' if _JOB_HANDLE else 'unavailable'}")

    # watchdog kills this process if it stalls for too long
    _start_watchdog(uuid, logger, log_file)

    # sample system + process-tree memory to a csv sidecar
    _start_mem_monitor(uuid, logger)

    # plugin pipelines contribute their worker via entry points; the
    # built-in TASKS table alone would not know about them.
    if task_type not in TASKS:
        try:
            from mbo_utilities.gui.tasks import load_plugin_tasks

            load_plugin_tasks()
        except Exception:
            logger.exception("Could not load plugin tasks")

    if task_type not in TASKS:
        logger.error(f"Unknown task type: {task_type}")
        print(f"Unknown task type: {task_type}", file=sys.stderr)
        _update_status(os.getpid(), "error", f"Unknown task type: {task_type}", uuid=uuid)
        sys.exit(1)

    task_func = TASKS[task_type]

    # run the task
    try:
        task_func(args, logger)
        logger.info("Task completed successfully")
        _update_status(os.getpid(), "completed", uuid=uuid)
        sys.exit(0)
    except Exception as e:
        try:
            from mbo_utilities._sysmem import mem_snapshot, format_mem_line
            logger.error("memory at failure: " + format_mem_line(mem_snapshot()))
        except Exception:
            pass
        logger.exception(f"Task failed: {e}")
        _update_status(os.getpid(), "error", message=str(e), details=traceback.format_exc(), uuid=uuid)
        sys.exit(1)


if __name__ == "__main__":
    main()
