"""Run external tools without ever hanging the pipeline.

Every tool this project drives is a third-party binary that can block forever: a trainer that
keeps a viewer open after it finishes, a solver that stops making progress, a GUI-subsystem
program that never closes. `run` puts three independent limits on each launch (a wall-clock
timeout, a no-output watchdog, and a cancel file) and always tears down the whole process tree.
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

_SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
_TASKKILL = str(_SYSTEM32 / "taskkill.exe")

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200


class ProcError(RuntimeError):
    """The tool exited with a failure code."""


class ProcTimeout(ProcError):
    """The tool ran past its wall-clock limit and was killed."""


class ProcStalled(ProcError):
    """The tool printed nothing for too long and was killed."""


class Cancelled(RuntimeError):
    """The cancel file appeared while the tool was running."""


@dataclass
class RunResult:
    returncode: int
    seconds: float
    # True when the process had to be killed after printing its completion marker.
    killed_after_done: bool = False


def _make_kill_on_close_job():
    """Create a Windows job object that kills its processes when this Python process dies.

    Without it, killing the orchestrator (closing its window, a parent timeout) would leave the
    GPU tool running in the background with nothing watching it.
    """
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_ulong),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_ulong),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_ulong),
            ("SchedulingClass", ctypes.c_ulong),
        ]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMIT),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    if not kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
        return None
    return job


_JOB = _make_kill_on_close_job()


def _assign_to_job(proc: subprocess.Popen) -> None:
    if _JOB is None:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    # Best effort: if this fails the explicit tree kill below still covers normal operation.
    kernel32.AssignProcessToJobObject(_JOB, int(proc._handle))


def kill_tree(pid: int) -> None:
    """Kill a process and everything it started."""
    if os.name == "nt":
        subprocess.run([_TASKKILL, "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=CREATE_NO_WINDOW)
    else:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def run(
    cmd: Sequence[str],
    *,
    log_path: Path,
    timeout: float,
    stall_timeout: Optional[float] = 600,
    on_line: Optional[Callable[[str], None]] = None,
    done_marker: Optional[str] = None,
    done_grace: float = 30,
    cancel_file: Optional[Path] = None,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    ok_codes: Sequence[int] = (0,),
) -> RunResult:
    """Run `cmd` to completion, appending its combined output to `log_path`.

    timeout        seconds of wall-clock time before the tool is killed (ProcTimeout).
    stall_timeout  seconds without a single line of output before it is killed (ProcStalled).
                   Pass None for tools that print nothing.
    done_marker    regex; once a line matches, the tool is considered finished and gets
                   `done_grace` seconds to exit by itself before being killed. A kill at that
                   point is success, not failure: it is how tools that linger are handled.
    cancel_file    if this path appears, the tool is killed and Cancelled is raised.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    marker = re.compile(done_marker) if done_marker else None

    state = {"last_output": time.monotonic(), "done_at": None}

    with open(log_path, "ab") as log:
        log.write(("\n$ " + subprocess.list2cmdline(list(cmd)) + "\n").encode("utf-8"))
        log.flush()
        proc = subprocess.Popen(
            list(cmd),
            stdin=subprocess.DEVNULL,  # a tool that asks a question must fail, not wait
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=env,
            creationflags=(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0,
        )
        _assign_to_job(proc)

        def pump() -> None:
            # Progress bars rewrite one line with "\r"; treat both terminators as line ends so
            # the watchdog and the progress parser see each update.
            buf = b""
            while True:
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                state["last_output"] = time.monotonic()
                buf += chunk
                parts = re.split(rb"[\r\n]+", buf)
                buf = parts.pop()
                for raw in parts:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    if marker and state["done_at"] is None and marker.search(line):
                        state["done_at"] = time.monotonic()
                    if on_line:
                        try:
                            on_line(line)
                        except Exception:
                            pass  # a progress-parsing bug must not take down the run

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        started = time.monotonic()
        killed_after_done = False
        failure: Optional[BaseException] = None

        try:
            while proc.poll() is None:
                time.sleep(0.5)
                now = time.monotonic()
                if cancel_file is not None and Path(cancel_file).exists():
                    failure = Cancelled("cancelled by user")
                elif state["done_at"] is not None:
                    if now - state["done_at"] > done_grace:
                        killed_after_done = True
                        kill_tree(proc.pid)
                        break
                    continue
                elif now - started > timeout:
                    failure = ProcTimeout(
                        f"{Path(cmd[0]).name} exceeded its {int(timeout)}s limit and was stopped")
                elif stall_timeout is not None and now - state["last_output"] > stall_timeout:
                    failure = ProcStalled(
                        f"{Path(cmd[0]).name} printed nothing for {int(stall_timeout)}s "
                        f"and was stopped")
                if failure is not None:
                    kill_tree(proc.pid)
                    break
        finally:
            if proc.poll() is None:
                kill_tree(proc.pid)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            reader.join(timeout=10)

        seconds = time.monotonic() - started
        if failure is not None:
            log.write(f"\n[scan3d] {failure}\n".encode("utf-8"))
            raise failure

        code = proc.returncode if proc.returncode is not None else -1
        finished = state["done_at"] is not None
        if code not in ok_codes and not (finished and killed_after_done):
            raise ProcError(
                f"{Path(cmd[0]).name} failed with exit code {code}; see {log_path}")
        return RunResult(returncode=code, seconds=seconds, killed_after_done=killed_after_done)
