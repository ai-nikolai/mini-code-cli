"""Background shell commands and sub-agent tasks.

The foreground agent loop is synchronous: it blocks in ``subprocess.run`` or an
HTTP call. This module gives it a small, thread-safe registry so long-running
work can be started and inspected without blocking the current turn:

  * ``shell`` tasks are real OS processes (their own process group, so cancel
    kills the whole tree);
  * ``subagent`` tasks run a nested :class:`~mini_code_cli.subagents.SubAgent`
    in a worker thread and pass it a cancellation event.

Output is streamed to ``<session>/background/<id>.log`` and mirrored into a
bounded in-memory tail. The registry never writes to the terminal: the
``background`` tool and the ``/background`` command read consistent snapshots,
so a worker thread cannot corrupt the console.

Nothing survives a process restart. The log files do, which is useful for
post-mortems, but PIDs are never reattached.
"""

import os
import signal
import subprocess
import threading
import time
from collections import deque

MAX_TAIL_LINES = 2_000
DEFAULT_TAIL_LINES = 40
MAX_TOOL_TAIL_CHARS = 20_000

SHELL = "shell"
SUBAGENT = "subagent"


def _truncate(text, limit=MAX_TOOL_TAIL_CHARS):
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    return text[-limit:]


def _read_tail(path, lines=DEFAULT_TAIL_LINES):
    """Best-effort tail of a log file; never raises."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=max(int(lines), 0)))
    except OSError:
        return ""


class BackgroundTask:
    """One task's state. Mutated by its worker and read through snapshots."""

    def __init__(self, task_id, kind, label, log_path):
        self.id = task_id
        self.kind = kind
        self.label = label
        self.log_path = log_path
        self.status = "running"
        self.pid = None
        self.started_at = time.time()
        self.ended_at = None
        self.exit_code = None
        self.result = ""
        self.error = ""
        self.cancel_reason = ""
        self.done = threading.Event()
        self.cancel_event = threading.Event()
        self._output = deque(maxlen=MAX_TAIL_LINES)
        self._lock = threading.Lock()
        self.thread = None
        self.process = None
        self.timer = None

    def append_output(self, text):
        if not text:
            return
        with self._lock:
            self._output.append(str(text))

    def output_tail(self, lines=DEFAULT_TAIL_LINES):
        with self._lock:
            text = "".join(self._output)
        if not text:
            text = _read_tail(self.log_path, lines)
        return _truncate(text)

    def snapshot(self):
        with self._lock:
            age = (self.ended_at or time.time()) - self.started_at
            return {
                "id": self.id,
                "kind": self.kind,
                "label": self.label,
                "status": self.status,
                "pid": self.pid,
                "age_seconds": max(age, 0.0),
                "exit_code": self.exit_code,
                "log_path": self.log_path,
                "result": self.result,
                "error": self.error,
            }

    def finish(self, status, exit_code=None, result="", error=""):
        with self._lock:
            if self.ended_at is None:
                self.ended_at = time.time()
            self.status = status
            self.exit_code = exit_code
            if result:
                self.result = str(result)
            if error:
                self.error = str(error)
        self.done.set()


class BackgroundTaskManager:
    """Owns the tasks for one foreground session."""

    def __init__(self, log_dir):
        self.log_dir = str(log_dir)
        self._tasks = {}
        self._next_id = 0
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def set_log_dir(self, log_dir):
        """Point future tasks at another directory (used on a restore)."""
        self.log_dir = str(log_dir)

    def _new_task(self, kind, label):
        with self._lock:
            self._next_id += 1
            task_id = f"bg-{self._next_id}"
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, f"{task_id}.log")
            task = BackgroundTask(task_id, kind, label, log_path)
            self._tasks[task_id] = task
            return task

    def get(self, task_id):
        with self._lock:
            return self._tasks.get(str(task_id))

    def list(self):
        with self._lock:
            return [task.snapshot() for task in self._tasks.values()]

    def running(self):
        return [snapshot for snapshot in self.list()
                if snapshot["status"] in ("running", "cancelling")]

    def shutdown(self):
        """Cancel everything still running; called when the session exits."""
        for snapshot in self.running():
            self.cancel(snapshot["id"])
        for task in list(self._tasks.values()):
            if task.thread is not None:
                task.thread.join(timeout=2.0)

    # -- starting ----------------------------------------------------------

    def start_shell(self, command, cwd=None, max_seconds=None):
        """Start ``command`` in its own process group and return the task."""
        command = str(command or "").strip()
        if not command:
            raise ValueError("command must not be empty")
        task = self._new_task(SHELL, command)
        try:
            env = os.environ.copy()
            kwargs = {}
            if os.name == "posix":
                # Give the task its own group so cancel kills its children too.
                kwargs["start_new_session"] = True
            process = subprocess.Popen(
                command, shell=True, cwd=cwd or None,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, **kwargs)
        except OSError as exc:
            task.finish("failed", error=str(exc))
            return task
        task.process = process
        task.pid = process.pid
        task.thread = threading.Thread(
            target=self._monitor_shell, args=(task, process), daemon=True)
        task.thread.start()
        if max_seconds:
            task.timer = threading.Timer(
                float(max_seconds), self._cancel_task,
                args=(task,), kwargs={"reason": "timeout"})
            task.timer.daemon = True
            task.timer.start()
        return task

    def start_callable(self, label, func, max_seconds=None):
        """Run ``func(should_stop)`` in a worker thread and track it."""
        task = self._new_task(SUBAGENT, label)
        task.thread = threading.Thread(
            target=self._monitor_callable, args=(task, func), daemon=True)
        task.thread.start()
        if max_seconds:
            task.timer = threading.Timer(
                float(max_seconds), self._cancel_task,
                args=(task,), kwargs={"reason": "timeout"})
            task.timer.daemon = True
            task.timer.start()
        return task

    # -- workers -----------------------------------------------------------

    def _monitor_shell(self, task, process):
        status, error = "failed", ""
        try:
            with open(task.log_path, "w", encoding="utf-8",
                      errors="replace") as log:
                stream = process.stdout
                while stream is not None:
                    chunk = stream.readline()
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    task.append_output(chunk)
            code = process.wait()
        except Exception as exc:  # noqa: BLE001 - a worker must never escape
            code, error = None, str(exc)
        finally:
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
            if task.cancel_reason:
                status = "cancelled"
            elif code == 0:
                status = "done"
            elif code is not None:
                status = "failed"
        task.finish(status, exit_code=code, error=error)

    def _monitor_callable(self, task, func):
        status, result, error = "done", "", ""
        try:
            result = func(task.cancel_event.is_set)
        except Exception as exc:  # noqa: BLE001
            status, error = "failed", str(exc)
        if task.cancel_event.is_set():
            status = "cancelled"
            if not error:
                error = task.cancel_reason or "cancelled"
        task.finish(status, result=result, error=error)

    # -- inspection / control ---------------------------------------------

    def tail(self, task_id, lines=DEFAULT_TAIL_LINES):
        task = self.get(task_id)
        if task is None:
            return None
        return task.output_tail(lines)

    def wait(self, task_id, timeout=None):
        task = self.get(task_id)
        if task is None:
            return None
        task.done.wait(None if timeout is None else max(float(timeout), 0.0))
        return task.snapshot()

    def cancel(self, task_id):
        task = self.get(task_id)
        if task is None:
            return None
        self._cancel_task(task)
        return task.snapshot()

    def _cancel_task(self, task, reason="cancelled"):
        if task.done.is_set():
            return
        task.cancel_reason = task.cancel_reason or reason
        task.cancel_event.set()
        if task.kind == SHELL and task.pid:
            with task._lock:
                task.status = "cancelling"
            self._kill_group(task, signal.SIGTERM)
            if not task.done.wait(2.0):
                self._kill_group(task, getattr(signal, "SIGKILL", signal.SIGTERM))
        else:
            with task._lock:
                task.status = "cancelling"

    @staticmethod
    def _kill_group(task, sig):
        pid = task.pid
        if os.name == "posix" and hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(pid), sig)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        process = task.process
        if process is not None:
            try:
                process.terminate() if sig == signal.SIGTERM else process.kill()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# the `background` tool
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTION = """Start, inspect, wait for, or cancel background tasks.

Use `start` for long-running work you do not want to block on (a full test
suite, a build, a download, a sub-agent). Shell tasks run in the background
with their output stored in a log; sub-agent tasks run a nested agent.
Poll with `list`/`output`, wait with `wait`, and stop with `cancel`.
"""

_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "background",
        "description": _TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "list", "output", "wait", "cancel"],
                    "description": "What to do.",
                },
                "kind": {
                    "type": "string",
                    "enum": [SHELL, SUBAGENT],
                    "description": "Task kind for action=start (default shell).",
                },
                "command": {
                    "type": "string",
                    "description": "Shell command for kind=shell, action=start.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for a shell task.",
                },
                "subagent": {
                    "type": "string",
                    "description": "Sub-agent name for kind=subagent.",
                },
                "task": {
                    "type": "string",
                    "description": "Task text for kind=subagent.",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Optional step cap for a sub-agent task.",
                },
                "task_id": {
                    "type": "string",
                    "description": "Task id (bg-1, ...) for output/wait/cancel.",
                },
                "tail": {
                    "type": "integer",
                    "description": "Lines of output to return (default 40).",
                },
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait (0 = return immediately).",
                },
            },
            "required": ["action"],
        },
    },
}


def _format_task(snapshot):
    state = snapshot.get("status", "?")
    age = snapshot.get("age_seconds", 0.0)
    label = (snapshot.get("label") or "").replace("\n", " ")
    if len(label) > 100:
        label = label[:97] + "..."
    extra = ""
    if snapshot.get("exit_code") is not None:
        extra = f" exit={snapshot['exit_code']}"
    elif snapshot.get("error"):
        extra = f" error={snapshot['error'][:100]}"
    return f"{snapshot['id']} {state} {age:6.1f}s{extra}: {label}"


def background_tool(manager, subagent_runner, action, kind=SHELL, command=None,
                    cwd=None, subagent=None, task=None, max_steps=None,
                    task_id=None, tail=DEFAULT_TAIL_LINES, timeout=0):
    """Tool entry point; ``manager`` and ``subagent_runner`` are bound by the session."""
    action = str(action or "").strip().lower()
    kind = str(kind or SHELL).strip().lower()
    try:
        tail = max(0, min(int(tail or 0), 500))
    except (TypeError, ValueError):
        tail = DEFAULT_TAIL_LINES
    try:
        timeout = max(0.0, float(timeout or 0))
    except (TypeError, ValueError):
        timeout = 0.0

    if action == "list":
        snapshots = manager.list()
        if not snapshots:
            return "No background tasks."
        return "\n".join(_format_task(snapshot) for snapshot in snapshots)

    if action == "start":
        if kind not in (SHELL, SUBAGENT):
            return f"Error: kind must be {SHELL!r} or {SUBAGENT!r}."
        if kind == SUBAGENT:
            if not subagent or not task:
                return "Error: kind=subagent requires `subagent` and `task`."
            if subagent_runner is None:
                return "Error: no sub-agents are enabled in this session."
            return _start_subagent(manager, subagent_runner, subagent, task,
                                   max_steps)
        if not command:
            return "Error: kind=shell requires `command`."
        try:
            started = manager.start_shell(command, cwd=cwd)
        except Exception as exc:  # noqa: BLE001
            return f"Error: could not start background task: {exc}"
        return (f"Started {started.id} ({started.kind}): {started.label}\n"
                f"Use background(action='output', task_id='{started.id}') "
                f"to read its output.")

    if action == "output":
        if not task_id:
            return "Error: output requires `task_id`."
        text = manager.tail(task_id, tail)
        if text is None:
            return f"Error: no background task {task_id!r}."
        snapshot = manager.get(task_id).snapshot()
        header = _format_task(snapshot)
        return (header + ("\n" + text if text else "\n(no output yet)"))

    if action == "wait":
        if not task_id:
            return "Error: wait requires `task_id`."
        snapshot = manager.wait(task_id, timeout)
        if snapshot is None:
            return f"Error: no background task {task_id!r}."
        text = manager.tail(task_id, tail)
        header = _format_task(snapshot)
        return (header + ("\n" + text if text else "\n(no output)"))

    if action == "cancel":
        if not task_id:
            return "Error: cancel requires `task_id`."
        snapshot = manager.cancel(task_id)
        if snapshot is None:
            return f"Error: no background task {task_id!r}."
        return _format_task(snapshot)

    return ("Error: unknown action. Use start, list, output, wait or cancel.")


def _start_subagent(manager, subagent_runner, subagent, task, max_steps):
    label = f"subagent {subagent}: {str(task).replace(chr(10), ' ')}"

    def worker(should_stop):
        return subagent_runner(subagent, task, max_steps, should_stop)

    try:
        started = manager.start_callable(label, worker)
    except Exception as exc:  # noqa: BLE001
        return f"Error: could not start sub-agent task: {exc}"
    return (f"Started {started.id} ({started.kind}): {started.label}\n"
            f"Use background(action='output', task_id='{started.id}') "
            f"to read its result.")


def background_tools():
    """The OpenAI-style tool schema contributed by this module."""
    return [dict(_TOOL_SCHEMA)]


def background_tools_dict(manager, subagent_runner=None):
    """The dispatch entry, with the session's manager and sub-agent runner bound."""
    import functools

    return {
        "background": {
            "function": functools.partial(
                background_tool, manager, subagent_runner),
            "description": _TOOL_DESCRIPTION,
        }
    }
