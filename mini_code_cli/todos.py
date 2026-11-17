"""Session-local structured todo list for the agent.

``--dynamic-todo-tracking`` adds three tools instead of asking the model to
maintain a free-form scratchpad:

* ``todo_write``  -- replace the whole list from a JSON array;
* ``todo_update`` -- change one item's status/content by id;
* ``todo_list``   -- render the current list.

The list is stored in the *session cache* (``<cache>/<session>/todos.json``),
not in ``--allowed-dir``: it is working state for the agent, not a project file,
and it must not pollute the task's git diff or a benchmark's filesystem checks.
The file is small, human-readable JSON and is restored with the session.
"""
import functools
import json
import os
import tempfile

TODO_FILENAME = "todos.json"
STATUSES = ("pending", "in_progress", "completed")
_STATUS_ALIASES = {
    "todo": "pending",
    "pending": "pending",
    "open": "pending",
    "in_progress": "in_progress",
    "in-progress": "in_progress",
    "inprogress": "in_progress",
    "doing": "in_progress",
    "active": "in_progress",
    "completed": "completed",
    "complete": "completed",
    "done": "completed",
    "finished": "completed",
}

MARKERS = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}


class TodoError(ValueError):
    """A caller-visible todo validation error."""


def _atomic_write(path, text):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".mini-todo-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _normalise_status(value, default="pending"):
    status = str(value or default).strip().lower().replace(" ", "_")
    if status not in _STATUS_ALIASES:
        raise TodoError(
            f"unknown todo status {value!r}; use one of {', '.join(STATUSES)}")
    return _STATUS_ALIASES[status]


class TodoStore:
    """Read/write one session's todo list."""

    def __init__(self, session_dir):
        self.path = os.path.join(session_dir, TODO_FILENAME)
        self.items = []
        self.load()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            self.items = []
            return self.items
        self.items = self._normalise(payload.get("items", [])
                                     if isinstance(payload, dict) else payload)
        return self.items

    def save(self):
        _atomic_write(self.path, json.dumps({"items": self.items}, indent=2)
                      + "\n")

    def _normalise(self, items):
        if not isinstance(items, list):
            raise TodoError("the todo list must be a JSON array")
        result = []
        for index, item in enumerate(items, 1):
            if isinstance(item, str):
                content, status = item, "pending"
            elif isinstance(item, dict):
                content = item.get("content") or item.get("title") or ""
                status = item.get("status", "pending")
            else:
                raise TodoError(
                    f"todo item {index} must be a string or an object with "
                    f"'content' and 'status'")
            content = " ".join(str(content).split())
            if not content:
                raise TodoError(f"todo item {index} has no content")
            result.append({"id": index, "content": content,
                           "status": _normalise_status(status)})
        return result

    def replace(self, items):
        self.items = self._normalise(items)
        self.save()
        return self.render()

    def update(self, item_id, status=None, content=None):
        try:
            item_id = int(item_id)
        except (TypeError, ValueError):
            raise TodoError(f"todo id must be an integer, got {item_id!r}")
        item = next((i for i in self.items if i["id"] == item_id), None)
        if item is None:
            raise TodoError(f"no todo with id {item_id}")
        if status is not None and str(status).strip():
            item["status"] = _normalise_status(status, item["status"])
        if content is not None and str(content).strip():
            item["content"] = " ".join(str(content).split())
        self.save()
        return self.render()

    def render(self):
        if not self.items:
            return "Todo list is empty."
        lines = []
        for item in self.items:
            lines.append(f"{item['id']}. {MARKERS[item['status']]} "
                         f"{item['content']}")
        return "\n".join(lines)


def _load_items(items):
    """Parse the model-facing ``items`` argument (JSON string or list)."""
    if isinstance(items, (list, tuple)):
        return list(items)
    text = str(items or "").strip()
    if not text:
        raise TodoError("items must be a non-empty JSON array")
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise TodoError(f"items is not valid JSON: {exc}")
    if not isinstance(value, list):
        raise TodoError("items must be a JSON array of todo objects")
    return value


def todo_write(items, store):
    """Replace the todo list with a JSON array of {content, status} objects.

    Param: items (required) -- a JSON array, e.g.
    ``[{"content": "read the task", "status": "completed"},
       {"content": "write the solver", "status": "in_progress"}]``.
    Statuses are ``pending``, ``in_progress`` or ``completed``. Ids are
    assigned in order and returned in the rendered list.
    """
    try:
        return store.replace(_load_items(items))
    except TodoError as exc:
        return f"Error: {exc}"


def todo_update(item_id: int, status: str = "", content: str = "", store=None):
    """Update one todo item by id.

    Param: item_id (required); status (optional: pending/in_progress/completed);
    content (optional: replacement text). Omit a field to leave it unchanged.
    """
    try:
        return store.update(item_id, status=status or None,
                            content=content or None)
    except TodoError as exc:
        return f"Error: {exc}"


def todo_list(store):
    """Return the current todo list, with ids and status markers."""
    return store.render()


def todo_functions(store):
    """The tool mapping to register when dynamic todo tracking is enabled."""
    return {
        "todo_write": functools.partial(todo_write, store=store),
        "todo_update": functools.partial(todo_update, store=store),
        "todo_list": functools.partial(todo_list, store=store),
    }
