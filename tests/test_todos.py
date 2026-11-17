"""Tests for the dynamic todo store and its agent-facing tools."""
import json
import os
import tempfile
import unittest

from mini_code_cli import todos, tools


class TodoStoreTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_write_update_and_render(self):
        store = todos.TodoStore(self.directory)
        text = store.replace([
            {"content": "read the task", "status": "done"},
            {"content": "write the solver", "status": "doing"},
            {"content": "run the tests"},
        ])
        self.assertIn("1. [x] read the task", text)
        self.assertIn("2. [~] write the solver", text)
        self.assertIn("3. [ ] run the tests", text)

        text = store.update(2, status="completed")
        self.assertIn("2. [x] write the solver", text)
        self.assertIn("3. [ ] run the tests", text)
        store.update(3, content="run the hidden tests")
        self.assertEqual(store.items[2]["content"], "run the hidden tests")

    def test_round_trip_in_the_session_file(self):
        store = todos.TodoStore(self.directory)
        store.replace(["one", "two"])
        again = todos.TodoStore(self.directory)
        self.assertEqual([item["content"] for item in again.items],
                         ["one", "two"])
        path = os.path.join(self.directory, todos.TODO_FILENAME)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(len(json.load(handle)["items"]), 2)

    def test_invalid_items_return_tool_errors(self):
        store = todos.TodoStore(self.directory)
        self.assertIn("not valid JSON", todos.todo_write("not json", store))
        self.assertIn("unknown todo status",
                      todos.todo_write('[{"content": "x", "status": "??"}]',
                                       store))
        self.assertIn("no todo with id", todos.todo_update(9, store=store))
        self.assertIn("must be an integer",
                      todos.todo_update("x", store=store))

    def test_tools_are_registered_without_the_bound_store_argument(self):
        store = todos.TodoStore(self.directory)
        functions = todos.todo_functions(store)
        self.assertEqual(set(functions),
                         {"todo_write", "todo_update", "todo_list"})
        schemas = {tool["function"]["name"]: tool
                   for tool in tools.util_get_tools(functions=functions)}
        self.assertNotIn("store",
                         schemas["todo_write"]["function"]["parameters"]
                         ["properties"])
        self.assertNotIn("store",
                         schemas["todo_list"]["function"]["parameters"]
                         ["properties"])
        self.assertIn("items",
                      schemas["todo_write"]["function"]["parameters"]
                      ["properties"])

    def test_corrupt_store_is_empty_not_fatal(self):
        path = os.path.join(self.directory, todos.TODO_FILENAME)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        store = todos.TodoStore(self.directory)
        self.assertEqual(store.items, [])
        self.assertIn("empty", store.render())


if __name__ == "__main__":
    unittest.main()
