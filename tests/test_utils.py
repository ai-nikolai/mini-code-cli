"""Tests for the model-call transport: retries, and the shape of the result.

The loop is deliberately tolerant of a dead server (it degrades to a failed
step), but a *transient* failure should not cost a turn -- these tests pin the
bounded retry and the fact that a client error is not retried.
"""
import unittest
from unittest import mock

from mini_code_cli import utils


class FakeResponse:
    """The slice of ``requests.Response`` the transport uses."""

    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = "{}"

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise utils.requests.exceptions.RequestException(
                f"{self.status_code} error")


def reply(content="hello"):
    return {
        "choices": [{
            "message": {"content": content, "tool_calls": []},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


class RetryTest(unittest.TestCase):

    def call(self, post):
        with mock.patch.object(utils.requests, "post", post), \
             mock.patch.object(utils.time, "sleep") as sleep, \
             mock.patch.object(utils, "_report"):
            result = utils.call_openai_server(
                [{"role": "user", "content": "hi"}], server_url="http://x")
        return result, sleep

    def test_a_transient_failure_is_retried_then_succeeds(self):
        post = mock.Mock(side_effect=[
            utils.requests.exceptions.RequestException("connection reset"),
            FakeResponse(200, reply("recovered")),
        ])
        result, sleep = self.call(post)

        content, tool_calls, usage, reasoning = result
        self.assertEqual(content, "recovered")
        self.assertEqual(tool_calls, [])
        self.assertEqual(usage["total_tokens"], 5)
        self.assertIsNone(reasoning)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    def test_a_client_error_is_not_retried(self):
        post = mock.Mock(return_value=FakeResponse(400))
        result, sleep = self.call(post)

        self.assertEqual(result, (None, None, None, None))
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_all_attempts_fail_boundedly(self):
        post = mock.Mock(
            side_effect=utils.requests.exceptions.RequestException("down"))
        result, sleep = self.call(post)

        self.assertEqual(result, (None, None, None, None))
        self.assertEqual(post.call_count, utils.LLM_MAX_ATTEMPTS)
        self.assertEqual(sleep.call_count, utils.LLM_MAX_ATTEMPTS - 1)

    def test_a_server_error_is_retried(self):
        post = mock.Mock(side_effect=[
            FakeResponse(503),
            FakeResponse(200, reply("ok")),
        ])
        result, _ = self.call(post)
        self.assertEqual(result[0], "ok")
        self.assertEqual(post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
