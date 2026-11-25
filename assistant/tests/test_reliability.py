import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import requests
from duckduckgo_search.exceptions import RatelimitException

from agent import harness, mcp, planner
from agent.results import failure, success
from agent.skills import file_ops, telegram_ops, web_search


class ToolReliabilityTests(unittest.TestCase):
    def test_native_tool_schema_and_response(self):
        with patch.object(planner.requests, "post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {"message": {"content": "", "tool_calls": [{
                "function": {"name": "write_file", "arguments": {"path": "/workspace/a.txt", "content": "hello"}},
            }]}}
            tool = planner.parse_tool_call(planner.ask_llm([]))
            schemas = {entry["function"]["name"]: entry["function"]["parameters"]
                       for entry in post.call_args.kwargs["json"]["tools"]}
        self.assertEqual(tool["name"], "write_file")
        self.assertEqual(tool["args"]["content"], "hello")
        self.assertEqual(schemas["write_file"]["required"], ["path", "content"])
        self.assertEqual(schemas["run_code"]["required"], ["code"])
        self.assertFalse(schemas["github_commit"]["additionalProperties"])
        self.assertIn("write_file(path, content)", mcp.call_tool("write_file", {"path": "/workspace/a.txt"})["error"])

    def test_provider_outage_does_not_repeat_with_different_queries(self):
        with patch.object(harness, "ask_llm", side_effect=[
            'TOOL: web_search\nARGS: {"query": "first"}', 'TOOL: web_search\nARGS: {"query": "second"}',
        ]), patch.object(harness, "call_tool", return_value=failure("Search unavailable.", {"unavailable": True})) as tool:
            result = harness._loop("Search the web", False)
        self.assertEqual(result.answer, "Search unavailable.")
        tool.assert_called_once_with("web_search", {"query": "first"})

    def test_empty_search_allows_only_one_reformulation(self):
        replies = [f'TOOL: web_search\nARGS: {{"query": "query {n}"}}' for n in range(3)]
        with patch.object(harness, "ask_llm", side_effect=replies), \
                patch.object(harness, "call_tool", return_value=failure("No relevant links found.")) as tool:
            result = harness._loop("Search the web", False)
        self.assertEqual(result.answer, "No relevant links found.")
        self.assertEqual(tool.call_count, 2)

    def test_answer_uses_returned_source_when_model_omits_it(self):
        source = "https://en.wiktionary.org/wiki/BRB"
        with patch.object(harness, "ask_llm", side_effect=[
            'TOOL: web_search\nARGS: {"query": "BRB meaning"}', "BRB means be right back.",
        ]), patch.object(harness, "call_tool", return_value=success([{"url": source}])):
            result = harness._loop("Can you search the web for what BRB means?", False)
        self.assertEqual(result.answer, f"BRB means be right back.\n\nSource\n{source}")

    def test_workspace_listing_and_path_errors(self):
        temp_base = "/workspace" if Path("/workspace").is_dir() else None
        with TemporaryDirectory(dir=temp_base) as temp, patch.object(file_ops, "WORKSPACE", Path(temp)):
            self.assertTrue(file_ops.write_file("/workspace/project/notes.md", "notes")["ok"])
            (Path(temp) / ".assistant").mkdir()
            self.assertEqual(file_ops.list_files("/workspace")["result"]["entries"], [
                {"path": "/workspace/project", "type": "directory"},
            ])
            self.assertIn("directory", file_ops.read_file("/workspace/project")["error"])
            self.assertIn("File not found", file_ops.read_file("/workspace/project/missing.md")["error"])
            self.assertFalse(file_ops.list_files("/workspace/..")["ok"])
            self.assertFalse(file_ops.write_file("/workspace/project/notes.md", "replace")["ok"])
            self.assertEqual(file_ops.read_file("/workspace/project/notes.md")["result"], "notes")


class SearchReliabilityTests(unittest.TestCase):
    def test_rate_limit_falls_back_to_bounded_mwmbl_results(self):
        results = [
            {"title": [{"value": "First"}], "url": "https://example.com/1", "extract": [{"value": "<b>Snippet</b>"}]},
            {"title": [{"value": "Unsafe"}], "url": "javascript:alert(1)"},
        ]
        with patch.dict(os.environ, {"WEB_SEARCH_FALLBACK": "1"}), \
                patch.object(web_search.DDGS, "text", side_effect=RatelimitException), \
                patch.object(web_search.requests, "get") as get:
            response = get.return_value.__enter__.return_value
            response.status_code = 200
            response.iter_content.return_value = [json.dumps(results).encode()]
            result = web_search.web_search("example")
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], [{"title": "First", "url": "https://example.com/1", "snippet": "Snippet"}])
        self.assertFalse(get.call_args.kwargs["allow_redirects"])

    def test_duckduckgo_only_never_sends_query_to_fallback(self):
        with patch.dict(os.environ, {"WEB_SEARCH_FALLBACK": "0"}), \
                patch.object(web_search.DDGS, "text", side_effect=RatelimitException), \
                patch.object(web_search, "_get_json") as get:
            result = web_search.web_search("what does BRB mean")
        get.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertTrue(result["result"]["unavailable"])

    def test_unrelated_links_are_rejected(self):
        unrelated = [{"title": "Crossword answer", "href": "https://example.org/crossword", "body": "Clues"}]
        with patch.dict(os.environ, {"WEB_SEARCH_FALLBACK": "1"}), \
                patch.object(web_search.DDGS, "text", return_value=unrelated), \
                patch.object(web_search, "_mwmbl", return_value=unrelated):
            result = web_search.web_search("official python website")
        self.assertFalse(result["ok"])
        self.assertIn("more specific query", result["error"])

    def test_brb_definition_follows_case_alias_and_cites_wiktionary(self):
        lower = {"en": [
            {"language": "Translingual", "definitions": [{"definition": "ISO language code."}]},
            {"language": "English", "definitions": [{"definition": "Alternative letter-case form of <a>BRB</a>."}]},
        ]}
        upper = {"en": [{"language": "English", "definitions": [{"definition": "Initialism of <a>be right back</a>."}]}]}
        with patch.dict(os.environ, {"WEB_SEARCH_FALLBACK": "1"}), \
                patch.object(web_search, "_get_json", side_effect=[lower, upper]) as get, \
                patch.object(web_search.DDGS, "text") as duckduckgo:
            result = web_search.web_search("can you search the web for what the word brb means?")
        self.assertEqual(result["result"], [
            {"title": "BRB — Wiktionary", "url": "https://en.wiktionary.org/wiki/BRB", "snippet": "Initialism of be right back."},
        ])
        self.assertEqual([call.args[0].rsplit("/", 1)[-1] for call in get.call_args_list], ["brb", "BRB"])
        duckduckgo.assert_not_called()


class RequestBoundaryTests(unittest.TestCase):
    def test_outside_file_targets_are_refused_before_prompting_the_model(self):
        for target in ("/etc/passwd", "/workspace/../etc/passwd", "/workspace/..", "/workspace-backup/file",
                       "file:///etc/passwd", "../etc/passwd", r"C:\Windows\system.ini"):
            with self.subTest(target=target), patch.object(harness, "ask_llm") as ask, patch.object(harness, "call_tool") as tool:
                outcome = harness._loop(f"write hello to {target}", False)
                self.assertFalse(outcome.succeeded)
                self.assertIn("outside", outcome.answer)
                ask.assert_not_called()
                tool.assert_not_called()

    def test_rewritten_path_cannot_bypass_the_original_request(self):
        reply = 'TOOL: write_file\nARGS: {"path":"/workspace/etc/passwd","content":"hello"}'
        with patch.object(harness, "ask_llm", return_value=reply), patch.object(harness, "call_tool") as tool:
            outcome = harness._loop("I need one change. Put hello in /etc/passwd.", False)
        self.assertIn("outside", outcome.answer)
        tool.assert_not_called()

    def test_workspace_paths_and_literal_content_still_work(self):
        path = "/workspace/example.txt"
        reply = 'TOOL: write_file\nARGS: {"path":"/workspace/example.txt","content":"/etc/passwd"}'
        with patch.object(harness, "ask_llm", side_effect=[reply, "Done."]), \
                patch.object(harness, "call_tool", return_value=success({"path": path})) as tool:
            outcome = harness._loop(f"Write {path} containing `/etc/passwd`.", False)
        tool.assert_called_once_with("write_file", {"path": path, "content": "/etc/passwd"})
        self.assertTrue(outcome.succeeded)

    def test_raw_tool_json_is_neither_dispatched_nor_delivered(self):
        raw = '```arduino\n{"name": "run_code", "parameters": {"code": "import os"}}\n```'
        with patch.object(harness, "ask_llm", return_value=raw) as ask, patch.object(harness, "call_tool") as tool, \
                patch.object(harness, "send_telegram", return_value=success({})) as send:
            answer = harness.run("run import os")
        self.assertEqual(ask.call_count, 2)
        self.assertIn("invalid tool call", answer)
        self.assertNotIn("parameters", answer)
        tool.assert_not_called()
        send.assert_called_once_with(answer)

    def test_native_import_os_is_refused_in_plain_text(self):
        reply = 'TOOL: run_code\nARGS: {"code":"import os"}'
        with patch.object(harness, "ask_llm", return_value=reply) as ask, \
                patch.object(harness, "send_telegram", return_value=success({})) as send, patch.object(requests, "Session") as session:
            answer = harness.run("run import os")
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(answer, "The snippet contains a prohibited operation.")
        session.assert_not_called()
        send.assert_called_once_with(answer)


class TelegramReliabilityTests(unittest.TestCase):
    def test_send_tool_cannot_deliver_internal_json(self):
        raw = '{"tool_calls": [{"function": {"name": "run_code", "arguments": {"code": "import os"}}}]}'
        with patch.object(telegram_ops, "telegram_request") as telegram:
            self.assertFalse(telegram_ops.send_telegram(raw)["ok"])
        telegram.assert_not_called()

    def test_conflict_has_a_specific_action_and_no_token(self):
        with patch.object(telegram_ops.requests, "post") as post:
            post.return_value.ok = False
            post.return_value.status_code = 409
            post.return_value.json.return_value = {"ok": False, "error_code": 409}
            with self.assertRaisesRegex(telegram_ops.TelegramError, "Another poller or webhook") as caught:
                telegram_ops.telegram_request("test-secret", "getUpdates", {})
        self.assertNotIn("test-secret", str(caught.exception))

    def test_safe_retry_and_unicode_limit(self):
        error = telegram_ops.TelegramError("sendMessage", "Rate limited.", retry_after=2)
        with patch.object(telegram_ops, "telegram_settings", return_value=("dummy", 42)), \
                patch.object(telegram_ops, "telegram_request", side_effect=[error, {"message_id": 5}]) as send, \
                patch.object(telegram_ops.time, "sleep") as wait:
            result = telegram_ops.send_telegram("😀" * 4000)
        self.assertTrue(result["ok"])
        self.assertEqual(send.call_count, 2)
        self.assertEqual(send.call_args.args[2], {"chat_id": 42, "text": "😀" * 2000})
        wait.assert_called_once_with(2)

    def test_does_not_repeat_an_unconfirmed_send(self):
        with patch.object(telegram_ops, "telegram_settings", return_value=("dummy", 42)), \
                patch.object(telegram_ops.requests, "post", side_effect=requests.ReadTimeout("test-secret")) as post:
            result = telegram_ops.send_telegram("hello")
        self.assertFalse(result["ok"])
        self.assertIn("timeout", result["error"])
        self.assertNotIn("test-secret", result["error"])
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
