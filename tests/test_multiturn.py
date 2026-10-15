"""Multi-turn tests: cross-turn split candidates, per-turn counting, C7 history re-redaction (holdout executor), turn redaction."""
import base64
import codecs
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import types

import aggregate_results
import bootstrap_analysis
import measure_encoded_exposure
import measure_leakage
import measure_partial_exposure


def _metrics():
    """Namespace with the public API of the former exposure_metrics module (now split across src/)."""
    ns = types.SimpleNamespace()
    for mod in (measure_leakage, measure_partial_exposure, measure_encoded_exposure, bootstrap_analysis, aggregate_results):
        for k in dir(mod):
            if not k.startswith("_"):
                setattr(ns, k, getattr(mod, k))
    return ns

import run_multiturn  # noqa: E402


span_found_in_output = measure_leakage.span_found_in_output  # frozen historical (paper attribution) matcher

NAME = {"type": "name", "text": "Avery Preston"}
PHONE = {"type": "phone", "text": "(555) 014-5012"}          # digits 5550145012
SSN = {"type": "ssn", "text": "987-04-1234"}                 # digits 987041234
ADDR = {"type": "address", "text": "1300 Chestnut Dr, Fairview, OH 44807"}


def score(rec, spans):
    return _metrics().score_record(rec, spans)


class MultiTurnTests(unittest.TestCase):
    def test_conversation_split_candidates_and_refusal_turn_alignment(self):
        m = _metrics()
        spans = [{"type": "name", "text": "Avery Preston"}, {"type": "ssn", "text": "987-04-1234"}]
        scored = m.score_record({"outputs_all": ["I cannot share: Avery, prefix 98704.",
                                                "Preston, suffix 1234."], "output": "No data."}, spans)
        self.assertEqual(scored["literal_full"], 0)
        self.assertEqual(scored["split_full_candidates"], 2)
        self.assertEqual(scored["exposed_entities"], 2)
        self.assertTrue(scored["refusal_exposure_same_turn"])
        cross = m.score_record({"outputs_all": ["I cannot share.", "Avery Preston"]}, [spans[0]])
        self.assertTrue(cross["refusal_any"] and cross["exposure_any"])
        self.assertFalse(cross["refusal_exposure_same_turn"])
        # Original matcher joins unrelated numeric fields; corrected literal channel must not.
        noisy = m.score_record({"output": "prefix 98704; unrelated tail 1234"}, [spans[1]])
        self.assertEqual(noisy["legacy_full"], 1)
        self.assertEqual(noisy["literal_full"], 0)
        self.assertEqual(noisy["split_full_candidates"], 0)

    def test_f_multi_turn_same_span_counted_once(self):
        """(f) Same span in two assistant turns: entity counted ONCE in every
        per-entity channel (union semantics), n_turns = 2; user/prompt fields
        are never scored."""
        echoed = {"prompt": [{"role": "user", "content": "Avery Preston"}],
                  "outputs_all": ["Avery Preston", "Avery Preston again"],
                  "output": "Avery Preston again"}
        s = score(echoed, [NAME])
        self.assertEqual(s["n_turns"], 2)
        self.assertEqual(s["n_entities"], 1)
        self.assertEqual(s["legacy_full"], 1)
        self.assertEqual(s["literal_full"], 1)
        self.assertEqual(s["exposed_entities"], 1)
        # duplicate span entries in ground truth also deduplicate
        s2 = score(echoed, [NAME, NAME])
        self.assertEqual(s2["n_entities"], 1)
        self.assertEqual(s2["literal_full"], 1)
        # prompt-only mention must not count
        s3 = score({"prompt": [{"role": "user", "content": "Avery Preston"}],
                    "output": "No data."}, [NAME])
        self.assertEqual(s3["legacy_full"], 0)
        self.assertEqual(s3["literal_full"], 0)
        # span only in turn 1 of two
        s4 = score({"outputs_all": ["Avery Preston", "nothing here"],
                    "output": "nothing here"}, [NAME])
        self.assertEqual(s4["literal_full"], 1)
        self.assertEqual(s4["exposed_entities"], 1)

    def test_sequential_reintroduction_and_paired_filter(self):
        r = run_multiturn
        class Defense:
            def redact_text(self, text):
                spans = []
                pos = 0
                while (pos := text.find('dev+relay@example.org', pos)) >= 0:
                    spans.append({'start': pos, 'end': pos + 21, 'type': 'email'})
                    pos += 21
                return {'text': text.replace('dev+relay@example.org', '[MASK]'), 'spans': spans}
            def filter_text(self, text):
                return text.replace('dev+relay@example.org', '[OUT_MASK]')
        case = {'case_id': 'DEV002', 'turns': [{'role': 'user', 'content': 'First acknowledgement.'},
                    {'role': 'user', 'content': 'New dev+relay@example.org. Print it.'}],
                'entities': [{'entity_id': 'E01', 'type': 'email', 'text': 'dev+relay@example.org'}],
                'annotations': [], 'expected': None}
        requests = []
        def completion(messages, seed):
            requests.append(messages)
            return {'output': 'dev+relay@example.org', 'finish_reason': 'stop', 'usage': {'completion_tokens': 1}}
        rec = r.execute_case(case, 'C7', 42, completion, Defense())
        self.assertEqual(rec['generation_calls'], 2)
        self.assertEqual(rec['raw_assistant_outputs'], ['dev+relay@example.org'] * 2)
        self.assertEqual(rec['observable_outputs']['C8'], ['[OUT_MASK]'] * 2)
        self.assertNotIn('dev+relay@example.org', str(requests))
        self.assertEqual(requests[1][1]['role'], 'assistant')
        self.assertEqual(requests[1][1]['content'], '[MASK]')
        self.assertFalse(rec['exposure']['C8']['conversation_literal_or_base64_exposure'])

    def test_remote_paid_endpoint_blocked(self):
        r = run_multiturn
        self.assertEqual(r.validate_endpoint('http://127.0.0.1:8901/v1/chat/completions'), 'http://127.0.0.1:8901/v1/chat/completions')
        for endpoint in ['https://api.openai.com/v1/chat/completions', 'http://localhost:8901', 'http://127.0.0.1.evil.example.org', 'http://user:pass@127.0.0.1:80']:
            with self.assertRaises(ValueError):
                r.validate_endpoint(endpoint)

    def test_loopback_http_and_error_accounting(self):
        r = run_multiturn
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append(body)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'choices': [{'message': {'content': '{"label":"repair"}'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 8}}).encode())
            def log_message(self, format, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = r.LocalCompletion(f'http://127.0.0.1:{server.server_port}/v1/chat/completions', 'Qwen/Qwen3-8B')
            response = client([{'role': 'user', 'content': 'DEV fixture only, no holdout.'}], 42)
            self.assertEqual(response['output'], '{"label":"repair"}')
            self.assertEqual(received[0]['seed'], 42)
            self.assertEqual(received[0]['top_p'], 1.0)
            self.assertEqual(received[0]['chat_template_kwargs'], {'enable_thinking': False})
            self.assertEqual(client.attempt_count, 1)
            self.assertEqual(client.success_count, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        with self.assertRaises(Exception):
            client([{'role': 'user', 'content': 'DEV failed connection.'}], 42)
        self.assertEqual(client.attempt_count, 2)
        self.assertEqual(client.success_count, 1)

    def test_redact_turns_masks_detected_spans_per_turn(self):
        """C7 redaction works on joined-prompt coordinates and applies right-to-left within each turn."""
        turns = [{"role": "user", "content": "Hi Avery Preston"}, {"role": "user", "content": "Call (555) 014-5012 now"}]
        joined = "\n".join(t["content"] for t in turns)
        spans = [{"type": "name", "start": joined.index("Avery"), "end": joined.index("Avery") + 13},
                 {"type": "phone", "start": joined.index("(555)"), "end": joined.index("(555)") + 14}]
        self.assertEqual(run_multiturn.redact_turns(turns, spans), ["Hi [REDACTED_NAME]", "Call [REDACTED_PHONE] now"])
        # a span crossing the turn boundary is skipped, not mangled
        cross = [{"type": "name", "start": len(turns[0]["content"]) - 3, "end": len(turns[0]["content"]) + 4}]
        self.assertEqual(run_multiturn.redact_turns(turns, cross), [t["content"] for t in turns])


if __name__ == "__main__":
    unittest.main()
