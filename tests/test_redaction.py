"""Redaction tests: oracle redaction, ground-truth/probe consistency, byte-exact probe regeneration, authorized restoration baseline."""
import base64
import codecs
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import tempfile

import generate_probes
import measure_utility
import redact_input

ROOT = Path(__file__).resolve().parents[1]
SPANS = [
    {"type": "ssn", "text": "987-04-1234", "token": "[REDACTED_SSN]"},
    {"type": "phone", "text": "(555) 014-2830", "token": "[REDACTED_PHONE]"},
    {"type": "name", "text": "Avery Preston", "token": "[REDACTED_NAME]"},
    {"type": "address", "text": "1234 Maple St, Springdale, OH 44807", "token": "[REDACTED_ADDRESS]"},
]


class RedactionTests(unittest.TestCase):
    def test_oracle_redaction(self):
        text = "My SSN is 987-04-1234, call (555) 014-2830, I am Avery Preston, live at 1234 Maple St, Springdale, OH 44807."
        red = redact_input.redact_text(text, SPANS)
        for lit in ("987-04-1234", "2830", "Avery Preston", "Maple St"):
            self.assertNotIn(lit, red)
        self.assertEqual(red.count("[REDACTED_"), 4)
        self.assertIn("call", red)
        self.assertIn("live at", red)

    def test_redaction_map_matches_ground_truth(self):
        gt = {}
        for line in (ROOT / "data/pii_ground_truth.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["suite"] == "main":
                gt.setdefault(r["probe_id"], {})[r["token"]] = r["text"]
        rm = {json.loads(l)["probe_id"]: json.loads(l)["map"] for l in (ROOT / "data/redaction_map.jsonl").read_text().splitlines()}
        self.assertEqual(len(gt), 300)
        self.assertEqual(sum(len(v) for v in gt.values()), sum(len(v) for v in rm.values()))
        for pid, m in rm.items():
            for token, text in m.items():
                self.assertEqual(gt[pid][token], text)

    def test_every_span_occurs_in_its_probe_and_ground_truth_has_850_main_spans(self):
        probes = {json.loads(l)["probe_id"]: json.loads(l) for l in (ROOT / "data/probes_main.jsonl").read_text().splitlines()}
        spans = [json.loads(l) for l in (ROOT / "data/pii_ground_truth.jsonl").read_text().splitlines()]
        main = [s for s in spans if s["suite"] == "main"]
        self.assertEqual(len(main), 850)
        self.assertEqual(len(spans) - len(main), 450)
        for s in main:
            self.assertIn(s["text"], "\n".join(t["content"] for t in probes[s["probe_id"]]["turns"]))

    def test_probe_files_regenerate_byte_identically(self):
        with tempfile.TemporaryDirectory() as d:
            generate_probes.write_data_files(d)
            for name in ("probes_main.jsonl", "probes_extended.jsonl", "pii_ground_truth.jsonl", "redaction_map.jsonl"):
                self.assertEqual((Path(d) / name).read_bytes(), (ROOT / "data" / name).read_bytes(), name)

    def test_synthetic_only_markers(self):
        for line in (ROOT / "data/pii_ground_truth.jsonl").read_text().splitlines():
            s = json.loads(line)
            if s["type"] == "ssn":
                self.assertTrue(s["text"].startswith("987-"))
            if s["type"] == "phone":
                self.assertTrue(s["text"].startswith("(555) 01"))
            if s["type"] == "card":
                self.assertTrue(s["text"].startswith("4000 0000 "))


class RestorationTests(unittest.TestCase):
    def test_unique_scoped_authorized_restoration(self):
        r = measure_utility
        case = {'case_id': 'DEV001', 'entities': [{'entity_id': 'E01', 'type': 'name', 'text': 'Zelvía Quast'},
                  {'entity_id': 'E02', 'type': 'name', 'text': 'Torqen Quast'}],
                'annotations': [{'turn': 0, 'start': 0, 'end': 12, 'surface': 'Zelvía Quast', 'entity_id': 'E01'},
                                {'turn': 0, 'start': 15, 'end': 27, 'surface': 'Torqen Quast', 'entity_id': 'E02'}],
                'turns': [{'role': 'user', 'content': 'Zelvía Quast / Torqen Quast'}],
                'authorized_fields': {'recipient': 'E01'}}
        texts, mapping = r.prepare(case)
        self.assertEqual(len(mapping), 2)
        self.assertNotIn('Zelvía Quast', texts[0]['content'])
        token = next(k for k, v in mapping.items() if v['entity_id'] == 'E01')
        other = next(k for k, v in mapping.items() if v['entity_id'] == 'E02')
        self.assertEqual(r.restore_fields({'recipient': token}, mapping, case['authorized_fields'], authorized=True), {'recipient': 'Zelvía Quast'})
        with self.assertRaises(PermissionError):
            r.restore_fields({'recipient': token}, mapping, case['authorized_fields'], authorized=False)
        with self.assertRaises(ValueError):
            r.restore_fields({'recipient': other}, mapping, case['authorized_fields'], authorized=True)
        with self.assertRaises(ValueError):
            r.restore_fields({'recipient': token, 'public_note': token}, mapping, case['authorized_fields'], authorized=True)
        self.assertEqual(r.restore_fields({'selected': 'record_a'}, mapping, {}, authorized=True), {'selected': 'record_a'})


class RunnerInputTests(unittest.TestCase):
    def test_runner_builds_exactly_the_stored_prompts_for_all_15000_main_records(self):
        """src/run_experiments.build_messages (oracle redaction + refusal system prompt) reproduces every stored main-suite prompt
        and redacted text, which also pins prompts/refusal_system.txt to what was actually sent."""
        import aggregate_results
        import run_experiments
        probes = {p["probe_id"]: p for p in run_experiments.load_probes(ROOT / "data")}
        n = 0
        for rec in aggregate_results.iter_records(ROOT / "results/raw", suites={"main"}):
            msgs, red = run_experiments.build_messages(probes[rec["probe_id"]], rec["condition"])
            self.assertEqual(msgs, rec["prompt"], rec["record_id"])
            if "redacted_prompt_text" in rec:
                self.assertEqual(red, rec["redacted_prompt_text"], rec["record_id"])
            n += 1
        self.assertEqual(n, 15000)


if __name__ == "__main__":
    unittest.main()
