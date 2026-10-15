"""Output-filter tests (C3/C5/C6/C8), including C8 == filter(C7) on all stored V8 C7/C8 pairs."""
import base64
import codecs
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import aggregate_results
import filter_output

ROOT = Path(__file__).resolve().parents[1]


class FilterTests(unittest.TestCase):
    def test_pattern_families_masked(self):
        filt, n = filter_output.apply_output_filter(
            "The SSN 987-04-1234 and phone (555) 014-2830, card 4000 0000 1111 2222, email avery@example.pii, "
            "addr 1234 Maple St, Springdale, OH 44807, name Avery Preston.")
        for token in ("[FILTERED_SSN]", "[FILTERED_PHONE]", "[FILTERED_CARD]", "[FILTERED_EMAIL]", "[ADDRESS]"):
            self.assertIn(token, filt)
        self.assertNotIn("987-04-1234", filt)
        self.assertNotIn("Avery Preston", filt)
        self.assertGreaterEqual(n, 5)

    def test_name_filter_drops_first_name_keeps_surname(self):
        filt, n = filter_output.apply_output_filter("Please call Avery Preston today.")
        self.assertEqual((filt, n), ("Please call Preston today.", 1))

    def test_stopword_pairs_untouched(self):
        text = "Thank You for the Order Summary."
        self.assertEqual(filter_output.apply_output_filter(text), (text, 0))

    def test_cascade_examines_adjacent_pair_after_a_fired_pair(self):
        filt, n = filter_output.apply_output_filter("Contact Pia Hartwell Quill Rourke now")
        self.assertNotIn("Pia", filt)
        self.assertNotIn("Hartwell", filt)
        self.assertGreaterEqual(n, 2)

    def test_clean_text_is_unchanged(self):
        text = "The report is ready for review."
        self.assertEqual(filter_output.apply_output_filter(text), (text, 0))

    def test_v8_c8_is_filter_of_c7_on_all_pairs(self):
        """V8 C8 records are a derived view: stored C8 output == apply_output_filter(stored C7 output) for every pair."""
        c7, c8 = {}, {}
        for r in aggregate_results.iter_records(ROOT / "results/raw", suites={"v8_main", "v8_extended"}):
            (c7 if r["condition"] == "C7" else c8)[(r["suite"], r["model"], r["probe_id"])] = r
        self.assertEqual((len(c7), len(c8)), (2700, 2700))
        bad = [k for k, r in c7.items()
               if [filter_output.apply_output_filter(o)[0] for o in r["outputs_all"]] != c8[k]["outputs_all"]]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
