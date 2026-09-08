import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("voice_trials", Path(__file__).resolve().parents[1] / "tools/voice_trials.py")
trials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trials)


def completed():
    rows = list(trials.trial_rows(seeds=[101], cast_sizes=[2]))
    for row in rows:
        row.update(status="complete", case_fingerprint="a" * 64, run_settings="model-sampler-v1",
                   correct_voices="2", render_seconds="123.5", output_file="test.mp4")
        row.update({key: "no" if key == "source_repeated" else "yes" for key in trials.METRICS})
    return rows


class TrialTests(unittest.TestCase):
    def test_pending_results_cannot_become_successes(self):
        report = trials.summarize(list(trials.trial_rows()))
        self.assertEqual(report["matched_case_seed_groups"], 0)
        self.assertEqual(len(report["incomplete_groups_excluded"]), 9)
        self.assertIsNone(report["results"][0]["correct_voice_rate"])

    def test_failures_count_and_incomplete_groups_are_explicit(self):
        rows = completed()
        rows[0].update(status="failed", notes="sampler error")
        report = trials.summarize(rows)
        self.assertEqual(report["results"][0]["correct_voice_rate"], 0)
        self.assertEqual(report["results"][1]["all_quality_checks_passed"], 1)
        report = trials.summarize(rows[:-1])
        self.assertEqual(report["matched_case_seed_groups"], 0)
        self.assertEqual(len(report["incomplete_groups_excluded"]), 1)

    def test_mismatched_and_duplicate_runs_are_rejected(self):
        for key, value in (("case_fingerprint", "b" * 64), ("run_settings", "changed"), ("pair_bias", "1.0")):
            rows = completed()
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Unmatched"):
                trials.summarize(rows)
        rows = completed()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            trials.summarize(rows + [rows[0]])

    def test_incorrect_words_and_source_copy_fail_quality_gate(self):
        rows = completed()
        rows[0]["source_repeated"] = "yes"
        rows[1]["dialogue_correct"] = "no"
        report = trials.summarize(rows)
        self.assertEqual(report["results"][0]["correct_voice_rate"], 1)
        self.assertEqual(report["results"][0]["all_quality_checks_passed"], 0)
        self.assertEqual(report["results"][1]["all_quality_checks_passed"], 0)
