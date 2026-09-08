"""Create and summarize matched, manually evaluated H3 voice trials. No network."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

VARIANTS = {
    "latent_separate": ("latent_only", "separate", "off"),
    "native_separate": ("native", "separate", "off"),
    "native_paired": ("native", "paired", "off"),
    "pair_bias": ("native", "paired", "pair_only"),
    "scheduled_bias": ("native", "paired", "scheduled"),
}
METRICS = ("dialogue_correct", "source_repeated", "intelligible", "lip_sync_correct", "emotion_correct")
FIELDS = ("case", "speakers", "seed", "variant", "reference_presentation", "av_layout", "binding_mode",
          "pair_bias", "voice_bias", "boundary_fade_seconds", "status", "case_fingerprint", "run_settings",
          "correct_voices", *METRICS, "render_seconds", "output_file", "notes")


def trial_rows(seeds=(101, 202, 303), cast_sizes=(1, 2, 3)):
    for speakers in cast_sizes:
        for seed in seeds:
            for variant, (presentation, layout, mode) in VARIANTS.items():
                row = dict.fromkeys(FIELDS, "")
                row.update(case=f"cast_{speakers}", speakers=speakers, seed=seed, variant=variant,
                           reference_presentation=presentation, av_layout=layout, binding_mode=mode,
                           pair_bias="0.5", voice_bias="0.5", boundary_fade_seconds="0.1", status="pending")
                yield row


def summarize(rows):
    groups = defaultdict(dict)
    for line, row in enumerate(rows, 2):
        try:
            speakers, seed = int(row["speakers"]), int(row["seed"])
            if not 1 <= speakers <= 3 or seed < 0 or not row["case"].strip():
                raise ValueError("invalid case/speakers/seed")
            variant = row["variant"]
            if variant not in VARIANTS:
                raise ValueError("unknown variant")
            if tuple(row[k] for k in ("reference_presentation", "av_layout", "binding_mode")) != VARIANTS[variant]:
                raise ValueError("variant switches do not match the declared experiment")
            if row["status"] not in ("pending", "complete", "failed"):
                raise ValueError("status must be pending, complete, or failed")
            group = groups[(row["case"], seed)]
            if variant in group:
                raise ValueError("duplicate case/seed/variant")
            group[variant] = row
            if row["status"] == "pending":
                continue
            if not re.fullmatch(r"[a-f0-9]{64}", row["case_fingerprint"]):
                raise ValueError("copy the full case_fingerprint from the conditioner")
            if not row["run_settings"].strip():
                raise ValueError("record an identifier for your fixed model/sampler settings")
            for key in ("pair_bias", "voice_bias", "boundary_fade_seconds"):
                value = float(row[key])
                maximum = 1 if key == "boundary_fade_seconds" else 3
                if not math.isfinite(value) or not 0 <= value <= maximum:
                    raise ValueError("invalid bias or fade")
            if row["status"] == "failed":
                if not row["notes"].strip():
                    raise ValueError("failed runs need an error/reason in notes")
                continue
            count = int(row["correct_voices"])
            if not 0 <= count <= speakers:
                raise ValueError("correct_voices must be between 0 and speakers")
            if any(row[key] not in ("yes", "no") for key in METRICS):
                raise ValueError("quality fields must be yes or no")
            seconds = float(row["render_seconds"])
            if not math.isfinite(seconds) or seconds <= 0 or not row["output_file"].strip():
                raise ValueError("complete runs need positive render_seconds and output_file")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"CSV line {line}: {exc}") from exc
    matched, incomplete = [], []
    for key, group in groups.items():
        finished = [r for r in group.values() if r["status"] != "pending"]
        for field in ("speakers", "case_fingerprint", "run_settings", "pair_bias", "voice_bias", "boundary_fade_seconds"):
            if len({str(r[field]) for r in finished}) > 1:
                raise ValueError(f"Unmatched {field} within {key}; compare the same sources/prompt/settings.")
        if set(group) != set(VARIANTS) or len(finished) != len(VARIANTS):
            incomplete.append({"case": key[0], "seed": key[1], "finished_variants": len(finished)})
        else:
            matched.append(group)
    results = []
    for variant in VARIANTS:
        trials = [group[variant] for group in matched]
        speakers = sum(int(r["speakers"]) for r in trials)
        completed = [r for r in trials if r["status"] == "complete"]
        correct = sum(int(r["correct_voices"]) for r in completed)
        clean = sum(int(r["correct_voices"]) == int(r["speakers"]) and
                    all(r[k] == ("no" if k == "source_repeated" else "yes") for k in METRICS)
                    for r in completed)
        results.append({"variant": variant, "trials": len(trials), "failed": len(trials) - len(completed),
                        "correct_voices": correct, "speaking_characters": speakers,
                        "correct_voice_rate": correct / speakers if speakers else None,
                        "all_quality_checks_passed": clean,
                        "mean_successful_render_seconds": (sum(float(r["render_seconds"]) for r in completed) /
                                                           len(completed)) if completed else None})
    return {"matched_case_seed_groups": len(matched), "incomplete_groups_excluded": incomplete,
            "results": results,
            "evaluation": "Manual listening/visual ratings. Failed renders count as unsuccessful. "
                          "run_settings is user-reported; no model files or videos are automatically verified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Write a NEW blank 45-run trial sheet")
    init.add_argument("csv", type=Path)
    report = sub.add_parser("summarize", help="Summarize only complete matched groups")
    report.add_argument("csv", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        with args.csv.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(trial_rows())
        print(f"Created {args.csv}; all results are pending.")
    else:
        try:
            with args.csv.open(newline="", encoding="utf-8-sig") as handle:
                result = summarize(list(csv.DictReader(handle)))
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
