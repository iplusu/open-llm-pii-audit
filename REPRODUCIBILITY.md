# Reproducibility

Two paths with different evidence status. Do not read one as covering the other.

## (A) Regenerate tables, figures and numbers from stored outputs - verified (scope below)
```
bash scripts/reproduce_from_outputs.sh        # generate_tables.py, generate_figures.py, verify_artifact.py --only 1,2,3,4
python3 scripts/verify_artifact.py            # all five categories; exits 2 on a CPU-only machine (category 5 BLOCKED)
```
Run from the repository root. Needs Python >= 3.10, and numpy + matplotlib for the figures step (a missing module is reported BLOCKED, never PASS).

| category | what it checks | CPU-only result |
|---|---|---|
| 1 file integrity | every file matches `provenance/SHA256SUMS` and no file is missing from it (a stale manifest fails); 38,832 records line-aligned between `model_outputs.jsonl` and `execution_metadata.jsonl`, per-suite counts, the 20,400 canonical records, 300 + 150 probes, 850 + 450 ground-truth spans, 128 calibration rows | **PASS** |
| 2 stored-output metric reproduction | the unit tests; probe files regenerate byte-identically; the 192 holdout cases regenerate to the recorded hashes; `generate_tables.py` is re-run into a fresh directory and every table in `results/tables/` and `results/processed/` must match the committed one byte for byte (including 13,824 holdout views re-scored fresh against the stored scores); figures render without label overlap | **PASS** |
| 3 claim consistency | the numbers listed in `CLAIMS_TO_EVIDENCE.md` are recomputed from the fresh tables and compared (the manuscript's wording is not in this repository and is not checked) | **PASS** |
| 4 stored-output and timestamp provenance signal | original C7/C8 run: stored C8 equals `filter(C7)` for only 410 of 1,800 pairs and every C8 stamp is more than 24 h after its C7 twin. This is a property of the stored records, **not** a statement about how they were produced (`known_limitations.md` L1) | **PASS** |
| 5 model re-generation | a 3-probe C0 smoke run through `src/run_experiments.py` | **BLOCKED** without a GPU, vLLM, the weights (`DEPENDENCY_MISSING`); never PASS here |

Exit codes: 0 every requested category passes; 1 any failure; 2 no failure but something BLOCKED or skipped. `--only` / `--skip-slow` runs are labelled PARTIAL and never reported as full verification. Steps that must produce a file delete it first, so output left by an earlier run is never evidence. Logs: `recompute_out/verification_<category>.log`, `recompute_out/verification_summary.txt`. The verifier's own failure propagation is tested in `tests/test_leakage_metrics.py` (class `VerifierExitCodeTests`).

What a pass means: the stored model outputs, run through the frozen matchers and aggregation in `src/`, give the stored tables and the audited claim numbers. The main table's headline columns (n, leakage rate and interval, refusal rate and interval) are exact; its embedding-based partial columns are not recomputed (`known_limitations.md` L10).

Where each table comes from: `scripts/generate_tables.py` reads `results/raw/model_outputs.jsonl` (+ `execution_metadata.jsonl` for holdout reconciliation and `data/` ground truth) and calls `src/aggregate_results.py` (leakage and exposure scoring), `src/bootstrap_analysis.py` (intervals, B = 2000, seed 42), `src/measure_utility.py` (utility, holdout scoring) and `src/filter_output.py` (C8 views). The table-to-claim map is `CLAIMS_TO_EVIDENCE.md`.

## (B) Re-run inference from the models - not validated
Not re-run in this release. Model weights, the AWQ INT4 checkpoints and the serving environment are not included. What exists:

- inputs: `configs/models.yaml` + `provenance/model_revisions.json` (repo and pinned revision), `configs/generation.yaml` (vLLM 0.30.0, temperature 0.2, max_tokens 256, request seed 42, Qwen3 thinking disabled), `configs/mitigation_conditions.yaml`, `data/`, `prompts/`;
- code: `src/run_experiments.py` (C0-C6), `src/run_multiturn.py` (C7/C8 with the live detector, sequential multi-turn, and the holdout case executor), launched by `scripts/run_full_experiments.sh` (never executed in this release, including the C7/C8 and extended-suite phases);
- converting a fresh run to the stored layout: `python3 src/aggregate_results.py --consolidate new_inference_out/<tree> --suite <suite> --out <dir>` (round-trip tested on a stored record).

Do not expect byte-identical outputs: sampling at temperature 0.2 is not bit-reproducible across hardware or vLLM versions (`known_limitations.md` L12). Known gaps that prevent an exact re-run: the INT4 path (L7), the original C7/C8 execution link (L1), partial holdout serving provenance (L4).
