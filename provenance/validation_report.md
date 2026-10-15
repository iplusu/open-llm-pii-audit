# Validation report

Output of running `scripts/verify_artifact.py` against this repository, 2026-10-15 (UTC 20:5x). Written after the run and therefore the one file exempt from `provenance/SHA256SUMS` (the verifier's completeness check exempts exactly `SHA256SUMS` and this report).

## Method
Clean-room style: the final tree (all files, `SHA256SUMS` already built) was copied to a fresh temporary directory and only the documented commands were run, with `python3` = Python 3.13.16 in a venv containing numpy and matplotlib 3.11.2 (a CPU-only host: no GPU, no vLLM, no model weights, no network needed). Nothing was run from the working tree itself.

## Run 1 - `python3 scripts/verify_artifact.py` (all five categories)
Wall time about 1.5 minutes. **Exit code 2** (no failure; one category BLOCKED).

| category | result | steps |
|---|---|---|
| 1 File integrity | **PASS** | 1.a all 68 files match `SHA256SUMS`; 1.b no file on disk is missing from it (68 on disk, 68 listed); 1.c 38,832 records line-aligned and unique, per-suite counts 15,000 / 3,600 / 1,800 / 3,600 / 3,600 / 864 / 10,368, 20,400 canonical records, every canonical cell has 300 (150 for the extended suite) records, 300 + 150 probes, 850 + 450 ground-truth spans, 128 calibration rows |
| 2 Stored-output metric reproduction | **PASS** (7/7) | 2.a 53 unit tests OK; 2.b/2.c probe, ground-truth and redaction-map files regenerate byte-identically; 2.d 192 holdout cases regenerate to the recorded sha256; 2.e `generate_tables.py` re-run into a fresh directory; 2.f every table in `results/tables` and `results/processed` equals the committed one byte for byte (includes 13,824 holdout views re-scored fresh, 0 mismatches against the stored scores); 2.g three figures render with 0 label overlaps |
| 3 Claim consistency | **PASS** | 21 claims recomputed from the fresh tables and found in `CLAIMS_TO_EVIDENCE.md`: B1c, B2b, B2c, B2d, B3-ALL, B3p, B3a, B3n, B3b, B3u, E1, E2, V4-5, V4-6, V4-10, V4-11, V4-12, V4-16, MAIN-C2, QUANT-C5, MAIN-C0. Numbers only; the paper text is not in this repository |
| 4 Provenance signal (original C7/C8) | **PASS** | 1,800 pairs; `filter(C7)` equals stored C8 for 410 (Llama 83, Mistral 75, Olmo 44, Qwen 112, Gemma 81, phi-4 15); every C8 stamp is 36.3-36.8 h after its C7 twin. This is a property of the stored records, not of how they were produced |
| 5 Model re-generation | **BLOCKED** | `DEPENDENCY_MISSING: nvidia-smi, python module vllm, python module openai`. Not run; not claimed |

## Run 2 - `bash scripts/reproduce_from_outputs.sh`
**Exit code 0.** `generate_tables.py` rewrote `results/tables` and `results/processed` in place; `diff -r` against the committed directories showed no difference. `verify_artifact.py --only 1,2,3,4` then passed (labelled PARTIAL by design, since category 5 is not requested) and the figures were written to `recompute_out/figures`.

## What this does and does not establish
- Established: the stored model outputs, run through the frozen matchers and aggregation in `src/`, give the committed tables and the 21 claim numbers above; the shipped data files and holdout cases are exactly what the seeded generators produce; the stored prompts of all 15,000 main-suite records are exactly what `src/run_experiments.py` builds from `data/` and `prompts/refusal_system.txt` (`tests/test_redaction.py`); V8 C8 equals the output filter applied to C7 on all 2,700 pairs (`tests/test_output_filter.py`).
- Not established: that any model would reproduce these outputs (category 5 was not run, and cannot run here); how the original C7/C8 records were produced (`known_limitations.md` L1); anything about the paper's wording; detector recall, paired-contrast intervals and the embedding partial channel (L5, L10).

## Issues found and fixed during validation
1. The first manifest build omitted `provenance/artifact_manifest.json` from `SHA256SUMS`; the completeness check (1.b) failed on it, the manifest builder was fixed and rebuilt last.
2. `reproduce_from_outputs.sh` first rewrote the committed figure PDFs in place; the PDF bytes differ per run (creation time stamp) and per matplotlib version, which broke the `SHA256SUMS` match (1.a). Figure PDFs are now written without a time stamp, and the reproduce script writes its figures to `recompute_out/figures` instead of over the committed ones.

## Cross-checks made while building (not re-runnable here)
The consolidated tables were compared with the archival analysis files they replace (archival files are not shipped, so the verifier cannot repeat this): channel, holdout exposure, holdout utility, T7 breakdown, T7 scheme survival, occurrence coverage (summary and detail), input-intact and utility tables are **byte-identical**; the headline columns of the main table (n, leakage rate and interval, refusal rate and interval; 50 cells) are identical at the printed precision, and all 74 stored exposure rows match on `n_probes`, `legacy_full`, `literal_full` and `exposure_probes`. The `partial_*` and per-probe-type columns differ from the archival table in 7 of 50 cells because they there include the embedding channel (`known_limitations.md` L10).
