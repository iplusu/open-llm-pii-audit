# Claims to evidence

Maps the paper's checkable numeric claims to the records that produce them and to the check that recomputes them. It covers the claims that were audited and re-derived during the evidence review, **not every number in the paper**; the paper text is not in this repository, so sentence-level wording is not checked here. Paths are relative to the repository root; `results/raw/model_outputs.jsonl` is filtered by its `suite` field (see `DATA_DICTIONARY.md`).

How to read the **status** column (all are produced by `python3 scripts/verify_artifact.py`, see `REPRODUCIBILITY.md`):

- **asserted** - category 3 recomputes the value from the tables regenerated in the same run and compares it with the number below; the claim id must also appear in this file.
- **test** - a unit test in `tests/` covers it (category 2.a).
- **table** - the value is a cell of a table that category 2.e/2.f regenerates from the raw records and compares byte-for-byte with the committed table. This repository's own tables were additionally compared, during the rebuild, with the archival analysis files they replace: identical for the channel, holdout (exposure and utility), T7, occurrence-coverage, input-intact and utility tables; identical in the headline columns of the main table (see `provenance/validation_report.md`).
- **script** - category 4 (provenance signal).
- **not checkable here** - needs the manuscript text or inputs this repository does not contain.

| id | claim | paper location | records (`suite`) | metric / denominator | result file | status |
|---|---|---|---|---|---|---|
| `B1a` | C8 any-channel exposure probes per model (122/195/152/156/159/120 for Llama, Mistral, Olmo, Qwen, Gemma, phi-4) | Sec 4.6 "C8 residual exposure is partial" | `v8_main` C7 (C8 = filter(C7)) | `any_probes` / 300 probes per model | `results/tables/channels_v8.csv` (run=main, cond=C8) | table |
| `B1b` | C7 any-channel exposure probes per model (127/223/154/157/163/121) | Sec 4.6 | `v8_main` C7 | `any_probes` / 300 | `results/tables/channels_v8.csv` (run=main, cond=C7) | table |
| `B1c` | C8 full-string probes: only Qwen3-8B (1) | Sec 4.6 | `v8_main` | `full_probes` / 300 | `results/tables/channels_v8.csv` | asserted |
| `B2a` | Extended C7 full-string spans per model, 8-34 summed over T7-T9 (paper matcher) | Sec 4.7 / Table 11 A | `v8_extended` C7 | `paper_matcher_spans` / 450 spans per model | `results/tables/channels_v8.csv` (run=ext) | table |
| `B2b` | Extended C8 full-string spans total 6, all Qwen3-8B T7 | Sec 4.7 / Table 11 A | `v8_extended` | `paper_matcher_spans` / 450 | `results/tables/channels_v8.csv` | asserted |
| `B2c` | T7 encoded-string repeat at C8: 23-32 of 50 for five models, Gemma 5 | Sec 4.7 | `v8_extended` | `input_echo_probes` / 50 | `results/tables/channels_v8.csv` | asserted |
| `B2d` | T8/T9 any-channel probes unchanged C7 -> C8 | Sec 4.7 / Table 11 B | `v8_extended` | `any_probes` / 50 | `results/tables/channels_v8.csv` | asserted |
| `B3-ALL` | Holdout, all scopes: C0 559/659, C2_unique 3/3, C7 369/442, C8 327/400 (literal / literal+base64+join) of 3,456 observations per view | Sec 4.8 | `holdout` | literal; literal or base64 or split / 3,456 | `results/tables/holdout_exposure.csv` | asserted |
| `B3p` | Holdout prohibited-exposure scope: C0/C7/C8 literal 224/157/139 of 1,836 | Sec 4.8 / Table 13 | `holdout` | literal / 1,836 per view | `results/tables/holdout_exposure.csv` | asserted |
| `B3a` | Holdout authorized-record scope: C0/C7/C8 literal 199/145/127 of 216 | Sec 4.8 | `holdout` | literal / 216 per view | `results/tables/holdout_exposure.csv` | asserted |
| `B3n` | Holdout no-PII-requested scope: C0/C2/C7/C8 literal 136/3/67/61 of 1,404 | Sec 4.8 | `holdout` | literal / 1,404 per view | `results/tables/holdout_exposure.csv` | asserted |
| `B3u` | Holdout utility macro-mean strict success C0/C2/C7/C8 = .367/.410/.387/.387 | Sec 4.8 | `holdout` | `strict_task_success` / 24 model-family cells | `results/tables/holdout_utility_by_family.csv` | asserted |
| `B3b` | Holdout C8-C7 prohibited-scope cluster bootstrap -1.0 pp [-4.2, 0.0] over 14 template groups; 13,824 views re-scored fresh with 0 mismatches against the stored scores | Sec 4.8 | `holdout` | template-group bootstrap, B=2000, seed 42 | `results/tables/holdout_exposure.csv`, `results/processed/confidence_intervals.csv` | asserted (re-scoring), table (interval) |
| `A1` | Original run: `filter(C7)` equals stored C8 for only 410 of 1,800 pairs; every C8 stamp is more than 24 h after its C7 twin (separate generation; cause of the 410 not determined) | Sec 3 (C7/C8 methods) | `c7c8_original` | byte identity / 1,800 pairs | `results/raw` (checked directly) | script |
| `A2` | V8 C8 is derived from the same C7 generation: identical on all 2,700 pairs | Sec 3 | `v8_main`, `v8_extended` | identity / 2,700 pairs | `tests/test_output_filter.py::test_v8_c8_is_filter_of_c7_on_all_pairs` | test |
| `E1` | C7 leaked spans: 738 (original run, primary) vs 730 (V8, secondary) | Sec 4.6 | `c7c8_original`, `v8_main` | `legacy_full` summed over models / 850 spans per model | `results/processed/aggregate_metrics.csv` | asserted |
| `E2` | 108 of 1,800 original C7 prompts carry a damaged placeholder (18 per model); 0 in V8 | Sec 4.6 | `c7c8_original`, `v8_main` prompts | damaged prompts / 1,800 | `results/raw/model_outputs.jsonl` (checked directly) | asserted |
| `V4-1` | T7 C8 any-channel probes per model | Sec 4.7, T7 channel table | `v8_extended` T7 | `any_probes` / 50 | `results/tables/t7_channel_breakdown.csv` | table |
| `V4-2` | T7 C8 encoded probes per model equal the C8 column of the extended table | Sec 4.6 | `v8_extended` T7 | `encoded_probes` / 50 | `results/tables/t7_channel_breakdown.csv` | table |
| `V4-3` | T7 C7 encoded probes per model equal the C7 column of the extended table | Sec 4.6 | `v8_extended` T7 | `encoded_probes` / 50 | `results/tables/t7_channel_breakdown.csv` | table |
| `V4-4` | T7 C8 input-echo probes equal encoded probes; 0 unchanged-echo probes | Sec 4.7 | `v8_extended` T7 | `input_echo_probes`, `unchanged_echo_probes` / 50 | `results/tables/t7_channel_breakdown.csv` | table |
| `V4-5` | T7 C8 partial-only probes 0/0/0/1/22/1 (Llama, Mistral, Olmo, Qwen, Gemma, phi-4); Gemma 27 = 5 encoded + 22 partial-only | Sec 4.7 | `v8_extended` T7 | `partial_only_probes` / 50 | `results/tables/t7_channel_breakdown.csv` | asserted |
| `V4-6` | T7 C8 literal-full 0/0/0/5/0/0; Qwen's literal-full probes are a subset of its encoded probes | Sec 4.7 | `v8_extended` T7 | `literal_full_probes`, `full_not_encoded` / 50 | `results/tables/t7_channel_breakdown.csv` | asserted |
| `V4-7` | Any-fragment probes at C8 for Gemma 23 versus partial-only 22 | Sec 4.7 | `v8_extended` T7 | `any_fragment_probes` / 50 | `results/tables/t7_channel_breakdown.csv` | table |
| `V4-8` | C7 -> C8 encoded-form entity counts: ROT13 names 13 -> 0; base64 name/phone/ssn 54/57/66 unchanged; leet names 74 unchanged | Sec 4.6 | `v8_extended` T7 | entities with a predefined encoded form in the output / 150 per model | `results/tables/t7_encoded_scheme_survival.csv` | table |
| `V4-9` | Main campaign C7 full-entity probes 16-87 per model; no main-campaign encoded-form probe at C7/C8 | Sec 4.6 | `v8_main` | `full_probes`, `encoded_probes` / 300 | `results/tables/channels_v8.csv` | table |
| `V4-10` | Main benchmark: 23 multi-occurrence entities per model (18 masked throughout, 3 mixed, 2 intact throughout), identical for the original and V8 runs | Sec 4.6, Sec 5 limits | stored C7 prompts, `data/probes_main.jsonl` | exact-literal survival per occurrence / 850 spans per model | `results/tables/occurrence_coverage.csv` | asserted (V8 run), table (original run) |
| `V4-11` | Extended suite: 82 multi-occurrence entities per model (71 masked throughout, 9 mixed, 2 intact throughout) | Sec 4.6 | `v8_extended` C7 prompts | same / 450 spans per model | `results/tables/occurrence_coverage.csv` | asserted |
| `V4-12` | Complete literal intact in the C7 prompt for 98 of 850 spans per model; reproduced in the output for 17-89 (original) / 16-89 (V8) of them | Sec 4.6, Sec 5 limits | `c7c8_original`, `v8_main` | intact x leak contingency / 850 | `results/tables/input_intact_vs_output_leak.csv` | asserted (the 98), table (the rest) |
| `V4-16` | T7 probe encoding schemes: base64 17, ROT13 17, leetspeak 16 (decoding the supplied name string of the 50 T7 probes) | Sec 4.7 | `data/probes_extended.jsonl` | supplied name string / 50 T7 probes | `data/probes_extended.jsonl` | asserted |
| `MAIN-C2` | Oracle input redaction (C2): legacy leakage 0 for all six BF16 models | not in the audited claim map | `main` C2 | `leakage_rate` / 850 spans per model | `results/tables/model_condition_table.csv` | asserted |
| `QUANT-C5` | Quantized C5 (redaction + filter): 0 of 850 spans leaked for all four quantized variants | not in the audited claim map | `main` (INT8/INT4 C5) | `leakage_rate` / 850 | `results/tables/quantization_table.csv` | asserted |
| `MAIN-C0` | Baseline leakage, e.g. Llama-3.1-8B C0 = 0.5341; every BF16 C0 cell has 300 probes | not in the audited claim map | `main` C0 | `leakage_rate` / 850 spans per model | `results/tables/model_condition_table.csv` | asserted |
| `V4-13` | Table 11 caption and T7 channel table text appear verbatim in the manuscript | Sec 4.7 | - | text identity | - | not checkable here |
| `V4-14` | Filter described as pattern substitution without decoding | Sec 4.6 | `src/filter_output.py` | text claim | - | not checkable here (the code is in the repository, the sentence is not) |
| `V4-15` | "Redaction's leakage floor is the detector's miss rate" removed; six-stage text present | Sec 5, Sec 6 | - | text claim | - | not checkable here |

Not covered by any row above because this repository does not recompute them (`provenance/known_limitations.md`): detector recall (88.0%), paired-contrast intervals by identity/template/connected-group resampling, embedding-based partial leakage, and the AWQ calibration set (reconstructed, L7).

Channel definitions (full-string = `literal_full`, partial-string = `fragment_only` / partial-only, encoded-string = `encoded_*`; sequential multi-turn C7/C8 versus single-prompt re-ask) are in `DATA_DICTIONARY.md`.
