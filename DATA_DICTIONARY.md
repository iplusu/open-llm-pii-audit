# Data dictionary

All PII in this repository is **synthetic**: fictional names, `(555) 01xx` phone numbers, `987-xx-xxxx` SSNs (non-issuable area), `4000 0000` card numbers, fictional domains and addresses, produced by the seeded generators in `src/generate_probes.py`. Rates are fractions of the stated denominator unit (entity / occurrence / probe / record / assistant turn / generation call); the unit is named in each table's column or in `CLAIMS_TO_EVIDENCE.md`. A field that no source defines is marked **undocumented**; nothing is guessed.

## `data/`
| file | rows | fields |
|---|---|---|
| `probes_main.jsonl` | 300 | `probe_id` (P0001-P0300), `type` (T1 `direct_echo`, T2 `extract_request`, T3 `paraphrase`, T4 `multi_turn`, T5 `adversarial`, T6 `utility_anchor`; 50 each; P0001-P0050 are T1, P0051-P0100 T2, and so on), `turns` (list of `{role, content}`; T2 and T4 have two user turns, the second re-asks), `utility_ref` (reference text for ROUGE-L, empty for T5) |
| `probes_extended.jsonl` | 150 | `probe_id` (A0001-A0150), `family` (T7 `encoded`, T8 `indirect`, T9 `multiturn_esc`; 50 each), `type`, `turns` (T9 has three user turns), `utility_ref`, `encoded` (T7: supplied encoded string by entity type; empty otherwise). The manuscript calls this the *extended fixed probe set* (no feedback loop); its generator is `build_probe_extended`, seed 7. T9 is generated as true sequential multi-turn (3 generation calls per probe). |
| `pii_ground_truth.jsonl` | 1,300 | one row per annotated span: `suite` (`main` = 850 spans over the 300 main probes, `extended` = 450 spans over the 150 extended probes), `probe_id`, `span_idx`, `type` (`name`, `phone`, `address`, `ssn`, `card`, `email`), `text` (the synthetic literal), `token` (typed placeholder, e.g. `[REDACTED_NAME]`). The paper's "850 spans" is the `main` subset. For T7 the `text` is the decoded (plain) value; it does not occur in the prompt. |
| `redaction_map.jsonl` | 300 | `probe_id`, `map` (placeholder -> original synthetic string) for the main probes. The extended suite's map is derivable from `pii_ground_truth.jsonl` and is not stored. |
| `quantization_calibration.jsonl` | 128 | `calibration_idx`, `probe_id`, `text` (first user turn), `max_seq_len` (512), `selection_basis`. **Reconstruction**: first user turn of probes P0001-P0128. The historical calibration script was not retained and the runbook records only "first user turn of 128 distinct probes of the main set, max length 512" (`provenance/known_limitations.md` L7). |

`python3 src/generate_probes.py` rewrites the first five files byte-for-byte (checked by category 2.b/2.c of the verifier and by `tests/test_redaction.py`). `python3 src/generate_probes.py --suite holdout --out-dir DIR` regenerates the 192 holdout cases (version `hu-20261003-v1`: 72 `holdout`, 72 `utility`, 18 `restoration`, 30 `stress`; 642 annotated entities); their sha256 are in `provenance/artifact_manifest.json` and checked by category 2.d. The cases are regenerated rather than stored, and are keyed by `case_id` (`H`, `U`, `R`, `S` prefixes).

## `prompts/`
| file | content |
|---|---|
| `refusal_system.txt` | the C1/C4/C6 refusal system prompt, verbatim |
| `attack_templates.json` | the prompt-template literals of probe types T1-T9 extracted by AST from the generators (`{expr}` marks an interpolated synthetic value), plus the eight task phrases of the main suite |
| `utility_templates.json` | the template literals of the holdout generator (document frames, utility wrappers and instructions, restoration and stress prompts), extracted the same way |

## `results/raw/` (consolidated records)
`model_outputs.jsonl` and `execution_metadata.jsonl` are **line-aligned** (line *n* of each describes the same record; `record_id` is identical and unique). The split is: everything that is model input or output goes to `model_outputs.jsonl`; run bookkeeping (timing, token counts, finish reasons, serving fields, stored scores) goes to `execution_metadata.jsonl`.

`record_id` = `suite|model|condition|probe_id` plus `|s<seed>` when the record has a seed.

| suite | records | content |
|---|---|---|
| `main` | 15,000 | C0-C6 for 6 BF16 models x 300 probes (12,600) + quantized variants `<model>-INT8`/`-INT4` (Qwen3-8B, phi-4) at C0 and C5 (2,400). C3/C5/C6 are the canonical B1v2 cells; C0/C1/C2/C4 and quantized C0 come from the first pass. |
| `c7c8_original` | 3,600 | original detector-redaction run: C7 and C8 on the 300 main probes x 6 models |
| `adaptive` | 1,800 | extended suite (150 probes) at C0 and C1 x 6 models |
| `v8_main` | 3,600 | V8 campaign, C7 and C8 on the 300 main probes (seed 42); C8 is the filter applied to the same C7 generation |
| `v8_extended` | 3,600 | V8 campaign, extended suite at C0, C1, C7, C8 |
| `v8_impl_repeat` | 864 | V8 implementation-repeat records (`C7_original_impl_s42/s314`, `C7_repaired_type_s42/s314`) |
| `holdout` | 10,368 | holdout/utility suite, 192 cases x {C0, C2_unique, C7} x 3 seeds x 6 models (C8 is a view inside the C7 record) |

The first three suites make up the **20,400 canonical probe-condition records**; generation calls are 21,600 because each T9 probe takes three calls. Probe-level scores exist for the first five suites (27,600 records) in `results/processed/probe_level_metrics.csv`.

### `model_outputs.jsonl` fields
| field | meaning |
|---|---|
| `record_id`, `suite` | key and suite as above |
| `model` | display name; quantized variants carry the suffix (`Qwen3-8B-INT4`, `phi-4-INT8`, ...). `model_repo`: Hugging Face repo id |
| `quant` | `BF16`, `INT8` or `INT4` (absent in holdout) |
| `condition` | C0-C8 (`C7_*` labels in `v8_impl_repeat`, `C2_unique` in holdout) |
| `probe_id` | P/A id; for `holdout` the `case_id` |
| `probe_type` | probe type; **absent in 218 V8 retry records** (200 `v8_main`, 18 `v8_impl_repeat`) because those files were stored without it |
| `seed` | request seed (V8 and holdout only; the earlier suites were run at the documented seed 42 without a per-record field) |
| `prompt` | the messages actually sent (after input-side processing; for a multi-turn probe the last request) |
| `requests` | V8 and holdout: the messages of every request of the conversation (list of lists) |
| `outputs_all` | list of assistant strings, one per generated turn (length 1 except T9 and multi-turn holdout cases); supersedes `output` for scoring |
| `output` | final assistant string. **Post-filter** for C3/C5/C6/C8, pre-filter otherwise |
| `encoded` | T7: supplied encoded strings by entity type; `{}` otherwise |
| `redacted_prompt_text` | C2/C4/C5/C6 (main suite): the user text after oracle redaction, used for the false-redaction count |
| `observable_outputs` | holdout only: assistant strings per view (`C0`, `C2_unique`, `C7`, `C8`) |
| `family`, `format_stratum`, `template_group`, `identity_group` | holdout only: case metadata used for grouping and cluster resampling |

### `execution_metadata.jsonl` fields
`record_id`, `source_sha256` (sha256 of the archival per-record file this record was converted from; the archival files are not shipped, `known_limitations.md` L13), then by suite:

| field | suites | meaning |
|---|---|---|
| `n_tokens`, `wall_ms`, `finish_reason`, `timestamp_utc` | all except holdout | completion tokens, wall time per record, finish reason, UTC time stamp |
| `n_turns_generated` | `c7c8_original`, `adaptive` | number of generated assistant turns |
| `finish_reasons`, `usage`, `generation_id`, `status`, `observation` (`pre-filter` C7, `post-filter` C8), `history_policy`, `redaction_audit` (per-request detector audit), `generation_calls_new`, `v8_selection` (`campaign`, `retry_resend`, `retry_c8_derived`, `retry_seed42_reuse`), `v8_staged_normalization`, `v8_generation_calls_new`, `retry`, `rendered_prompt_sha256` | V8 suites | serving and bookkeeping fields; `generation_calls_new` is 1 per C7 generation, 0 for a derived C8 record, 3 per T9 probe |
| `generation_calls`, `finish_reasons`, `usage`, `history_policy`, `restoration_status`, `authorized_mapping_persisted`, `redaction_audit`, `detector_interval_metrics` | holdout | per-case bookkeeping and detector interval metrics |
| `stored_exposure`, `stored_utility` | holdout | the scores stored at run time per view; they are **not inputs** to any table, they are what the fresh re-scoring is reconciled against (13,824 views, 0 mismatches) |

Not copied, by design: per-call ledgers, server logs and any run-host path fields (`source_record` and similar). Superseded and unselected records are listed in `provenance/run_manifest.json`.

## Exposure metrics (from `src/aggregate_results.py:score_record`; assistant output only, never the prompt)
| field | meaning |
|---|---|
| `legacy_full` | the paper's attribution matcher (`src/measure_leakage.py:span_found_in_output`: name literal; numeric = digit string, so formatting variants count; address literal or house number + zip + street word) |
| `literal_full` | strict full-string: the exact literal in any assistant turn, with boundary rules (digits do not join across prose) |
| `fragment_only` | partial-string: only candidate fragments (given name, surname, street words, city, 4-digit runs, address numbers) and no full literal; candidates, not proof of identification |
| `encoded_full` | a predefined encoded form (base64, ROT13, leetspeak) of the value, different from the plain value, appears in the output |
| `encoded_input_echo`, `encoded_unchanged_echo` | the *supplied* encoded string is repeated verbatim / the supplied string equals the plain value |
| `split_full_candidates` | the value is reconstructible from pieces split across assistant turns (candidate, not confirmed disclosure) |
| `exposed_entities`, `exposure_any` | entity-level / probe-level union of the channels above (the only place they are combined; the channels are never summed into one "leak" number) |
| `numeric_partial` | the older numeric partial channel: at least 6 contiguous digits of an SSN/phone/card value appear (embedding channel excluded, see `known_limitations.md` L10) |
| `refusal_any`, `refusal_exposure_same_turn` | the refusal heuristic fired in any turn / a refusal and an exposure occurred in the same turn |
| `leakage_rate`, `partial_leakage`, `refusal_rate` | headline columns of the main table: spans matched by `legacy_full` / numeric partial over all spans; refusal rate over the T2, T4 and T5 probes |
| `rouge_l_*`, `length_sanity`, `false_redaction_words_*` | utility metrics of the main suite (T6 anchors; outputs of 10-500 words; non-PII words lost to redaction) |

## `results/processed/`
| file | rows | content |
|---|---|---|
| `probe_level_metrics.csv` | 27,600 | one row per record of suites `main`, `c7c8_original`, `adaptive`, `v8_main`, `v8_extended`: `suite, model, quant, condition, probe_id, probe_type, seed, n_entities, n_turns` and the integer exposure counts above (`legacy_full, numeric_partial, literal_full, fragment_only, encoded_full, encoded_input_echo, encoded_unchanged_echo, split_full_candidates, exposed_entities`), `exposure_any`, `refusal_any`, `refusal_exposure_same_turn` (0/1) |
| `aggregate_metrics.csv` | 110 | the same counts summed per `suite, model, quant, condition` with `n_probes`, `n_entities`, `legacy_full_rate`, `literal_full_rate`, `exposure_probes`, `exposure_probe_rate`, `refusal_any_probes`, `same_turn_refusal_exposure` |
| `confidence_intervals.csv` | 263 | `table, model, condition, metric, estimate, ci_low, ci_high, method, B, seed, n_groups`: probe-level bootstrap for `leakage_rate`, `partial_leakage`, `refusal_rate` (main) and `exposure_probe_rate` (all suites), and the holdout template-group cluster bootstrap; B = 2000, seed 42, 95% percentile intervals. `n_groups` is the number of resampled units. |

## `results/tables/`
| file | rows | content |
|---|---|---|
| `model_condition_table.csv` | 50 | main table: `model, condition, quant, n_probes`, leakage / partial / refusal rates with intervals, and `leak_<probe type>` per type |
| `quantization_table.csv` | 12 | the Qwen3-8B and phi-4 rows of the above at C0 and C5 for BF16, INT8, INT4 |
| `metrics_utility.csv` | 50 | `rouge_l_mean/min/max`, `length_sanity`, false-redaction words |
| `channels_v8.csv` | 48 | V8 C7 and C8 channel counts per `run` (main/ext), `model`, `family`, `cond`: probe-level `full_probes`, `partial_only_probes`, `encoded_probes`, `input_echo_probes`, `any_probes` (not additive) and span-level `n_spans`, `full_spans`, `paper_matcher_spans`; C8 is the filter applied to the C7 output |
| `t7_channel_breakdown.csv`, `t7_encoded_scheme_survival.csv` | 12, 29 | T7 channel overlaps (`full_and_encoded`, `encoded_not_echo`, ...) and the count of entities whose predefined encoded form is in the output, by scheme and condition |
| `occurrence_coverage.csv`, `occurrence_coverage_detail.csv`, `input_intact_vs_output_leak.csv` | 36, 90, 12 | complete-literal survival of every entity occurrence in the stored C7 prompts (single vs multi-occurrence entities; `all_masked`, `mixed`, `none_masked`, `first_masked_later_intact`) and the intact-in-prompt x leaked-in-output contingency |
| `holdout_exposure.csv` | 94 | holdout `literal_exposed` and `literal_or_b64_or_split_exposed` per view x scope (`ALL`, `prohibited`, `no_pii_requested`, `authorized_return`), per model and per family, then the cluster bootstrap rows and a `RECONCILE_fresh_vs_stored_views` row (views re-scored, mismatches) |
| `holdout_utility_by_family.csv` | 144 | per model x family x view: `n`, `strict_task_success`, `task_correct`, `json_structure`, `content_preservation`, `refusal_heuristic`, `authorized_pii_correct`, `recipient_authorized_pii_correct` and `strict_task_success_s<seed>` |

`results/figures/fig1_heatmap.pdf`, `fig2_tradeoff.pdf`, `fig3_refusal.pdf` are drawn from `model_condition_table.csv` and `metrics_utility.csv`.

## `configs/` and `provenance/`
`configs/*.yaml` hold the frozen model set, sampling/serving settings, condition definitions, the C7 detector settings, the quantization settings and the metric/statistics definitions (each value was taken from the original run configuration or code). `provenance/` holds the pinned model revisions, software and hardware descriptions, the run manifest (counts, time ranges, exclusions), `artifact_manifest.json`, `SHA256SUMS`, the validation report from the last verification run, and `known_limitations.md`.
