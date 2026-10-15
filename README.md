# open-llm-pii-audit

Code, synthetic probes, stored model outputs and analysis tables for an empirical audit of **in-context PII leakage and its mitigations in six open-weight LLMs** (Qwen3-8B, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct, Olmo-3-7B-Instruct, phi-4, gemma-4-E4B-it; plus INT8/INT4 variants of Qwen3-8B and phi-4). The paper's tables and figures are recomputed on a CPU from the stored outputs. **No model inference was run to build or verify this repository.**

The paper itself (title in `CITATION.cff`) is unpublished and not part of this repository; the author block is anonymous.

## What is verified, and what is not

| path | status |
|---|---|
| Recompute tables, figures and numbers from stored outputs: `bash scripts/reproduce_from_outputs.sh` | **verified** - categories 1-4 of `scripts/verify_artifact.py` pass (result and exit code in `provenance/validation_report.md`) |
| Unmodified full check: `python3 scripts/verify_artifact.py` | exits **2**: categories 1-4 pass, category 5 (re-generation) is **BLOCKED** on a machine without a GPU, vLLM and the model weights. It is never reported as a pass. |
| Re-run inference (`scripts/run_full_experiments.sh`) | **not run, not validated** here; weights and the serving environment are not included |

This is a reproduction of the audited numbers from stored outputs, not a claim that every number in the paper is reproduced. What each check covers, and what it does not, is in `REPRODUCIBILITY.md`; open limits are in `provenance/known_limitations.md`.

## Layout
```
configs/      models, generation/serving, mitigation conditions C0-C8, quantization, evaluation definitions (YAML)
data/         synthetic probes (main 300, extended 150), ground-truth spans, redaction map, quantization calibration texts
prompts/      refusal system prompt, T1-T9 prompt templates, holdout/utility templates
src/          generators, runners, redaction, output filter, leakage / partial / encoded / utility metrics, bootstrap, aggregation
scripts/      reproduce_from_outputs.sh, run_full_experiments.sh, generate_tables.py, generate_figures.py, verify_artifact.py
results/      raw/ (consolidated model outputs + run metadata), processed/ (probe-level, aggregate, intervals), tables/, figures/
tests/        53 unit tests (matchers, redaction and runner inputs, output filter, multi-turn, verifier exit codes)
provenance/   manifests and SHA256SUMS, pinned model revisions, software/hardware, run manifest, validation report, known limitations
```
`DATA_DICTIONARY.md` defines every file and field, `CLAIMS_TO_EVIDENCE.md` maps the paper's audited numbers to the records and checks that reproduce them.

## Install
Python >= 3.10, CPU only. The code uses the standard library; `numpy` and `matplotlib` are needed for the figures.
```
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate open-llm-pii-audit
```

## Run (from the repository root)
```
bash scripts/reproduce_from_outputs.sh                  # tables + figures + verification categories 1-4, about 3 minutes on one core
python3 scripts/verify_artifact.py --only 1,2,3,4       # the verification alone (a PARTIAL run by design)
python3 -m unittest discover -s tests                   # the 53 unit tests
python3 src/generate_probes.py --out-dir /tmp/probes    # regenerate the probe/ground-truth files (byte-identical to data/)
```
Verification logs and the regenerated figures go to `recompute_out/` (git-ignored). `scripts/generate_tables.py` rewrites `results/tables` and `results/processed` in place and reproduces them byte for byte; `scripts/generate_figures.py` writes `results/figures` by default, but the committed PDFs were drawn with matplotlib 3.11.2 and other versions change the PDF bytes, so `reproduce_from_outputs.sh` leaves them untouched.

## Conditions
C0 none; C1 refusal system prompt (`prompts/refusal_system.txt`); C2 oracle input redaction (ground-truth spans); C3 output filter; C4 C1+C2; C5 C2+C3; C6 C1+C2+C3; C7 detector input redaction (Presidio + spaCy; only what the detector saw is redacted; histories re-scanned before each request); C8 C7 + output filter. Exposure is reported by channel - full-string, partial-string (fragment), encoded-string - and is never summed into one number except as the labelled `exposure_any` union; sequential multi-turn generation (T9, holdout, C7/C8) is distinct from a single prompt that re-asks. Definitions: `DATA_DICTIONARY.md`, `configs/mitigation_conditions.yaml`, `configs/evaluation.yaml`.

**All PII is synthetic.** Names, `(555) 01xx` phones, `987-xx-xxxx` SSNs (non-issuable area), `4000 0000` cards and fictional domains/addresses come from the seeded generators in `src/generate_probes.py`; `tests/test_redaction.py` checks every ground-truth span against these markers.

## Resources
- Recompute path: CPU only, no network, about 3 minutes (`generate_tables.py` ~1.5 min, tests and checks ~1 min, figures seconds).
- The original inference (not included, not re-run): one NVIDIA H100 NVL (93 GiB), vLLM 0.30.0, temperature 0.2, max_tokens 256, request seed 42, Qwen3 thinking disabled; pinned model revisions in `provenance/model_revisions.json`.

## Size and large data
Everything ships in this repository (`results/raw/` is ~92 MB uncompressed; the whole repository is ~95 MB, ~7 MB zipped); there is no external download. Superseded first-pass cells, unselected V8 attempts, call ledgers and server logs are not included (`provenance/run_manifest.json`).

## License
- Code (`src/`, `scripts/`, `tests/`, configs): **MIT** (`LICENSE`).
- `data/` (synthetic probes, annotations, calibration texts): **CC BY 4.0** (`data/LICENSE`).
- Stored model outputs (`results/raw/`) and derived tables: **not relicensed here.** They were produced by third-party models with their own licences and terms (Qwen3, Mistral, Llama 3.1, Olmo 3, phi-4, Gemma 4); redistribution terms were not checked for this release (`known_limitations.md` L15). Check them before publishing `results/raw/`.

## Citation
See `CITATION.cff`.
