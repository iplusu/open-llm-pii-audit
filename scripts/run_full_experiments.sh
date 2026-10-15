#!/usr/bin/env bash
# Full experiment launcher (NEW INFERENCE; needs a GPU host with vLLM 0.30.0 and the pinned model snapshots in a local
# Hugging Face cache; NOT run for this repository's release checks). Resumable: existing per-record files are skipped.
#
# Phase 1: BF16 main matrix  - 6 models x 7 conditions (C0-C6) x 300 probes = 12,600 generations
# Phase 2: quantization      - (Qwen3-8B, phi-4) x (INT8, INT4) x (C0, C5) x 300 = 2,400 generations
# Phase 3: detector C7/C8 (main probes) and Phase 4: extended-suite C0/C1 (T7-T9), both via src/run_multiturn.py (C7/C8 need presidio + spaCy models)
# Output goes to new_inference_out/ (never to results/raw). Convert to the consolidated JSONL layout with
# src/aggregate_results.py before comparing with the stored outputs.
#
# NOTE: Phase 2 INT4 uses the AWQ checkpoints described in configs/quantization.yaml, which are NOT included; the
# runtime-bitsandbytes INT4 path in src/run_experiments.py is the superseded route (provenance/known_limitations.md L7).
# Launch:  nohup bash scripts/run_full_experiments.sh > new_inference_out/inference.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY="${PY:-python3}"
mkdir -p new_inference_out/logs

echo "===== PHASE 1/4: BF16 main matrix (6 models x 7 conditions) ====="
"$PY" src/run_experiments.py \
  --jobs "Qwen/Qwen3-8B::BF16" \
  --jobs "mistralai/Mistral-7B-Instruct-v0.3::BF16" \
  --jobs "RedHatAI/Llama-3.1-8B-Instruct::BF16" \
  --jobs "allenai/Olmo-3-7B-Instruct::BF16" \
  --jobs "microsoft/phi-4::BF16" \
  --jobs "google/gemma-4-E4B-it::BF16" \
  --conditions C0,C1,C2,C3,C4,C5,C6

echo "===== PHASE 2/4: quantization sweep (2 models x INT8/INT4 x C0,C5) ====="
"$PY" src/run_experiments.py \
  --jobs "Qwen/Qwen3-8B::INT8" \
  --jobs "Qwen/Qwen3-8B::INT4" \
  --jobs "microsoft/phi-4::INT8" \
  --jobs "microsoft/phi-4::INT4" \
  --conditions C0,C5

echo "===== PHASE 3/4: detector C7/C8 on the main probes (live detector, sequential multi-turn) ====="
"$PY" - <<'PYEOF'
import json, sys
sys.path.insert(0, "src")
from run_experiments import load_probes
from pathlib import Path
probes = load_probes(Path("data"))
Path("new_inference_out/main_probes_with_spans.jsonl").write_text("".join(json.dumps(p) + "\n" for p in probes))
PYEOF
"$PY" src/run_multiturn.py \
  --probes new_inference_out/main_probes_with_spans.jsonl --results-dir new_inference_out/c7c8 \
  --jobs "Qwen/Qwen3-8B" --jobs "mistralai/Mistral-7B-Instruct-v0.3" --jobs "RedHatAI/Llama-3.1-8B-Instruct" \
  --jobs "allenai/Olmo-3-7B-Instruct" --jobs "microsoft/phi-4" --jobs "google/gemma-4-E4B-it" \
  --conditions C7,C8

echo "===== PHASE 4/4: extended suite T7-T9, C0 and C1 (sequential multi-turn for T9) ====="
"$PY" src/run_multiturn.py \
  --probes data/probes_extended.jsonl --results-dir new_inference_out/extended \
  --jobs "Qwen/Qwen3-8B" --jobs "mistralai/Mistral-7B-Instruct-v0.3" --jobs "RedHatAI/Llama-3.1-8B-Instruct" \
  --jobs "allenai/Olmo-3-7B-Instruct" --jobs "microsoft/phi-4" --jobs "google/gemma-4-E4B-it" \
  --conditions C0,C1

echo "===== COMPLETE: $(find new_inference_out -name '*.json' ! -name '*.tmp' | wc -l) record files ====="
