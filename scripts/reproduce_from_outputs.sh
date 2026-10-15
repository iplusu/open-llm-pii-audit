#!/usr/bin/env bash
# Regenerate every table and figure from the stored model outputs (CPU only, no inference, no network), then verify.
# Usage (repository root):  bash scripts/reproduce_from_outputs.sh
# Needs python3 with numpy + matplotlib (requirements.txt). Rewrites results/tables and results/processed in place (deterministic, byte-identical
# to the committed files); figures go to recompute_out/figures so the committed PDFs (drawn with matplotlib 3.11.2, whose PDF bytes depend on the
# matplotlib version) stay valid against provenance/SHA256SUMS. Verification logs go to recompute_out/. Exit code = scripts/verify_artifact.py's (0 all pass, 1 fail, 2 blocked/partial).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
"$PY" scripts/generate_tables.py
# Categories 1-4 are verifiable on CPU; category 5 (re-generation) needs a GPU and is reported BLOCKED here, never PASS.
# (verify_artifact.py clears recompute_out/ when it starts, so it runs before the figures are written there.)
rc=0
"$PY" scripts/verify_artifact.py --only 1,2,3,4 || rc=$?
"$PY" scripts/generate_figures.py --out recompute_out/figures
exit "$rc"
