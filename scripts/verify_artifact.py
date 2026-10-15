#!/usr/bin/env python3
"""Machine-checked verification of this repository, five categories, with failure propagation.
  1 File integrity (SHA256SUMS, record inventory) | 2 Stored-output metric reproduction (tests, regenerated data/tables/figures)
  3 Claim consistency (recomputed numbers vs the claims in CLAIMS_TO_EVIDENCE.md; manuscript text is NOT in this repo)
  4 Stored-output & timestamp provenance signal for the original C7/C8 run (NOT execution provenance; see provenance/known_limitations.md L1)
  5 Model re-generation (needs a GPU, vLLM and model weights; BLOCKED on a CPU-only host, never reported as PASS)

Step statuses: PASS | FAIL | BLOCKED (DEPENDENCY_MISSING: a required tool/module is not available) | SKIPPED_DEPENDENCY (an upstream step/category
did not PASS in THIS run, so the step was not run) | SKIPPED_PARTIAL (not requested in a partial run).
Exit code: 0 = every requested category PASS; 1 = at least one FAIL; 2 = no FAIL but at least one BLOCKED / SKIPPED_DEPENDENCY.
A partial run (--skip-slow or --only) prints PARTIAL, lists the categories not run as SKIPPED_PARTIAL, and is never reported as full verification.
Outputs a step declares in `fresh` are deleted before the step and must exist afterwards, so files left over from an earlier run are never evidence.
Needs numpy + matplotlib importable by `python3` (figures step); a missing module is reported BLOCKED, never PASS.
Usage (repository root): python3 scripts/verify_artifact.py [--skip-slow] [--only 1,3] [--root DIR] [--spec categories.json]
(--root/--spec exist so the isolated tests in tests/test_leakage_metrics.py can run fixtures; default = the real repository.)
Logs: recompute_out/verification_<category>.log and recompute_out/verification_summary.txt."""
import argparse, csv, datetime, hashlib, json, os, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

PASS, FAIL, BLOCKED, SKIP_DEP, SKIP_PART = "PASS", "FAIL", "BLOCKED", "SKIPPED_DEPENDENCY", "SKIPPED_PARTIAL"
SUITE_COUNTS = {"main": 15000, "c7c8_original": 3600, "adaptive": 1800, "v8_main": 3600, "v8_extended": 3600, "v8_impl_repeat": 864, "holdout": 10368}
RESULT_FILES = ["tables/model_condition_table.csv", "processed/probe_level_metrics.csv"]
DATA = ["probes_main", "probes_extended", "pii_ground_truth", "redaction_map", "quantization_calibration"]
DEFAULT = [
    {"id": "1", "name": "File integrity (SHA256SUMS complete and matching, 38,832-record inventory, probe/ground-truth counts)", "steps": [
        {"id": "1.a", "builtin": "sums"}, {"id": "1.b", "builtin": "sums_complete"}, {"id": "1.c", "builtin": "inventory"}]},
    {"id": "2", "slow": True, "name": "Stored-output metric reproduction (recomputed from results/raw, compared with the stored tables)", "steps": [
        {"id": "2.a", "cmd": "python3 -B -m unittest discover -s tests"},
        {"id": "2.b", "cmd": "python3 src/generate_probes.py --out-dir recompute_out/regen_probes", "fresh": [f"recompute_out/regen_probes/{n}.jsonl" for n in DATA]},
        {"id": "2.c", "cmd": " && ".join(f"cmp data/{n}.jsonl recompute_out/regen_probes/{n}.jsonl" for n in DATA), "after": ["2.b"]},
        {"id": "2.d", "builtin": "holdout_cases"},
        {"id": "2.e", "cmd": "python3 scripts/generate_tables.py --out recompute_out/results", "fresh": [f"recompute_out/results/{f}" for f in RESULT_FILES]},
        {"id": "2.f", "cmd": "diff -r recompute_out/results/tables results/tables && diff -r recompute_out/results/processed results/processed", "after": ["2.e"]},
        {"id": "2.g", "cmd": "python3 scripts/generate_figures.py --tables recompute_out/results/tables --out recompute_out/figures", "needs_py": ["numpy", "matplotlib"], "after": ["2.e"],
         "fresh": [f"recompute_out/figures/fig{n}.pdf" for n in ("1_heatmap", "2_tradeoff", "3_refusal")]}]},
    {"id": "3", "name": "Claim consistency (recomputed numbers only; the manuscript text is not part of this repository)", "steps": [
        {"id": "3.a", "builtin": "claims", "after": ["2.e"]}]},
    {"id": "4", "name": "Stored-output & timestamp provenance signal, original C7/C8 (NOT execution provenance; V8 derived-view identity is in category 2.a)", "steps": [
        {"id": "4.a", "builtin": "c7c8_provenance"}]},
    {"id": "5", "name": "Model re-generation smoke (3 probes, C0, one model; needs GPU + vLLM + weights; never run in the CPU-only release checks)", "steps": [
        {"id": "5.a", "cmd": "python3 src/run_experiments.py --jobs 'Qwen/Qwen3-8B::BF16' --conditions C0 --limit 3 --results-dir recompute_out/new_inference",
         "needs": ["nvidia-smi"], "needs_py": ["vllm", "openai"]}]},
]


def read_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(root):
    f = root / "provenance/SHA256SUMS"
    if not f.is_file():
        return None
    out = {}
    for line in f.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = h
    return out


def b_sums(root, log):
    sums = read_sums(root)
    if sums is None:
        log.write(f"provenance/SHA256SUMS missing under {root}\n"); return False
    bad = [f for f, h in sums.items() if not (root / f).is_file() or sha256_file(root / f) != h]
    for f in bad[:20]:
        log.write(f"MISMATCH/MISSING: {f}\n")
    log.write(f"SHA256SUMS entries {len(sums)}, mismatches/missing: {len(bad)}\n")
    return len(sums) > 0 and not bad


# the sums file cannot list itself, and the validation report records the result of a run that must precede it
EXEMPT_FROM_SUMS = {"provenance/SHA256SUMS", "provenance/validation_report.md"}
SKIP_DIRS = {".git", "recompute_out", "__pycache__", "new_inference_out", ".pytest_cache"}


def b_sums_complete(root, log):
    """Every file in the tree except EXEMPT_FROM_SUMS is listed (a stale manifest cannot hide a new or changed file)."""
    sums = read_sums(root)
    if sums is None:
        return False
    on_disk = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS)}
    on_disk -= EXEMPT_FROM_SUMS
    missing_from_sums, stale = sorted(on_disk - set(sums)), sorted(set(sums) - on_disk)
    log.write(f"files on disk {len(on_disk)}, listed {len(sums)}; unlisted: {missing_from_sums[:10]}; listed-but-absent: {stale[:10]}\n")
    return not missing_from_sums and not stale


def b_inventory(root, log):
    raw = root / "results/raw"
    ok = True
    n_out = n_meta = 0
    ids, cells, per_suite = set(), Counter(), Counter()
    meta = read_jsonl(raw / "execution_metadata.jsonl")
    for rec in read_jsonl(raw / "model_outputs.jsonl"):
        m = next(meta, None)
        n_out += 1; n_meta += m is not None
        if m is None or m["record_id"] != rec["record_id"] or not m.get("source_sha256"):
            ok = False; log.write(f"output/metadata misaligned at {rec['record_id']}\n"); break
        ids.add(rec["record_id"]); per_suite[rec["suite"]] += 1
        cells[(rec["suite"], rec["model"], rec["condition"], rec.get("seed"))] += 1
    n_meta += sum(1 for _ in meta)
    ok &= n_out == n_meta == len(ids) == sum(SUITE_COUNTS.values())
    log.write(f"model_outputs {n_out}, execution_metadata {n_meta}, unique record_ids {len(ids)}\n")
    ok &= dict(per_suite) == SUITE_COUNTS
    log.write(f"per-suite {dict(per_suite)} (expected {SUITE_COUNTS})\n")
    canonical = per_suite["main"] + per_suite["c7c8_original"] + per_suite["adaptive"]
    ok &= canonical == 20400
    log.write(f"canonical probe-condition records (main + c7c8_original + adaptive) = {canonical} (expected 20,400)\n")
    want = lambda s: 150 if s == "adaptive" else 300 if s in ("main", "c7c8_original", "v8_main") else None
    bad = [(k, n) for k, n in cells.items() if k[0] in ("main", "c7c8_original", "adaptive") and n != want(k[0])]
    ok &= not bad
    log.write(f"canonical cells with unexpected count: {bad[:5]}\n")
    gt = list(read_jsonl(root / "data/pii_ground_truth.jsonl"))
    counts = (sum(1 for _ in read_jsonl(root / "data/probes_main.jsonl")), sum(1 for _ in read_jsonl(root / "data/probes_extended.jsonl")),
              sum(g["suite"] == "main" for g in gt), sum(g["suite"] == "extended" for g in gt))
    ok &= counts == (300, 150, 850, 450)
    log.write(f"probes main/extended, ground-truth spans main/extended = {counts} (expected 300, 150, 850, 450)\n")
    ok &= sum(1 for _ in read_jsonl(root / "data/quantization_calibration.jsonl")) == 128
    return ok


def b_holdout_cases(root, log):
    """Holdout case files regenerate deterministically and hash to the values recorded in provenance/artifact_manifest.json."""
    sys.path.insert(0, str(root / "src"))
    import generate_probes as g
    want = json.loads((root / "provenance/artifact_manifest.json").read_text())["holdout_case_files_sha256"]
    got = {f"{suite}.jsonl": hashlib.sha256(g.serialize(rows).encode("utf-8")).hexdigest() for suite, rows in g.build_assets().items()}
    log.write(f"regenerated {got}\nexpected    {want}\n")
    return got == want


def _csv(path):
    return list(csv.DictReader(open(path)))


def b_claims(root, log):
    """Recompute-and-compare of the claims listed in CLAIMS_TO_EVIDENCE.md, against the tables regenerated in THIS run."""
    t = root / "recompute_out/results/tables"; pr = root / "recompute_out/results/processed"
    ch, h7, occ, inp = _csv(t / "channels_v8.csv"), _csv(t / "t7_channel_breakdown.csv"), _csv(t / "occurrence_coverage.csv"), _csv(t / "input_intact_vs_output_leak.csv")
    agg, mc = _csv(pr / "aggregate_metrics.csv"), _csv(t / "model_condition_table.csv")
    ho = {(r[0], r[1]): r for r in csv.reader(open(t / "holdout_exposure.csv")) if len(r) == 5}
    recon = next((r[2:4] for r in csv.reader(open(t / "holdout_exposure.csv")) if r and r[0] == "RECONCILE_fresh_vs_stored_views"), None)
    hu = _csv(t / "holdout_utility_by_family.csv")
    macro = lambda v: round(sum(float(r["strict_task_success"]) for r in hu if r["view"] == v and r["family"] in ("classification", "routing", "action_extraction", "summary")) / 24, 3)
    hv = lambda v, sc: tuple(int(x) for x in ho[(v, sc)][2:])
    models = ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Olmo-3-7B-Instruct", "Qwen3-8B", "gemma-4-E4B-it", "phi-4"]
    ci = lambda run, fam, cond, col: {r["model"]: int(r[col]) for r in ch if r["run"] == run and r["family"] == fam and r["cond"] == cond}
    legacy = lambda suite: sum(int(r["legacy_full"]) for r in agg if r["suite"] == suite and r["condition"] == "C7")
    t7 = {(r["model"], r["cond"]): r for r in h7}
    mcell = lambda m, c: next(r for r in mc if r["model"] == m and r["condition"] == c)
    import re
    full_token = re.compile(r"\[REDACTED_[A-Z_]+\]")
    damaged = lambda msgs: any("REDACTED" in full_token.sub("", m["content"]) or re.search(r"[A-Z]_[A-Z]+\]", full_token.sub("", m["content"])) for m in msgs)
    dmg = {"c7c8_original": Counter(), "v8_main": Counter()}
    for rec in read_jsonl(root / "results/raw/model_outputs.jsonl"):
        if rec["condition"] == "C7" and rec["suite"] in dmg:
            dmg[rec["suite"]][rec["model"]] += damaged(rec["prompt"])
    import base64, codecs
    names = {g["probe_id"]: g["text"] for g in read_jsonl(root / "data/pii_ground_truth.jsonl") if g["suite"] == "extended" and g["type"] == "name"}
    leet = str.maketrans({c: d for c, d in zip("aeiosAEIOS", "4310543105")})
    scheme = Counter()
    for pr in read_jsonl(root / "data/probes_extended.jsonl"):
        if pr["family"] == "T7":
            e, n = pr["encoded"]["name"], names[pr["probe_id"]]
            scheme["base64" if base64.b64encode(n.encode()).decode() == e else "rot13" if codecs.encode(n, "rot_13") == e else "leet" if n.translate(leet) == e else "other"] += 1
    C = [
        ("V4-16", "T7 probe encoding schemes (supplied name string): base64 17, ROT13 17, leetspeak 16", dict(scheme) == {"base64": 17, "rot13": 17, "leet": 16}),
        ("E2", "Damaged C7 prompts (placeholder defect): 108/1,800 in the original run (18 per model), 0 in V8",
         sum(dmg["c7c8_original"].values()) == 108 and set(dmg["c7c8_original"].values()) == {18} and sum(dmg["v8_main"].values()) == 0),
        ("B1c", "C8 full-string probes (main): only Qwen3-8B, 1", {m: ci("main", "main", "C8", "full_probes")[m] for m in models} == dict(zip(models, [0, 0, 0, 1, 0, 0]))),
        ("B2b", "Extended C8 paper-matcher full-string spans: 6, all Qwen3-8B T7",
         sum(int(r["paper_matcher_spans"]) for r in ch if r["run"] == "ext" and r["cond"] == "C8") == 6 == int(next(r for r in ch if r["run"] == "ext" and r["family"] == "T7" and r["cond"] == "C8" and r["model"] == "Qwen3-8B")["paper_matcher_spans"])),
        ("B2c", "T7 C8 encoded-input repeat 23-32/50 for five models, Gemma 5",
         all(23 <= int(t7[(m, "C8")]["input_echo_probes"]) <= 32 for m in models if m != "gemma-4-E4B-it") and int(t7[("gemma-4-E4B-it", "C8")]["input_echo_probes"]) == 5),
        ("B2d", "T8/T9 any-channel probes unchanged C7->C8",
         all(ci("ext", f, "C7", "any_probes") == ci("ext", f, "C8", "any_probes") for f in ("T8", "T9"))),
        ("B3-ALL", "Holdout ALL: C0 559/659, C2_unique 3/3, C7 369/442, C8 327/400 of 3,456",
         [hv(v, "ALL") for v in ("C0", "C2_unique", "C7", "C8")] == [(559, 659, 3456), (3, 3, 3456), (369, 442, 3456), (327, 400, 3456)]),
        ("B3p", "Holdout prohibited C0/C7/C8 literal 224/157/139 of 1,836", [hv(v, "prohibited") for v in ("C0", "C7", "C8")] == [(224, hv("C0", "prohibited")[1], 1836), (157, hv("C7", "prohibited")[1], 1836), (139, hv("C8", "prohibited")[1], 1836)]),
        ("B3a", "Holdout authorized-record C0/C7/C8 literal 199/145/127 of 216", [(hv(v, "authorized_return")[0], hv(v, "authorized_return")[2]) for v in ("C0", "C7", "C8")] == [(199, 216), (145, 216), (127, 216)]),
        ("B3n", "Holdout no-PII-requested C0/C2/C7/C8 literal 136/3/67/61 of 1,404", [(hv(v, "no_pii_requested")[0], hv(v, "no_pii_requested")[2]) for v in ("C0", "C2_unique", "C7", "C8")] == [(136, 1404), (3, 1404), (67, 1404), (61, 1404)]),
        ("B3b", "Holdout views re-scored fresh: 13,824, 0 mismatches vs stored exposure", recon == ["13824", "0"]),
        ("B3u", "Holdout utility macro-mean strict success C0/C2_unique/C7/C8 = .367/.410/.387/.387 (24 model-family cells)", [macro(v) for v in ("C0", "C2_unique", "C7", "C8")] == [0.367, 0.41, 0.387, 0.387]),
        ("E1", "C7 leaked spans: original 738, V8 730", (legacy("c7c8_original"), legacy("v8_main")) == (738, 730)),
        ("V4-5", "T7 C8 partial-only probes 0/0/0/1/22/1 (model order Llama, Mistral, Olmo, Qwen, Gemma, phi-4)", [int(t7[(m, "C8")]["partial_only_probes"]) for m in models] == [0, 0, 0, 1, 22, 1]),
        ("V4-6", "T7 C8 literal-full 0/0/0/5/0/0", [int(t7[(m, "C8")]["literal_full_probes"]) for m in models] == [0, 0, 0, 5, 0, 0]),
        ("V4-10", "Main benchmark: 23 multi-occurrence entities per model, 18 masked throughout, 3 mixed",
         all((int(r["entities"]), int(r["all_masked"]), int(r["mixed"])) == (23, 18, 3) for r in occ if r["run"] == "v8_corrected_main" and r["entity_class"] == "multi")),
        ("V4-11", "Extended: 82 multi-occurrence entities per model (71 masked throughout, 9 mixed, 2 none)",
         all((int(r["entities"]), int(r["all_masked"]), int(r["mixed"]), int(r["none_masked"])) == (82, 71, 9, 2) for r in occ if r["run"] == "v8_extended" and r["entity_class"] == "multi")),
        ("V4-12", "Complete literal intact in the C7 prompt for 98/850 spans per model", all(int(r["intact_leak"]) + int(r["intact_noleak"]) == 98 for r in inp)),
        ("MAIN-C2", "C2 oracle redaction legacy leakage = 0 for all six BF16 models", all(float(mcell(m, "C2")["leakage_rate"]) == 0 for m in models)),
        ("QUANT-C5", "Quantized C5 leakage = 0 of 850 for all four quantized variants",
         all(float(mcell(m, "C5")["leakage_rate"]) == 0 and mcell(m, "C5")["n_probes"] == "300" for m in ("Qwen3-8B-INT8", "Qwen3-8B-INT4", "phi-4-INT8", "phi-4-INT4"))),
        ("MAIN-C0", "C0 baseline legacy leakage: Llama-3.1-8B 0.5341; 300 probes in every BF16 C0 cell", float(mcell("Llama-3.1-8B-Instruct", "C0")["leakage_rate"]) == 0.5341 and all(mcell(m, "C0")["n_probes"] == "300" for m in models)),
    ]
    md = (root / "CLAIMS_TO_EVIDENCE.md").read_text() if (root / "CLAIMS_TO_EVIDENCE.md").is_file() else ""
    ok = True
    for cid, text, good in C:
        listed = f"`{cid}`" in md
        log.write(f"{'ok  ' if good and listed else 'FAIL'} {cid}: {text}  [value reproduced: {bool(good)}; listed in CLAIMS_TO_EVIDENCE.md: {listed}]\n")
        ok &= bool(good) and listed
    return ok


def b_c7c8_provenance(root, log):
    """Original C7/C8: stored C8 equals filter(stored C7) for only 410/1,800 pairs; every C8 stamp is > 24 h after its C7 twin."""
    sys.path.insert(0, str(root / "src"))
    from filter_output import apply_output_filter
    ts = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    c7, c8 = {}, {}
    for (rec, meta) in zip(read_jsonl(root / "results/raw/model_outputs.jsonl"), read_jsonl(root / "results/raw/execution_metadata.jsonl")):
        if rec["suite"] == "c7c8_original":
            (c7 if rec["condition"] == "C7" else c8)[(rec["model"], rec["probe_id"])] = (rec, meta)
    per, gaps = Counter(), []
    for k, (r7, m7) in c7.items():
        r8, m8 = c8[k]
        per[k[0]] += apply_output_filter(r7["output"])[0] == r8["output"]
        gaps.append((ts(m8["timestamp_utc"]) - ts(m7["timestamp_utc"])).total_seconds() / 3600)
    expect = {"Llama-3.1-8B-Instruct": 83, "Mistral-7B-Instruct-v0.3": 75, "Olmo-3-7B-Instruct": 44, "Qwen3-8B": 112, "gemma-4-E4B-it": 81, "phi-4": 15}
    log.write(f"pairs {len(gaps)}, filter(C7)==stored C8: {sum(per.values())} {dict(per)}; C8-C7 gap hours min {min(gaps):.1f} max {max(gaps):.1f}\n")
    return len(gaps) == 1800 and sum(per.values()) == 410 and dict(per) == expect and min(gaps) > 24


BUILTINS = {"sums": b_sums, "sums_complete": b_sums_complete, "inventory": b_inventory, "holdout_cases": b_holdout_cases, "claims": b_claims,
            "c7c8_provenance": b_c7c8_provenance}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true", help="PARTIAL run: skip the slow metric-reproduction category (and its dependents)")
    ap.add_argument("--only", default="", help="comma-separated category ids; PARTIAL run unless all categories are named")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--spec", default="", help="JSON list of categories (same schema as DEFAULT); for isolated tests")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    cats = json.loads(Path(a.spec).read_text()) if a.spec else DEFAULT
    want = {x for x in a.only.split(",") if x} or {c["id"] for c in cats}
    if a.skip_slow:
        want -= {c["id"] for c in cats if c.get("slow")}
    partial = want != {c["id"] for c in cats}
    logdir = root / "recompute_out"
    shutil.rmtree(logdir, ignore_errors=True); logdir.mkdir(parents=True)  # nothing from an earlier run survives
    res, cat_status = {}, {}
    for c in cats:
        if c["id"] not in want:
            cat_status[c["id"]] = SKIP_PART; print(f"[{c['id']}] {c['name']}: {SKIP_PART} (not requested; this run is PARTIAL)"); continue
        log = (logdir / f"verification_{c['id']}.log").open("w")
        bad_up = [u for u in c.get("needs_cat", []) if cat_status.get(u) != PASS]
        for s in c["steps"]:
            sid, why = s["id"], ""
            ups = [u for u in s.get("after", []) if res.get(u, (None,))[0] != PASS]
            tools = [t for t in s.get("needs", []) if not shutil.which(t)]
            tools += [f"python module {m}" for m in s.get("needs_py", [])
                      if subprocess.run([sys.executable, "-c", f"import {m}"], capture_output=True).returncode]
            if ups or (bad_up and not s.get("manifest")):
                st, why = SKIP_DEP, "upstream not PASS in this run: " + ", ".join(ups or bad_up)
            elif tools:
                st, why = BLOCKED, "DEPENDENCY_MISSING: " + ", ".join(tools)
            elif s.get("manifest") or s.get("builtin"):
                name = "sums" if s.get("manifest") else s["builtin"]
                try:
                    st = PASS if BUILTINS[name](root, log) else FAIL
                except Exception as e:  # a crashing check is a FAIL, never a pass
                    log.write(f"builtin {name} raised {type(e).__name__}: {e}\n"); st, why = FAIL, f"{type(e).__name__}: {e}"
            else:
                for f in s.get("fresh", []): (root / f).unlink(missing_ok=True)
                t0 = time.time()
                cmd = s["cmd"].replace("python3 ", f"{sys.executable} ", 1) if s["cmd"].startswith("python3 ") else s["cmd"]
                p = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True)
                log.write(f"$ {s['cmd']}\n{p.stdout}{p.stderr}\n[exit {p.returncode}]\n\n")
                missing = [f for f in s.get("fresh", []) if not (root / f).is_file() or (root / f).stat().st_mtime < t0 - 1]
                if p.returncode == 127: st, why = BLOCKED, "DEPENDENCY_MISSING: command not found (exit 127)"
                elif p.returncode: st, why = FAIL, f"exit {p.returncode}"
                elif missing: st, why = FAIL, "expected fresh output not produced: " + ", ".join(missing)
                else: st = PASS
            res[sid] = (st, why); log.write(f"[{sid}] {st} {why}\n")
        log.close()
        sts = [res[s["id"]][0] for s in c["steps"]]
        cat_status[c["id"]] = FAIL if FAIL in sts else BLOCKED if BLOCKED in sts else SKIP_DEP if SKIP_DEP in sts else PASS
        print(f"[{c['id']}] {c['name']}: {cat_status[c['id']]}  ({sts.count(PASS)}/{len(sts)} steps PASS)  detail: recompute_out/verification_{c['id']}.log")
        for s in c["steps"]:
            if res[s["id"]][0] != PASS: print(f"     {res[s['id']][0]}: [{s['id']}] {s.get('cmd') or s.get('builtin') or 'sums'}  -- {res[s['id']][1]}")
    vals = [v for k, v in cat_status.items() if k in want]
    code = 1 if FAIL in vals else 2 if any(v in (BLOCKED, SKIP_DEP) for v in vals) else 0
    banner = ("PARTIAL verification (NOT full): requested categories " + ", ".join(sorted(want)) + "; not run: "
              + (", ".join(sorted(set(cat_status) - want)) or "none") if partial else "FULL machine-checked verification: all categories requested")
    print(f"SUMMARY: {banner}; requested-category result: {'ALL PASS' if code == 0 else 'NOT PASS'}; exit code {code}")
    (logdir / "verification_summary.txt").write_text(json.dumps({"exit_code": code, "partial": partial, "categories": cat_status,
                                                                 "steps": {k: list(v) for k, v in res.items()}}, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
