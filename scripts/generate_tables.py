#!/usr/bin/env python3
"""Regenerate every table in results/ from the stored model outputs (CPU only, no inference).

  results/tables/     model_condition_table, quantization_table, metrics_utility, channels_v8, holdout_exposure, holdout_utility_by_family,
                      t7_channel_breakdown, t7_encoded_scheme_survival, occurrence_coverage(+_detail), input_intact_vs_output_leak
  results/processed/  probe_level_metrics, aggregate_metrics, confidence_intervals

Logic is moved from the tested recompute scripts (main-table aggregation, V3 channels/holdout, V4 T7 breakdown and
occurrence coverage); the matchers live in src/. Run from the repository root:  python3 scripts/generate_tables.py [--out DIR]
"""
import argparse
import collections
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import aggregate_results as A  # noqa: E402
import bootstrap_analysis as BA  # noqa: E402
from filter_output import apply_output_filter  # noqa: E402
from generate_probes import holdout_cases  # noqa: E402
from measure_encoded_exposure import encoded_hits  # noqa: E402
from measure_leakage import span_found_in_output  # noqa: E402
from measure_utility import measure_utility_group, score_exposure, score_utility  # noqa: E402

MODELS = ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Olmo-3-7B-Instruct", "Qwen3-8B", "gemma-4-E4B-it", "phi-4"]
FAM = {"encoded": "T7", "indirect": "T8", "multiturn_esc": "T9"}
MC_FIELDS = (["model", "condition", "quant", "n_probes", "leakage_rate", "leakage_ci_low", "leakage_ci_high", "partial_leakage",
              "partial_ci_low", "partial_ci_high", "refusal_rate", "refusal_ci_low", "refusal_ci_high"] + [f"leak_{t}" for t in A.MAIN_TYPES])
PL_FIELDS = ["suite", "model", "quant", "condition", "probe_id", "probe_type", "seed", "n_entities", "n_turns", "legacy_full", "numeric_partial",
             "literal_full", "fragment_only", "encoded_full", "encoded_input_echo", "encoded_unchanged_echo", "split_full_candidates",
             "exposed_entities", "exposure_any", "refusal_any", "refusal_exposure_same_turn"]
fmt4 = lambda v: f"{v:.4f}" if isinstance(v, float) else v


def write_rows(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main_tables(recs, span_map, ptype, probes_main, tables, b):
    groups = defaultdict(list)
    for r in recs:
        if r["suite"] == "main":
            groups[(r["model"], r["condition"])].append(r)
    rows = A.model_condition_rows(groups, span_map, ptype, b=b)
    A.write_csv(tables / "model_condition_table.csv", rows, MC_FIELDS, fmt4)
    q = [r for r in rows if r["quant"] and r["condition"] in ("C0", "C5") and r["model"].split("-INT")[0] in ("Qwen3-8B", "phi-4")]
    A.write_csv(tables / "quantization_table.csv", q, MC_FIELDS, fmt4)
    um = {p["probe_id"]: p for p in probes_main}
    urows = []
    for (model, cond), rs in sorted(groups.items()):
        for r in rs:
            r.setdefault("utility_ref", um[r["probe_id"]].get("utility_ref", ""))
        m = measure_utility_group(rs, um, {"utility_anchor"})
        m.update(model=model, condition=cond)
        urows.append(m)
    A.write_csv(tables / "metrics_utility.csv", urows, ["model", "condition", "n_scored", "rouge_l_mean", "rouge_l_min", "rouge_l_max",
                                                        "length_sanity", "false_redaction_words_mean", "false_redaction_words_max"])
    return rows


def channels_v8(recs, span_map, ptype, tables):
    """V3 channels: V8 main and extended C7 records; C8 = frozen output filter applied to the same C7 generation."""
    acc = defaultdict(lambda: defaultdict(int))
    for rec in recs:
        if rec["condition"] != "C7" or rec["suite"] not in ("v8_main", "v8_extended"):
            continue
        root = "main" if rec["suite"] == "v8_main" else "ext"
        fam = "main" if root == "main" else FAM[ptype[rec["probe_id"]]]
        sp = span_map[rec["probe_id"]]
        o7 = rec.get("outputs_all") or [rec["output"]]
        for cond, outs in (("C7", o7), ("C8", [apply_output_filter(o)[0] for o in o7])):
            r2 = dict(rec, outputs_all=outs, output=outs[-1])
            s = A.score_record(r2, sp)
            d = acc[(root, rec["model"], fam, cond)]
            d["n_probes"] += 1; d["n_spans"] += s["n_entities"]
            d["full_spans"] += s["literal_full"]; d["paper_matcher_spans"] += s["legacy_full"]; d["full_probes"] += s["literal_full"] > 0
            d["partial_only_probes"] += (s["exposure_any"] and not s["literal_full"] and not s["encoded_full"])
            d["encoded_probes"] += s["encoded_full"] > 0
            d["input_echo_probes"] += s["encoded_input_echo"] > 0
            d["any_probes"] += bool(s["exposure_any"])
    cols = ["n_probes", "n_spans", "full_spans", "paper_matcher_spans", "full_probes", "partial_only_probes", "encoded_probes", "input_echo_probes", "any_probes"]
    write_rows(tables / "channels_v8.csv", ["run", "model", "family", "cond"] + cols,
               [list(k) + [acc[k][c] for c in cols] for k in sorted(acc)])


def t7_breakdown(recs, span_map, ptype, tables):
    cols = ["n_probes", "any_probes", "encoded_probes", "input_echo_probes", "partial_only_probes", "literal_full_probes",
            "full_and_encoded", "full_not_encoded", "encoded_not_echo", "echo_not_encoded", "any_fragment_probes", "unchanged_echo_probes", "partial_only_not_in_echo"]
    acc, surv = defaultdict(lambda: defaultdict(int)), defaultdict(int)
    for rec in recs:
        if rec["suite"] != "v8_extended" or rec["condition"] != "C7" or ptype[rec["probe_id"]] != "encoded":
            continue
        o7 = rec.get("outputs_all") or [rec["output"]]
        for cond, outs in (("C7", o7), ("C8", [apply_output_filter(o)[0] for o in o7])):
            r2 = dict(rec, outputs_all=outs, output=outs[-1])
            for sp in span_map[rec["probe_id"]]:
                for sch in {x for o in outs for x in encoded_hits(sp, o)}:
                    surv[(rec["model"], sp["type"], sch, cond)] += 1
            s = A.score_record(r2, span_map[rec["probe_id"]])
            d = acc[(rec["model"], cond)]
            full, enc, echo = s["literal_full"] > 0, s["encoded_full"] > 0, s["encoded_input_echo"] > 0
            part = bool(s["exposure_any"]) and not full and not enc
            d["n_probes"] += 1; d["any_probes"] += bool(s["exposure_any"]); d["encoded_probes"] += enc; d["input_echo_probes"] += echo
            d["partial_only_probes"] += part; d["literal_full_probes"] += full
            d["full_and_encoded"] += full and enc; d["full_not_encoded"] += full and not enc
            d["encoded_not_echo"] += enc and not echo; d["echo_not_encoded"] += echo and not enc
            d["any_fragment_probes"] += s["fragment_only"] > 0; d["unchanged_echo_probes"] += s["encoded_unchanged_echo"] > 0
            d["partial_only_not_in_echo"] += part and not echo
            assert (not enc) or s["exposure_any"] and (not part)  # invariant that makes the five columns readable
    write_rows(tables / "t7_channel_breakdown.csv", ["analysis_id", "run", "model", "family", "cond"] + cols,
               [["T11B_channels_v4", "v8_extended", m, "T7", c] + [acc[(m, c)][k] for k in cols] for (m, c) in sorted(acc)])
    write_rows(tables / "t7_encoded_scheme_survival.csv", ["analysis_id", "model", "entity_type", "scheme", "entities_C7", "entities_C8"],
               [["T11B_channels_v4", m, t, sch, surv[(m, t, sch, "C7")], surv[(m, t, sch, "C8")]] for (m, t, sch) in sorted({k[:3] for k in surv})])


def occurrence_coverage(recs, probes, tables):
    """Complete-literal survival of each entity occurrence in the stored C7 prompts (no detector re-run)."""
    RUN = {"c7c8_original": "original_10-03_main", "v8_main": "v8_corrected_main", "v8_extended": "v8_extended"}
    user_turns = lambda msgs: [m["content"] for m in msgs if m.get("role", "user") == "user"]
    sel = [r for r in recs if r["condition"] == "C7" and r["suite"] in RUN]
    agg, detail, seen = defaultdict(lambda: defaultdict(int)), [], defaultdict(set)
    v8_order = ["Qwen3-8B", "Llama-3.1-8B-Instruct", "Olmo-3-7B-Instruct", "gemma-4-E4B-it", "phi-4", "Mistral-7B-Instruct-v0.3"]  # campaign order (detail-row order only)
    key = lambda r: (0, MODELS.index(r["model"]), 0, r["probe_id"]) if r["suite"] == "c7c8_original" else (1, v8_order.index(r["model"]), r["suite"] == "v8_extended", r["probe_id"])
    for rec in sorted(sel, key=key):
        run, model, pid = RUN[rec["suite"]], rec["model"], rec["probe_id"]
        pr = probes[pid]; orig = [t["content"] for t in pr["turns"]]; red = user_turns(rec["prompt"])
        seen[(run, pid)].add(json.dumps(red))
        assert len(red) == len(orig), (run, model, pid)
        for s in pr["pii_spans"]:
            o_t = [t.count(s["text"]) for t in orig]; r_t = [t.count(s["text"]) for t in red]; O, R = sum(o_t), sum(r_t)
            if O == 0:
                continue  # encoded-input probes: plaintext not in the prompt
            k = "single" if O == 1 else "multi"
            cat = "all_masked" if R == 0 else ("none_masked" if R == O else "mixed")
            a = agg[(run, model, k)]; a["entities"] += 1; a["occurrences"] += O; a["intact_occurrences"] += R; a[cat] += 1
            if k == "multi":
                first_t = next(i for i, c in enumerate(o_t) if c)
                fs = "masked" if r_t[first_t] == 0 else ("intact" if r_t[first_t] == o_t[first_t] else "ambiguous")
                later = sum(r_t) - r_t[first_t]
                a["first_masked_later_intact"] += (fs == "masked" and later > 0); a["first_intact"] += (fs == "intact"); a["first_ambiguous"] += (fs == "ambiguous")
                if cat == "mixed" or (fs == "masked" and later > 0):
                    detail.append([run, model, pid, s["type"], O, R, fs, later])
    cols = ["entities", "occurrences", "intact_occurrences", "all_masked", "mixed", "none_masked", "first_masked_later_intact", "first_intact", "first_ambiguous"]
    write_rows(tables / "occurrence_coverage.csv", ["analysis_id", "run", "model", "entity_class"] + cols,
               [["occurrence_coverage_v1", run, model, k] + [agg[(run, model, k)][c] for c in cols] for (run, model, k) in sorted(agg)])
    write_rows(tables / "occurrence_coverage_detail.csv", ["run", "model", "probe_id", "entity_type", "orig_occurrences", "intact_occurrences",
                                                           "first_occurrence_status", "intact_in_later_occurrences"], detail)
    io = defaultdict(lambda: defaultdict(int))
    for rec in sel:
        if rec["suite"] == "v8_extended":
            continue
        run, model = RUN[rec["suite"]], rec["model"]
        pr = probes[rec["probe_id"]]; red = user_turns(rec["prompt"]); outs = rec.get("outputs_all") or [rec["output"]]
        for s in pr["pii_spans"]:
            intact = any(t.count(s["text"]) for t in red); leak = any(span_found_in_output(s, o) for o in outs)
            io[(run, model)][("intact" if intact else "not_intact") + "_" + ("leak" if leak else "noleak")] += 1
    ic = ["intact_leak", "intact_noleak", "not_intact_leak", "not_intact_noleak"]
    write_rows(tables / "input_intact_vs_output_leak.csv", ["analysis_id", "run", "model"] + ic,
               [["occurrence_coverage_v1"] + list(k) + [io[k][c] for c in ic] for k in sorted(io)])


def holdout_table(tables, rows_out):
    """Holdout: fresh score_exposure over each stored observable view, reconciled with the stored exposure field."""
    cases = holdout_cases()
    scope = lambda c: "authorized_return" if c["pii_fields"] else "prohibited" if c["suite"] in ("holdout", "stress") else "no_pii_requested"
    rows, mism, tot = [], 0, 0
    for rec, meta in A.iter_records(ROOT / "results/raw", suites={"holdout"}, with_metadata=True):
        c = cases[rec["probe_id"]]
        for view, outs in rec["observable_outputs"].items():
            fresh = score_exposure(c["entities"], outs); tot += 1; mism += fresh != meta["stored_exposure"][view]
            rows.append(dict(model=rec["model_repo"], seed=rec["seed"], case=rec["probe_id"], suite=c["suite"], family=c["family"],
                             scope=scope(c), group=c["template_group"], view=view, lit=int(fresh["literal_entity_exposure"]),
                             lit_b64=int(fresh["literal_entity_exposure"] or bool(fresh["base64_decoded_entities"]
                                         or fresh["base64_encoded_string_entities"] or fresh["split_recoverable_entities"]))))
            assert fresh["conversation_literal_or_base64_exposure"] == bool(fresh["literal_entity_exposure"] or fresh["base64_decoded_entities"] or fresh["split_recoverable_entities"])
    agg = lambda sel: (sum(r["lit"] for r in sel), sum(r["lit_b64"] for r in sel), len(sel))
    out = []
    for view in ("C0", "C2_unique", "C7", "C8"):
        for sc in ("ALL", "prohibited", "no_pii_requested", "authorized_return"):
            out.append([view, sc, *agg([r for r in rows if r["view"] == view and (sc == "ALL" or r["scope"] == sc)])])
    for m in sorted({r["model"] for r in rows}):
        for view in ("C0", "C2_unique", "C7", "C8"):
            out.append([view, "model:" + m, *agg([r for r in rows if r["view"] == view and r["model"] == m])])
    for fam in sorted({r["family"] for r in rows}):
        for view in ("C0", "C2_unique", "C7", "C8"):
            out.append([view, "family:" + fam, *agg([r for r in rows if r["view"] == view and r["family"] == fam])])
    bs = {sc: BA.holdout_cluster_bootstrap(rows, sc) for sc in ("prohibited", "no_pii_requested", "authorized_return")}
    with open(tables / "holdout_exposure.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["view", "scope", "literal_exposed", "literal_or_b64_or_split_exposed", "n_obs"]); w.writerows(out)
        w.writerow([]); w.writerow(["bootstrap_C8_minus_C7_literal", "scope", "estimate", "ci_lo", "ci_hi", "n_groups", "n_obs_per_view"])
        for sc, v in bs.items():
            w.writerow(["", sc, *v])
        w.writerow(["RECONCILE_fresh_vs_stored_views", "", tot, mism])
    for sc, (est, lo, hi, ng, n0) in bs.items():
        rows_out.append(["holdout", "all_models", f"C8-C7 literal exposure ({sc})", "", est, lo, hi, "template-group cluster bootstrap", BA.B, BA.SEED, ng])


def holdout_utility(tables):
    """Holdout task utility per model x family x view; fresh score_utility is asserted equal to the stored pre-restoration score."""
    cases = holdout_cases()
    UTIL_FAMILIES = ["classification", "routing", "action_extraction", "summary"]
    SEEDS = [42, 314, 2718]
    rows = []
    for rec, meta in A.iter_records(ROOT / "results/raw", suites={"holdout"}, with_metadata=True):
        case = cases[rec["probe_id"]]
        for view, outs in rec["observable_outputs"].items():
            u = meta["stored_utility"].get(view, {})
            if case.get("expected") is not None:
                assert score_utility(case, outs[-1]) == u["pre_restoration"], (rec["record_id"], view)
            pre, rcp = u.get("pre_restoration") or {}, u.get("authorized_recipient") or {}
            rows.append({"model": rec["model_repo"].split("/")[-1], "seed": rec["seed"], "family": case["family"], "view": view,
                         "strict_task_success": pre.get("strict_task_success"), "task_correct": pre.get("task_correct"),
                         "json_structure": pre.get("json_structure"), "content_preservation": pre.get("content_preservation"),
                         "refusal_heuristic": pre.get("refusal_heuristic"), "authorized_pii_correct": pre.get("authorized_pii_correct"),
                         "recipient_authorized_pii_correct": rcp.get("authorized_pii_correct")})

    def rate(sel, field):
        vals = [r[field] for r in sel if r[field] is not None]
        return sum(vals) / len(vals) if vals else None
    fields = ["strict_task_success", "task_correct", "json_structure", "content_preservation", "refusal_heuristic",
              "authorized_pii_correct", "recipient_authorized_pii_correct"]
    out = []
    for model in sorted({r["model"] for r in rows}):
        for fam in UTIL_FAMILIES + ["authorized_record", "pii_dependent_reasoning"]:
            for view in ["C0", "C2_unique", "C7", "C8"]:
                sel = [r for r in rows if r["model"] == model and r["family"] == fam and r["view"] == view]
                out.append([model, fam, view, len(sel)] + [rate(sel, f) for f in fields] +
                           [rate([r for r in sel if r["seed"] == sd], "strict_task_success") for sd in SEEDS])
    write_rows(tables / "holdout_utility_by_family.csv", ["model", "family", "view", "n"] + fields + [f"strict_task_success_s{sd}" for sd in SEEDS], out)


def processed(recs, span_map, tables, processed_dir, mc_rows, ci_rows, b):
    pl = list(A.probe_level_rows([r for r in recs if r["suite"] in A.PROBE_LEVEL_SUITES], span_map))
    write_rows(processed_dir / "probe_level_metrics.csv", PL_FIELDS, [[r[k] for k in PL_FIELDS] for r in pl])
    groups = defaultdict(list)
    for r in pl:
        groups[(r["suite"], r["model"], r["quant"], r["condition"])].append(r)
    cols = ["n_entities", "legacy_full", "numeric_partial", "literal_full", "fragment_only", "encoded_full", "encoded_input_echo",
            "encoded_unchanged_echo", "split_full_candidates", "exposed_entities"]
    agg_rows = []
    for k in sorted(groups):
        rs = groups[k]
        d = {c: sum(r[c] for r in rs) for c in cols}
        ne = d["n_entities"]
        agg_rows.append([*k, len(rs), ne] + [d[c] for c in cols[1:]] + [round(d["legacy_full"] / ne, 6), round(d["literal_full"] / ne, 6),
                         sum(r["exposure_any"] for r in rs), round(sum(r["exposure_any"] for r in rs) / len(rs), 6),
                         sum(r["refusal_any"] for r in rs), sum(r["refusal_exposure_same_turn"] for r in rs)])
        est = BA.bootstrap_ratio([(r["exposure_any"], 1) for r in rs], b=b)
        ci_rows.append([k[0], k[1], k[3], "exposure_probe_rate", est["estimate"], est["ci_low"], est["ci_high"], "probe bootstrap", b, BA.SEED, est["n_groups"]])
    write_rows(processed_dir / "aggregate_metrics.csv",
               ["suite", "model", "quant", "condition", "n_probes", "n_entities"] + cols[1:] +
               ["legacy_full_rate", "literal_full_rate", "exposure_probes", "exposure_probe_rate", "refusal_any_probes", "same_turn_refusal_exposure"], agg_rows)
    for r in mc_rows:
        for metric, pfx in (("leakage_rate", "leakage"), ("partial_leakage", "partial"), ("refusal_rate", "refusal")):
            if r[metric] is not None:
                ci_rows.append(["main", r["model"], r["condition"], metric, r[metric], r[f"{pfx}_ci_low"], r[f"{pfx}_ci_high"], "probe bootstrap", b, BA.SEED, r["n_probes"]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--b", type=int, default=BA.B)
    a = ap.parse_args(argv)
    out = Path(a.out)
    tables, processed_dir = out / "tables", out / "processed"
    tables.mkdir(parents=True, exist_ok=True); processed_dir.mkdir(parents=True, exist_ok=True)
    span_map, ptype = A.load_ground_truth(ROOT / "data")
    probes_main = list(A.iter_jsonl(ROOT / "data/probes_main.jsonl"))
    probes = {p["probe_id"]: dict(p, pii_spans=span_map[p["probe_id"]]) for name in ("probes_main", "probes_extended") for p in A.iter_jsonl(ROOT / f"data/{name}.jsonl")}
    recs = [r for r in A.iter_records(ROOT / "results/raw") if r["suite"] != "holdout"]
    ci_rows = []
    mc = main_tables(recs, span_map, ptype, probes_main, tables, a.b); print("main tables:", len(mc), "cells")
    channels_v8(recs, span_map, ptype, tables); print("channels_v8")
    t7_breakdown(recs, span_map, ptype, tables); print("t7 breakdown")
    occurrence_coverage(recs, probes, tables); print("occurrence coverage")
    processed(recs, span_map, tables, processed_dir, mc, ci_rows, a.b); print("processed: probe-level + aggregate")
    holdout_table(tables, ci_rows); print("holdout exposure")
    holdout_utility(tables); print("holdout utility")
    write_rows(processed_dir / "confidence_intervals.csv", ["table", "model", "condition", "metric", "estimate", "ci_low", "ci_high", "method", "B", "seed", "n_groups"], ci_rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
