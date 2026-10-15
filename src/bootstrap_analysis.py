"""Bootstrap / interval statistics.

Moved unchanged: bootstrap_ci and _stable_salt (probe-level rate CIs, pipeline aggregate), joint_table, zero_event_upper,
bootstrap_ratio, rotated_references (exposure reanalysis). holdout_cluster_bootstrap is the template-group cluster
bootstrap of the holdout C8-C7 literal-exposure difference (formerly inline in the holdout recompute script).
"""
from __future__ import annotations

import collections
import hashlib
import random
from collections import defaultdict

from measure_partial_exposure import fragment_values

B = 2000
SEED = 42

def _stable_salt(s: str) -> int:
    """Deterministic, process-independent per-string salt. Replaces hash(),
    which is randomized per Python process (PYTHONHASHSEED unset) and would
    make the bootstrap CIs irreproducible across runs."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def bootstrap_ci(flags, idx, rng, b=B, seed=SEED):
    """Bootstrap 95% CI of a per-probe rate. flags: list of (num, denom) or
    (num, None) where None = not applicable."""
    nums = [f[0] for f in flags]
    dens = [f[1] for f in flags]
    valid = [(n, d) for n, d in zip(nums, dens) if d]
    if not valid:
        return None, None, None
    n = len(valid)
    obs = sum(n for n, _ in valid) / sum(d for _, d in valid)

    def sample_rate():
        tot_n = tot_d = 0
        for i in range(n):
            j = rng.randrange(n)
            tot_n += valid[j][0]
            tot_d += valid[j][1]
        return tot_n / tot_d

    samples = [sample_rate() for _ in range(b)]
    samples.sort()
    lo = samples[int(0.025 * b)]
    hi = samples[int(0.975 * b) - 1]
    return obs, lo, hi


def joint_table(rows, exposure="exposure_any"):
    cells = [0, 0, 0, 0]
    for row in rows:
        cells[(0 if row["refusal_any"] else 2) + (0 if row[exposure] else 1)] += 1
    a, b, c, d = cells
    return {"n_aligned": len(rows), "refusal_exposure": a, "refusal_no_exposure": b,
            "no_refusal_exposure": c, "neither": d,
            "p_exposure_given_refusal": a / (a + b) if a + b else None,
            "p_exposure_given_no_refusal": c / (c + d) if c + d else None}


def zero_event_upper(n, alpha=0.05):
    """Exact one-sided binomial bound ONLY under independent exchangeable probe trials."""
    if n <= 0 or not 0 < alpha < 1:
        raise ValueError("n must be positive; alpha must be in (0,1)")
    return 1 - alpha ** (1 / n)


def bootstrap_ratio(rows, groups=None, b=2000, seed=42):
    if not rows or b < 40:
        raise ValueError("Nonempty rows and at least 40 resamples required")
    groups = list(range(len(rows))) if groups is None else groups
    if len(groups) != len(rows) or any(den <= 0 for _, den in rows):
        raise ValueError("Group length mismatch or nonpositive denominator")
    clustered = defaultdict(lambda: [0, 0])
    for (num, den), group in zip(rows, groups):
        clustered[group][0] += num
        clustered[group][1] += den
    values = list(clustered.values())
    n = len(values)
    rng = random.Random(seed)
    samples = []
    for _ in range(b):
        draw = rng.choices(values, k=n)
        samples.append(sum(v[0] for v in draw) / sum(v[1] for v in draw))
    samples.sort()
    return {"estimate": sum(v[0] for v in rows) / sum(v[1] for v in rows),
            "ci_low": samples[int(0.025 * b)], "ci_high": samples[int(0.975 * b) - 1],
            "n_groups": n, "ci_informative": n > 1 and samples[0] != samples[-1], "b": b}


def rotated_references(probes, rotation=1, disjoint_fragments=False):
    """Type-preserving foreign references; exclude repeated exact truth values.

    Shared fragments are retained on purpose to measure non-specificity. This
    is not an estimated false-positive rate for arbitrary real identities.
    The disjoint sensitivity excludes overlapping candidate fragments and
    omits entities when the existing pool has no eligible foreign value.
    """
    pool = defaultdict(set)
    for probe in probes:
        for s in probe["pii_spans"]:
            pool[s["type"]].add(s["text"])
    pools = {kind: sorted(values) for kind, values in pool.items()}
    result = {}
    for probe in probes:
        own = {(s["type"], s["text"]) for s in probe["pii_spans"]}
        own_folded = {(kind, value.casefold()) for kind, value in own}
        own_fragments = defaultdict(set)
        for kind, value in own:
            own_fragments[kind].update(v.casefold() for _, v in fragment_values(kind, value))
        chosen = []
        used = set()
        for kind, value in sorted(own):
            values = pools[kind]
            start = values.index(value)
            for step in range(len(values)):
                candidate = values[(start + rotation + step) % len(values)]
                shared = own_fragments[kind] & {v.casefold() for _, v in fragment_values(kind, candidate)}
                if ((kind, candidate.casefold()) not in own_folded and (kind, candidate) not in used
                        and not (disjoint_fragments and shared)):
                    chosen.append({"type": kind, "text": candidate})
                    used.add((kind, candidate))
                    break
            else:
                if not disjoint_fragments:
                    raise ValueError(f"No foreign {kind} value for {probe['probe_id']}")
        result[probe["probe_id"]] = chosen
    return result


def holdout_cluster_bootstrap(rows, scope, B=2000, seed=42):
    """C8-C7 literal-exposure difference on one permission scope; resamples template_group clusters."""
    d = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r["scope"] == scope and r["view"] in ("C7", "C8"):
            x = d[r["group"]]; x[2] += r["view"] == "C7"; x[0] += r["lit"] if r["view"] == "C7" else 0; x[1] += r["lit"] if r["view"] == "C8" else 0
    g = list(d); rng = random.Random(seed); ests = []
    for _ in range(B):
        s = [d[rng.choice(g)] for _ in g]; n = sum(x[2] for x in s)
        ests.append((sum(x[1] for x in s) - sum(x[0] for x in s)) / n)
    ests.sort(); n0 = sum(x[2] for x in d.values())
    return (sum(x[1] - x[0] for x in d.values()) / n0, ests[int(.025 * B)], ests[int(.975 * B) - 1], len(g), n0)
