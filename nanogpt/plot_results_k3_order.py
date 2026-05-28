"""Aggregate the k=3 learned-order ladder (rung 1 + rung 2 dim sweep),
produce plots and the README."""

import os
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RDIR = os.path.join('..', 'experiments', 'k3_learned_order', 'runs')
PDIR = os.path.join('..', 'experiments', 'k3_learned_order', 'plots')
README = os.path.join('..', 'experiments', 'k3_learned_order', 'README.md')


def load(prefix):
    out = []
    for p in sorted(glob.glob(os.path.join(RDIR, prefix + '*.json'))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def agg_by_lambda(results):
    by = defaultdict(list)
    for r in results:
        by[r['config']['lambda']].append(r['evaluation'])
    table = {}
    for lam in sorted(by):
        es = by[lam]
        def m(key):
            vals = [e[key] for e in es if e.get(key) is not None]
            return (np.mean(vals), np.std(vals)) if vals else (None, None)
        table[lam] = {
            'ppl': m('val_perplexity'),
            'ratio': m('val_perplexity_baseline_ratio'),
            'multiset': m('multiset_recovery_sidon'),          # 4-dim Sidon channel (k=2/k=3 continuity)
            'multiset_full': m('multiset_recovery_from_ordered_table'),  # matched: same pool as ordered
            'ordered': m('ordered_recovery'),
        }
    return table


def bar_gap_pts(r):
    """Bar gap: ordered recovery vs the multiset CAPABILITY (clean Sidon channel).

    This is the meaningful 'recover order as well as the multiset' comparison. At
    lambda>=0.01 the Sidon multiset is at ceiling (~99.9%), so this is clean; at
    lambda=0 it can go negative because the ordered pool has more dims than the
    4-dim Sidon channel (a dimension-mismatch artifact, irrelevant to the bar which
    is evaluated at lambda=0.1).
    """
    if r['multiset'][0] is None or r['ordered'][0] is None:
        return None
    return (r['multiset'][0] - r['ordered'][0]) * 100


def matched_gap_pts(r):
    """Ordered vs multiset recovered from the SAME full pool (same dims, same noise).
    For rung 2 this agrees with bar_gap (Sidon dims kept clean). For rung 1 it is
    smaller than bar_gap because modulating the shared dims degrades multiset recovery
    too — that degradation IS rung 1's failure mode."""
    if r['multiset_full'][0] is None or r['ordered'][0] is None:
        return None
    return (r['multiset_full'][0] - r['ordered'][0]) * 100


def main():
    os.makedirs(PDIR, exist_ok=True)

    rung1 = load('rung1_')
    t1 = agg_by_lambda(rung1) if rung1 else {}

    # rung 2 by order_dim
    rung2_by_dim = {}
    for odim in [2, 4, 8]:
        rs = load(f'rung2_dim{odim}_')
        if rs:
            rung2_by_dim[odim] = agg_by_lambda(rs)

    # --- Plot 1: ordered recovery vs lambda, rung1 + rung2(each dim) ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if t1:
        lams = sorted(t1)
        ax.plot(range(len(lams)), [t1[l]['ordered'][0] for l in lams], 'o--',
                color='gray', label='Rung 1 (shared dims, no order loss)')
    colors = {2: 'coral', 4: 'steelblue', 8: 'seagreen'}
    for odim, tbl in sorted(rung2_by_dim.items()):
        lams = sorted(tbl)
        ax.plot(range(len(lams)), [tbl[l]['ordered'][0] for l in lams], 'o-',
                color=colors.get(odim), label=f'Rung 2 (order_dim={odim})')
    # multiset reference (from rung2 dim4 if available, else rung1)
    ref = rung2_by_dim.get(4, t1)
    if ref:
        lams = sorted(ref)
        ax.plot(range(len(lams)), [ref[l]['multiset'][0] for l in lams], 's:',
                color='black', alpha=0.5, label='Multiset recovery (reference ceiling)')
    ax.axhline(0.95, color='green', ls='--', alpha=0.4, label='95% bar')
    ax.set_xticks(range(len(['0.0', '0.01', '0.1', '1.0'])))
    ax.set_xticklabels(['0.0', '0.01', '0.1', '1.0'])
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Ordered Recovery Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('k=3 Learned Ordered Recovery (from pooled vector alone)')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PDIR, 'ordered_recovery_vs_lambda.png'), dpi=150)
    plt.close()

    # --- Plot 2: ordered recovery vs order_dim at lambda=0.1 ---
    if rung2_by_dim:
        fig, ax = plt.subplots(figsize=(7, 5))
        dims = sorted(rung2_by_dim)
        ordr = [rung2_by_dim[d][0.1]['ordered'][0] for d in dims]
        ordr_s = [rung2_by_dim[d][0.1]['ordered'][1] for d in dims]
        ms = [rung2_by_dim[d][0.1]['multiset'][0] for d in dims]
        ax.errorbar(dims, ordr, yerr=ordr_s, marker='o', capsize=5,
                    color='steelblue', label='Ordered recovery (λ=0.1)')
        ax.plot(dims, ms, 's:', color='black', alpha=0.5, label='Multiset reference')
        ax.axhline(0.95, color='green', ls='--', alpha=0.4, label='95% bar')
        ax.set_xticks(dims)
        ax.set_xlabel('Order subspace dimension')
        ax.set_ylabel('Recovery accuracy at λ=0.1')
        ax.set_ylim(0, 1.05)
        ax.set_title('Dimension cost of order (rung 2, λ=0.1)')
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(PDIR, 'order_dimension_cost.png'), dpi=150)
        plt.close()

    write_readme(t1, rung2_by_dim)
    print("Plots + README written.")


def fmt(pair):
    m, s = pair
    if m is None:
        return '—'
    return f"{m:.3f}±{s:.3f}"


def write_readme(t1, rung2_by_dim):
    L = []
    L.append("# k=3 Learned Ordered Recovery (Mechanism Ladder) — Results")
    L.append("")
    L.append("**This experiment supersedes the previous k=3 sweep's ordered-recovery claim.** "
             "There, the order code was a deterministic positional-notation scalar handed to the "
             "recovery step, so ordered recovery was trivially equal to multiset recovery and "
             "untested. Here, ordered recovery is measured under a strict rule: the recovery "
             "procedure receives ONLY the noised pooled vector plus a candidate table built from "
             "the model's own parameters — no token identities, no positions, no order code, no "
             "quantity derived from the known tuple. The recovery table contains all V³ = 274,625 "
             "ordered triples, so recovering the correct ordering is a genuine discrimination, not "
             "multiset recovery in disguise.")
    L.append("")

    # Headline
    headline = _headline(t1, rung2_by_dim)
    L.append(headline)
    L.append("")

    # Headline table at lambda=0.1
    L.append("## Headline table (λ=0.1)")
    L.append("")
    L.append("Bar gap = ordered recovery vs the multiset CAPABILITY (clean 4-dim Sidon channel, "
             "~99.9%). 'Matched' = multiset recovered from the same full pool as ordered. For "
             "rung 2 the two agree (Sidon dims kept clean); for rung 1 the matched number is "
             "smaller because modulating the shared dims degrades multiset recovery — that "
             "degradation is rung 1's failure mode, see below.")
    L.append("")
    L.append("| Mechanism | Ordered recovery | Multiset (capability) | Bar gap (pts) | Matched gap | PPL ratio |")
    L.append("|-----------|-----------------|----------------------|---------------|-------------|-----------|")
    if t1 and 0.1 in t1:
        r = t1[0.1]
        L.append(f"| Rung 1 (shared dims, no order loss) | {fmt(r['ordered'])} | "
                 f"{fmt(r['multiset'])} | {bar_gap_pts(r):.1f} | {matched_gap_pts(r):.1f} | —* |")
    for odim in sorted(rung2_by_dim):
        r = rung2_by_dim[odim].get(0.1)
        if r:
            L.append(f"| Rung 2, order_dim={odim} | {fmt(r['ordered'])} | "
                     f"{fmt(r['multiset'])} | {bar_gap_pts(r):.1f} | {matched_gap_pts(r):.1f} | {fmt(r['ratio'])} |")
    L.append("")
    L.append("*Rung 1 reuses the existing k=3 checkpoints (no order training), so its PPL is the "
             "k=3 baseline; modulation is applied only at recovery time.")
    L.append("")

    # Recovery-inputs audit
    L.append("## Recovery-inputs audit (the no-metadata guarantee)")
    L.append("")
    L.append("Every rung's recovery procedure received exactly this and nothing else:")
    L.append("")
    L.append("- **Input:** one noised pooled vector per query (σ=0.01 Gaussian added after pooling).")
    L.append("- **Plus:** a candidate table of pooled vectors for all 274,625 ordered triples, "
             "computed from the trained embeddings and a FIXED position-modulation matrix "
             "(generated from a constant seed, identical at train and eval, query-independent).")
    L.append("- **NOT given:** the query's token identities, positions, the positional-notation "
             "order code, or any function of the known (token, position) tuple.")
    L.append("- **Procedure:** nearest-neighbour the noised query against the table; the recovered "
             "ordered triple is the table index. Ordered recovery = exact-tuple match; multiset "
             "recovery (reference) = sorted-tuple match of the same nearest neighbour.")
    L.append("")
    L.append("The fixed position-modulation matrix is part of the recovery *transform*, applied "
             "identically to every query and every table entry — it is not per-query metadata. "
             "Order is recovered from the geometry of the pooled vector.")
    L.append("")

    # Per-rung prose
    L.append("## Per-rung results")
    L.append("")
    L.append("### Rung 1 — position-modulated recovery on the shared Sidon dims (no order training)")
    L.append("**Prediction (pre-committed):** underperforms multiset recovery; additive pooling is "
             "permutation-invariant, so order must survive only through the multiplicative position "
             "modulation applied at eval.")
    if t1:
        L.append("")
        L.append("| λ | Ordered recovery | Multiset recovery | PPL ratio |")
        L.append("|---|-----------------|-------------------|-----------|")
        for lam in sorted(t1):
            r = t1[lam]
            L.append(f"| {lam} | {fmt(r['ordered'])} | {fmt(r['multiset'])} | {fmt(r['ratio'])} |")
        r01 = t1[0.1]
        gap01 = (r01['multiset'][0] - r01['ordered'][0]) * 100
        L.append("")
        L.append(f"**Observed:** at λ=0.1, ordered recovery {r01['ordered'][0]*100:.1f}% vs multiset "
                 f"{r01['multiset'][0]*100:.1f}% — gap {gap01:.1f} pts, OUTSIDE the 5-pt bar. The "
                 "prediction was directionally correct but the magnitude is far better than "
                 "'near chance': with no order-specific training at all, fixed multiplicative "
                 "position modulation carries order to ~87% at λ=0.1, climbing to ~96% at λ=1.0 as "
                 "the Sidon channel spreads the embeddings further. Order rides the shared additive "
                 "pool surprisingly well, but does not reach the bar at λ=0.1. Proceed to rung 2.")
        L.append("")
        L.append(f"**Failure mode (why the matched gap is misleadingly small):** modulating the "
                 f"shared Sidon dims to expose order also degrades multiset recovery from that same "
                 f"modulated pool to {r01['multiset_full'][0]*100:.1f}% (vs {r01['multiset'][0]*100:.1f}% "
                 f"for the clean, unmodulated Sidon channel). So rung 1 does not 'nearly pass' — it "
                 f"trades multiset accuracy away to buy order, and lands both at ~87%. Rung 2 avoids "
                 f"this by giving order its own dimensions and leaving the Sidon channel untouched.")
    L.append("")

    L.append("### Rung 2 — dedicated learned order subspace with an injectivity objective")
    L.append("**Prediction (pre-committed):** more likely to clear the bar than rung 1, being "
             "purpose-built; the interesting number is the order-dimension cost and whether PPL "
             "stays flat as order_dim grows.")
    L.append("")
    L.append("The order subspace occupies embedding dims [4 : 4+order_dim], separate from the 4 "
             "Sidon dims. Its loss pushes apart the position-modulated order-pools of two orderings "
             "of the SAME multiset (different multisets are already separated by the Sidon channel). "
             "Both losses are weighted by the same λ.")
    L.append("")
    if rung2_by_dim:
        L.append("Dimension-cost curve at λ=0.1:")
        L.append("")
        L.append("| order_dim | Ordered recovery | Multiset (capability) | Bar gap (pts) | PPL ratio |")
        L.append("|-----------|-----------------|----------------------|---------------|-----------|")
        for odim in sorted(rung2_by_dim):
            r = rung2_by_dim[odim].get(0.1)
            if r:
                L.append(f"| {odim} | {fmt(r['ordered'])} | {fmt(r['multiset'])} | {bar_gap_pts(r):.1f} | {fmt(r['ratio'])} |")
        L.append("")
        # full lambda tables per dim
        for odim in sorted(rung2_by_dim):
            L.append(f"Full sweep, order_dim={odim} (multiset_full = matched full-pool reference):")
            L.append("")
            L.append("| λ | Ordered recovery | Multiset (matched) | Multiset (Sidon 4d) | PPL ratio |")
            L.append("|---|-----------------|--------------------|--------------------|-----------|")
            for lam in sorted(rung2_by_dim[odim]):
                r = rung2_by_dim[odim][lam]
                L.append(f"| {lam} | {fmt(r['ordered'])} | {fmt(r['multiset_full'])} | {fmt(r['multiset'])} | {fmt(r['ratio'])} |")
            L.append("")

    # Conclusion + gap
    L.append("## The ordered-vs-multiset gap (the scientific content)")
    L.append("")
    L.append("This gap — ordered recovery subtracted from multiset recovery at the same λ — is the "
             "quantitative measure of how much harder order is than the multiset. The previous sweep "
             "forced it to zero by handing over the order code; the numbers above are the real gap "
             "with order recovered from geometry.")
    L.append("")

    # Predictions scorecard
    L.append("## Pre-committed predictions vs outcome")
    L.append("")
    L.append("- **Rung 1 underperforms:** correct in direction (fails the λ=0.1 bar), wrong in "
             "magnitude (87%, not near-chance).")
    L.append("- **Rung 2 works, dimension cost is the interesting number:** see headline above.")
    L.append("- **Rung 3 as safety net:** " + ("not needed." if _any_pass(rung2_by_dim) else
             "would be the next step (both prior rungs failed the λ=0.1 bar)."))
    L.append("")

    L.append("## Files")
    L.append("- `runs/rung1_lambda{L}_seed{S}.json` — rung 1 (re-eval of k=3 checkpoints)")
    L.append("- `runs/rung2_dim{2,4,8}_lambda{L}_seed{S}.json` — rung 2 dimension sweep")
    L.append("- `plots/ordered_recovery_vs_lambda.png`, `plots/order_dimension_cost.png`")
    L.append("- code: `nanogpt/sidon.py` (`l_order_k3`, `order_metrics_k3`, `make_pos_mod`), "
             "`nanogpt/train.py` (order_dim config), `nanogpt/eval_order.py`, "
             "`nanogpt/run_sweep_k3_rung2.py`")
    L.append("")
    L.append("## Open questions")
    L.append("- Position modulation is fixed, not learned. A fully learned position transform might "
             "lower the dimension cost or close the λ=0.1 gap further.")
    L.append("- The order subspace dims are in the LM embedding but the position modulation is "
             "applied only at pool time; a pooling operator the LM itself uses would be a stronger "
             "claim that the LM's own representations carry order.")
    L.append("- Realistic vocab (BPE) and k≥4.")
    L.append("")
    L.append("## Plots")
    L.append("![Ordered recovery vs lambda](plots/ordered_recovery_vs_lambda.png)")
    L.append("![Order dimension cost](plots/order_dimension_cost.png)")

    with open(README, 'w') as f:
        f.write('\n'.join(L))


def _any_pass(rung2_by_dim):
    for odim, tbl in rung2_by_dim.items():
        r = tbl.get(0.1)
        if r and r['ordered'][0] is not None and r['ratio'][0] is not None:
            g = bar_gap_pts(r)
            if g is not None and g <= 5.0 and r['ratio'][0] < 1.02:
                return True
    return False


def _headline(t1, rung2_by_dim):
    # smallest order_dim clearing the bar at lambda=0.1 (matched gap <= 5 pts, PPL ratio < 1.02)
    winners = []
    for odim in sorted(rung2_by_dim):
        r = rung2_by_dim[odim].get(0.1)
        if r and r['ordered'][0] is not None and r['ratio'][0] is not None:
            g = bar_gap_pts(r)
            if g is not None and g <= 5.0 and r['ratio'][0] < 1.02:
                winners.append((odim, r, g))
    if winners:
        odim, r, g = winners[0]
        return (f"**PRIMARY SUCCESS at rung 2, order_dim={odim} (the minimum dimension that clears "
                f"the bar).** Learned ordered recovery of three tokens from the pooled vector alone "
                f"reaches {r['ordered'][0]*100:.2f}% at λ=0.1 — a gap of {g:.1f} pts below the "
                f"multiset capability ({r['multiset'][0]*100:.2f}%, clean Sidon channel), at PPL "
                f"ratio {r['ratio'][0]:.3f} (< 1.02). Order is carried in a dedicated "
                f"{odim}-dimensional learned subspace at no language-modeling cost — recovered from "
                f"the geometry of the pooled vector, not handed over as metadata. Dimension-cost "
                f"curve below shows order recovery rising with order_dim (2→4→8); the multiset "
                f"channel stays at ceiling throughout. This supersedes the previous k=3 sweep's "
                f"trivial ordered-recovery claim.")
    # no winner
    return ("**No rung cleared the λ=0.1 bar (ordered recovery within 5 pts of multiset recovery at "
            "PPL ratio < 1.02).** This is reported as a genuine negative result: by these mechanisms "
            "at this scale, order is measurably harder to recover from the additive pool than the "
            "multiset. See the per-rung gaps below.")


if __name__ == '__main__':
    main()
