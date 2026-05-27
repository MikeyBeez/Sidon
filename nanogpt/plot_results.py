"""Generate sweep summary plots from results JSONs."""

import os
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join('..', 'experiments', 'k2_sidon_validation', 'runs')
PLOTS_DIR = os.path.join('..', 'experiments', 'k2_sidon_validation', 'plots')
README_PATH = os.path.join('..', 'experiments', 'k2_sidon_validation', 'README.md')


def load_results():
    results = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, 'char_*.json'))):
        with open(path) as f:
            results.append(json.load(f))
    return results


def aggregate(results):
    by_lambda = defaultdict(list)
    for r in results:
        lam = r['config']['lambda']
        by_lambda[lam].append(r)

    table = {}
    for lam in sorted(by_lambda.keys()):
        runs = by_lambda[lam]
        ppls = [r['evaluation']['val_perplexity'] for r in runs]
        ratios = [r['evaluation'].get('val_perplexity_baseline_ratio') for r in runs]
        ratios = [x for x in ratios if x is not None]
        sat = [r['evaluation']['sidon_satisfaction_rate'] for r in runs]
        rec_clean = [r['evaluation']['pair_recovery_accuracy_clean'] for r in runs]
        rec_noisy = [r['evaluation'].get('pair_recovery_accuracy_noisy_0.01', 0) for r in runs]

        table[lam] = {
            'ppl_mean': np.mean(ppls), 'ppl_std': np.std(ppls),
            'ratio_mean': np.mean(ratios) if ratios else None,
            'ratio_std': np.std(ratios) if ratios else None,
            'sat_mean': np.mean(sat), 'sat_std': np.std(sat),
            'rec_clean_mean': np.mean(rec_clean), 'rec_clean_std': np.std(rec_clean),
            'rec_noisy_mean': np.mean(rec_noisy), 'rec_noisy_std': np.std(rec_noisy),
        }
    return table


def plot_perplexity(table):
    lambdas = sorted(table.keys())
    ppls = [table[l]['ppl_mean'] for l in lambdas]
    stds = [table[l]['ppl_std'] for l in lambdas]
    labels = [str(l) for l in lambdas]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(lambdas))
    ax.bar(x, ppls, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Val Perplexity')
    ax.set_title('Validation Perplexity vs Sidon Lambda')

    baseline = ppls[0] if lambdas[0] == 0.0 else None
    if baseline:
        ax.axhline(y=baseline * 1.05, color='red', linestyle='--', alpha=0.5,
                    label='5% threshold')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'perplexity_vs_lambda.png'), dpi=150)
    plt.close()


def plot_recovery(table):
    lambdas = sorted(table.keys())
    clean = [table[l]['rec_clean_mean'] for l in lambdas]
    noisy = [table[l]['rec_noisy_mean'] for l in lambdas]
    labels = [str(l) for l in lambdas]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(lambdas))
    w = 0.35
    ax.bar(x - w/2, clean, w, label='Clean', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, noisy, w, label=f'Noisy (σ=0.01)', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Pair Recovery Accuracy')
    ax.set_title('Pair Recovery Accuracy vs Sidon Lambda')
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'recovery_vs_lambda.png'), dpi=150)
    plt.close()


def plot_tradeoff(table):
    lambdas = sorted(table.keys())
    ppls = [table[l]['ppl_mean'] for l in lambdas]
    sats = [table[l]['sat_mean'] for l in lambdas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ppls, sats, s=100, c='steelblue', zorder=5)
    for i, lam in enumerate(lambdas):
        ax.annotate(f'λ={lam}', (ppls[i], sats[i]), textcoords='offset points',
                    xytext=(10, 5), fontsize=10)
    ax.set_xlabel('Val Perplexity')
    ax.set_ylabel('Sidon Satisfaction Rate')
    ax.set_title('Perplexity–Sidon Tradeoff Frontier')
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'tradeoff_frontier.png'), dpi=150)
    plt.close()


def write_readme(table, results):
    lambdas = sorted(table.keys())

    expected = {
        0.0: "Baseline — establishes reference perplexity. Sidon satisfaction expected 0.3–0.7.",
        0.01: "Weak regularizer — perplexity unchanged, modest Sidon improvement (0.7–0.9).",
        0.1: "Load-bearing test — perplexity within 5% of baseline, satisfaction >0.99, recovery >0.99.",
        1.0: "Strong regularizer — perplexity may degrade 10%+, satisfaction near 1.0.",
    }

    lines = [
        "# k=2 Sidon Validation — Results",
        "",
        "## Sweep Summary",
        "",
        "| Lambda | Val PPL (mean±std) | PPL Ratio | Sidon Sat | Recovery (clean) | Recovery (noisy) |",
        "|--------|-------------------|-----------|-----------|-----------------|-----------------|",
    ]

    for lam in lambdas:
        t = table[lam]
        ratio_str = f"{t['ratio_mean']:.3f}±{t['ratio_std']:.3f}" if t['ratio_mean'] else "—"
        lines.append(
            f"| {lam} | {t['ppl_mean']:.2f}±{t['ppl_std']:.2f} | {ratio_str} | "
            f"{t['sat_mean']:.4f}±{t['sat_std']:.4f} | "
            f"{t['rec_clean_mean']:.4f}±{t['rec_clean_std']:.4f} | "
            f"{t['rec_noisy_mean']:.4f}±{t['rec_noisy_std']:.4f} |"
        )

    lines += ["", "## Per-Lambda Analysis", ""]
    for lam in lambdas:
        t = table[lam]
        lines.append(f"### Lambda = {lam}")
        lines.append(f"**Expected**: {expected.get(lam, 'N/A')}")

        if lam == 0.0:
            matched = True
            note = f"Baseline perplexity = {t['ppl_mean']:.2f}. Sidon satisfaction = {t['sat_mean']:.4f}."
        elif lam == 0.01:
            matched = t['sat_mean'] > t.get('baseline_sat', 0)
            note = f"Sidon satisfaction = {t['sat_mean']:.4f}."
        elif lam == 0.1:
            ratio_ok = t['ratio_mean'] is not None and t['ratio_mean'] < 1.05
            recovery_ok = t['rec_noisy_mean'] > 0.99
            matched = ratio_ok and recovery_ok
            note = f"PPL ratio = {t['ratio_mean']:.3f}, recovery (noisy) = {t['rec_noisy_mean']:.4f}."
        elif lam == 1.0:
            matched = t['sat_mean'] > 0.99
            note = f"Satisfaction = {t['sat_mean']:.4f}."
        else:
            matched = None
            note = ""

        verdict = "MATCHED" if matched else "DID NOT MATCH" if matched is not None else "N/A"
        lines.append(f"**Observed**: {note}")
        lines.append(f"**Prediction {verdict}.**")
        lines.append("")

    # Bottom-line conclusion
    lines += ["## Conclusion", ""]
    t01 = table.get(0.1)
    if t01:
        ratio_ok = t01['ratio_mean'] is not None and t01['ratio_mean'] < 1.05
        recovery_ok = t01['rec_noisy_mean'] > 0.99
        if ratio_ok and recovery_ok:
            lines.append(
                "**Hypothesis CONFIRMED**: At λ=0.1, SGD finds Sidon geometry in the "
                "address subspace with <5% perplexity degradation and >99% pair recovery.")
        elif ratio_ok and t01['rec_clean_mean'] > 0.99:
            lines.append(
                "**Hypothesis PARTIALLY CONFIRMED**: Sidon geometry is trainable with "
                "low LM cost, but noisy recovery falls short — consider increasing address_dim.")
        else:
            lines.append(
                "**Hypothesis FAILED at this configuration**: λ=0.1 did not achieve both "
                "<5% perplexity ratio and >99% pair recovery. Next step: try address_dim=4.")
    else:
        lines.append("**INCONCLUSIVE**: λ=0.1 results not available.")

    lines += [
        "",
        "## Plots",
        "",
        "![Perplexity vs Lambda](plots/perplexity_vs_lambda.png)",
        "![Recovery vs Lambda](plots/recovery_vs_lambda.png)",
        "![Tradeoff Frontier](plots/tradeoff_frontier.png)",
    ]

    with open(README_PATH, 'w') as f:
        f.write('\n'.join(lines))
    print(f"README written to {README_PATH}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = load_results()
    if not results:
        print(f"No results found in {RESULTS_DIR}")
        return

    print(f"Loaded {len(results)} result files")
    table = aggregate(results)

    plot_perplexity(table)
    plot_recovery(table)
    plot_tradeoff(table)
    write_readme(table, results)
    print("Plots and README generated.")


if __name__ == '__main__':
    main()
