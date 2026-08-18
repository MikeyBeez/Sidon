# Sidon k=2 with address_dim=4 and Early Stopping — Specification

## Hypothesis

Two issues with the prior k=2 sweep can be addressed in a single follow-on:

1. **Geometric capacity** at `address_dim=2` left some pair-sums stuck close together (min pairwise distance ≈ 0.001 even at λ=1.0). Quadrupling the address subspace to `address_dim=4` should give the regularizer the room to push all pairs apart.

2. **Catastrophic overfit at 20k steps** invalidated the LM-cost comparison: lambda=0 baseline reached val_PPL=175 while train_PPL=1.10. The 5k sweep showed Sidon truly costs ~0% LM at a healthy training point. Replacing fixed-step training with early stopping based on val loss will produce honest LM-cost numbers.

Combined hypothesis: at `address_dim=4` with early stopping at val-loss minimum, at least one lambda achieves both:
- `val_perplexity_baseline_ratio < 1.05` (LM cost under 5%)
- `pair_recovery_accuracy_noisy_0.01 > 0.99` (geometry good enough for downstream noise)

Falsifiable: if no lambda achieves both, geometric capacity is not the limiting factor and the proposal needs structural revision (e.g., different loss formulation, position-modulated address contribution).

## Setup

**Codebase**: existing `~/Code/Sidon/nanogpt/` from the prior sweep. Modifications:
- `train.py`: add val-loss tracking every 250 steps; track best-val checkpoint; final-checkpoint = best-val (NOT last step).
- `sidon.py`: change `ADDRESS_SUBSPACE = slice(0, 4)` (was `slice(0, 2)`).
- Keep L_sidon's exhaustive enumeration over char-level pairs.

**Dataset**: char-level Tiny Shakespeare (vocab=65). Same train/val split as the prior sweep.

**Model**: identical to prior sweep (d_model=256, n_layers=6, n_heads=8, block_size=256, batch_size=64, AdamW lr=3e-4, weight_decay=0.1).

**Max training**: 10000 steps. Early stopping triggers when val_loss has not improved for 1500 consecutive steps (i.e., 6 consecutive val-loss measurements at the 250-step cadence). When triggered, training halts and the model is restored to the best-val checkpoint before evaluation.

**Held fixed across runs**: model, data, optimizer, early-stop patience, seeds {1337, 1338, 1339}.

**Varied across runs**: lambda ∈ {0, 0.01, 0.1, 1.0}.

Total runs: 4 × 3 = 12.

## Procedure

1. Modify `train.py` to:
   - Compute val_loss every 250 steps (use NanoGPT's `estimate_loss()` pattern over a held-out batch sample, or full-val if cheap).
   - Save the model state when val_loss improves.
   - Track patience counter; halt when patience exceeded.
   - At end of training, restore best-val state.
   - Record `best_step`, `best_val_loss`, `best_val_perplexity` in the output JSON.

2. Modify `sidon.py` to use `ADDRESS_SUBSPACE = slice(0, 4)`. Update any docstrings.

3. Run 12-run sweep (4 lambdas × 3 seeds). Save each result to `runs/char_dim4_lambdaX_seedY.json`.

4. Evaluate each run at the best-val checkpoint:
   - val_perplexity (recompute on full val set, not just an estimate)
   - sidon_satisfaction_rate (exhaustive over all distinct pair-of-pairs)
   - pair_recovery_accuracy_clean and _noisy_0.01
   - min and mean pairwise distance

5. Generate plots and README following prior conventions.

## Metrics (per run JSON, augmented)

```json
{
  "config": {
    "lambda": 0.1,
    "address_dim": 4,
    "gamma": 1.0,
    "vocab": "char",
    "seed": 1337,
    "max_steps": 10000,
    "early_stop_patience_steps": 1500
  },
  "training": {
    "best_step": 2750,
    "best_val_loss": 1.52,
    "best_val_perplexity": 4.57,
    "stopped_at_step": 4250,
    "stop_reason": "early_stop_patience",
    "final_lm_loss_at_best": 1.40,
    "final_sidon_loss_at_best": 0.18,
    "val_loss_curve_steps": [250, 500, ...],
    "val_loss_curve": [3.4, 2.6, ...]
  },
  "evaluation": {
    "val_perplexity": 4.57,
    "val_perplexity_baseline_ratio": 1.02,
    "sidon_satisfaction_rate": 0.995,
    "pair_recovery_accuracy_clean": 1.000,
    "pair_recovery_accuracy_noisy_0.01": 0.992,
    "min_pairwise_distance": 0.61,
    "mean_pairwise_distance": 3.82
  }
}
```

## Expected outcomes

- **λ = 0 (baseline)**: best-val PPL between 4.0 and 6.0 (the typical NanoGPT char-level neighborhood). Best step somewhere in 1500–4000 range. Sidon satisfaction ≈ 0 (no regularization).
  - *Interpretation*: establishes the honest baseline that the broken 20k sweep couldn't produce.

- **λ = 0.01**: best-val PPL within 1% of baseline. Sidon satisfaction modest (0.4–0.7). Recovery_noisy 0.6–0.8.
  - *Interpretation*: weak regularizer, useful for confirming the gradient pressure direction.

- **λ = 0.1**: best-val PPL within 3% of baseline. Sidon satisfaction > 0.97. Recovery_noisy > 0.99.
  - *Interpretation*: the load-bearing test. If this lands, the proposal is validated at k=2 with the corrected protocol.

- **λ = 1.0**: best-val PPL within 5–10% of baseline. Sidon satisfaction > 0.99. Recovery_noisy > 0.99.
  - *Interpretation*: strong regularizer. Reveals the LM/Sidon Pareto frontier at this address_dim.

**Key diagnostic to look for**: `min_pairwise_distance` should be > gamma (= 1.0) at λ ≥ 0.1. If `min` stays small even at address_dim=4, the issue is not capacity — it's the loss formulation (hinge cutoff stops pushing once `distance > gamma`, allowing pairs to settle at the boundary with no further repulsion).

**Falsification**:
- If no lambda achieves both PPL ratio < 1.05 AND recovery_noisy > 0.99, but `min_pairwise_distance` is still small: the loss formulation is the bottleneck. Next experiment uses a soft repulsion (e.g., `-log(distance)`) instead of hinge.
- If best-step values are highly variable across seeds (>2x range), the val_loss curve is noisy or the patience is too tight; widen to 3000 steps and retry.

## Files to create

```
~/Code/Sidon/
├── nanogpt/
│   ├── train.py        # modified for early stopping
│   ├── sidon.py        # ADDRESS_SUBSPACE updated to slice(0, 4)
│   └── eval.py         # unchanged
└── experiments/
    └── k2_dim4_earlystop/
        ├── plan.md     # this spec
        ├── runs/
        │   ├── char_dim4_lambda0.0_seed1337.json
        │   ├── ... (12 total)
        ├── README.md
        └── plots/
            ├── perplexity_vs_lambda.png
            ├── recovery_vs_lambda.png
            ├── tradeoff_frontier.png
            └── best_step_vs_lambda.png
```

## Success criteria

1. All 12 runs complete with JSON written.
2. README aggregates across seeds and produces:
   - Table: `λ → (best_val_ppl mean±std, PPL_ratio mean±std, sidon_sat mean±std, recovery_noisy mean±std, min_dist mean±std)`.
   - Per-lambda matched/not-matched against expected outcomes.
   - Conclusion: hypothesis confirmed, failed, or partially confirmed (with which axis failed).
3. New plot `best_step_vs_lambda.png` showing when each lambda's val-min occurred — if val-min systematically delays with higher lambda, the Sidon pressure is interacting with LM convergence.

## Runtime budget

- Each run: max 10k steps with early stop typically firing 3-5k. On 5070 Ti: 8-15 minutes per run.
- 12 runs: 2-3 hours total. Suitable for background execution.

## Stretch

If main result lands clean and time permits, run **two** quick ablations using λ=0.1 seed=1337 only (3 single runs):

1. `address_dim = 8`: does going further help, or does dim=4 already saturate the recovery? Establishes whether dim=4 was lucky or principled.

2. `gamma = 0.5`: does halving the margin help close the min_pairwise_distance issue, or does it just lower the bar?

3. **Soft repulsion loss** (only if min_pairwise_distance is still small): replace hinge with `-log(distance + epsilon)` and rerun λ=0.1 seed=1337. If min_pairwise_distance jumps significantly, the hinge cutoff was the real bottleneck and the paper's loss formulation should be updated.

These three single-runs are 30 minutes total and produce the diagnostic data for a clean writeup.

## Notes for the executing agent

- Early stopping must restore the best-val checkpoint, not stop training and use the final state. If you use the final state from the early-stop trigger point, you'll be evaluating the model after some post-best degradation.
- When computing val_loss every 250 steps, use full validation set if it fits in one batch on GPU (it should at this scale — Tiny Shakespeare's val split is tiny). Using a sampled estimate adds noise that can fool early stopping.
- The `min_pairwise_distance` metric is the load-bearing diagnostic for whether more address dimensions actually solved the geometric capacity problem. Report it prominently in the README.
- Compare `best_step` across lambdas: if higher lambda systematically delays best-step, that's evidence that Sidon regularization is acting like a regularizer in the standard sense (slowing overfit), which is independently interesting.
