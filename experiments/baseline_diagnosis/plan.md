# LM Baseline Diagnosis — Specification

## Hypothesis

The lambda=0 baseline reported val PPL = 182.6 at 20k steps on character-level Tiny Shakespeare. This is **worse than uniform-random guessing** (PPL ≈ 65 for vocab=65), which is not a possible outcome of correct training + correct evaluation. The cause is one of three possibilities:

1. **Eval pipeline is broken** — val PPL is being computed incorrectly (wrong split, wrong reduction, wrong base for perplexity, eval-vs-train mode confusion).
2. **Training pipeline is broken** — the model itself produces incoherent output by 20k steps despite low train loss (severe data leakage between train/val, or a numerical issue corrupting weights).
3. **Catastrophic overfitting plus another factor** — pure overfitting alone cannot push val PPL above uniform-random; if val PPL really is 182, something else is amplifying it (e.g., dropout still on during eval, layer-norm stats wrong, eval on a corrupted split).

The diagnostic is designed to distinguish these. Falsifiable form: at least one of the four checks below will produce a clear pointer to the cause; if all four come back clean and val PPL is still 182, the underlying problem is novel and requires deeper investigation.

## Setup

**Codebase**: same `~/Code/Sidon/nanogpt/` as the previous sweep. No new training runs of the model itself — this diagnostic re-uses the existing 20k-step lambda=0 checkpoint if it was saved, otherwise re-trains a single lambda=0 baseline with the original config but with extra logging.

**Model config**: identical to the previous sweep (d_model=256, n_layers=6, n_heads=8, block_size=256, batch_size=64, lr=3e-4).

**Held fixed**: lambda=0 (no Sidon regularizer in any of the diagnostic runs), seed=1337, char-level Tiny Shakespeare.

**Varied**: nothing — this is a single-condition diagnostic with four observations.

## Procedure

If the 20k checkpoint was saved, load it and skip to step 3. Otherwise:

1. Re-train lambda=0 baseline for 20k steps, **logging val_perplexity every 500 steps** (not just at the end). Save model checkpoint at 5k, 10k, 15k, 20k steps. Also save train_loss every 100 steps.

2. After training, save the full `val_perplexity_curve` to `diagnostics/val_ppl_curve.json` with fields `{steps: [...], val_ppl: [...], train_loss_at_those_steps: [...]}`.

3. **Check A — Sample generation**: load the 20k-step checkpoint. Generate 3 samples of 500 characters each at temperature 0.8, starting from a 10-character prompt drawn from the training set. Save to `diagnostics/samples_20k.txt`. **Interpretation**: if samples are coherent English / pseudo-Shakespeare, the model is fine and eval is broken. If samples are mostly the same character repeated, or random noise, the model itself is broken.

4. **Check B — Independent eval recompute**: write a fresh standalone `recompute_val_ppl.py` that loads the val tensor independently (re-read `data/tiny_shakespeare_char.bin`, take the val 10% split with the documented offset, NOT through any cached state), puts the model in `eval()` mode explicitly, computes val loss with `F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction='mean')` over all val tokens, and reports `exp(mean_loss)`. Compare against the training loop's reported val_perplexity. Save to `diagnostics/independent_val_ppl.json` with fields `{training_loop_reported: 182.6, independent_recompute: <value>, match: bool}`.

5. **Check C — Data split sanity**: print to `diagnostics/data_split_info.txt`:
   - `len(train_data)`, `len(val_data)` (in tokens)
   - `train_data[:50]`, `val_data[:50]` (first 50 tokens of each, as decoded text)
   - `set(train_data) == set(val_data)` (do both splits use the same vocab characters?)
   - `len(set(train_data).intersection(set(val_data))) / 65` (overlap)
   - Verify val_data is held out: check that `val_data[:1000]` is **not** found as a substring of the training text. If it is, the splits are leaking.

6. **Check D — Val PPL curve shape**: from the curve saved in step 2, identify:
   - Step at which val_ppl is minimum (`min_step`, `min_val_ppl`)
   - Whether the curve climbs **smoothly** from `min_val_ppl` to 182 over remaining steps (suggests catastrophic overfit), or **hockey-sticks** at a specific step (suggests numerical / state corruption at that step).
   - Save analysis to `diagnostics/curve_shape.txt`.

## Metrics

Outputs land in `~/Code/Sidon/experiments/baseline_diagnosis/diagnostics/`:

```
diagnostics/
├── val_ppl_curve.json          # full curve, every 500 steps
├── val_ppl_curve.png           # plot
├── samples_20k.txt             # 3 generated samples
├── independent_val_ppl.json    # recomputed val PPL
├── data_split_info.txt         # split sanity check
└── curve_shape.txt             # analysis of curve shape
```

## Expected outcomes

Four observations × three possible causes give a small diagnostic matrix. The likely patterns:

- **Cause = eval broken**: samples are coherent, independent recompute differs from reported 182, val PPL curve in step 2 may already show the problem if logged consistently the right way.
  - *Action*: fix the eval code, re-run the full 20k sweep with corrected eval.

- **Cause = training broken**: samples are incoherent, independent recompute also returns ~182 (or worse), val PPL curve shows the model genuinely failed to learn.
  - *Action*: investigate training (likely data leakage or numerical issue), then re-run.

- **Cause = real catastrophic overfitting**: samples are *too* coherent (verbatim training data), independent recompute returns ~182, val PPL curve shows a clear minimum around 3k-5k steps then climbs smoothly.
  - *Action*: use the minimum-val-PPL checkpoint (around 3k-5k) as the actual baseline for Sidon comparisons. Stop training all sweeps at the val-min, not at 20k. Re-run the lambda sweep with this corrected stopping rule.

- **Cause = none of the above (all checks clean)**: samples coherent, independent recompute matches, splits are clean, curve shape is smooth and bounded. In this case val PPL of 182 wouldn't be reproducible, and the original report may have been a transient anomaly.
  - *Action*: re-run the lambda=0 baseline once more for replication. If it lands at a sensible value, the previous 182 was a one-off.

## Success criteria

The diagnostic is complete when:

1. All four check outputs exist as files in `diagnostics/`.
2. A single one-paragraph summary in `diagnostics/README.md` states the most likely cause from {eval broken, training broken, real overfit, replication issue} with one sentence of evidence per check.
3. A specific recommendation for the next sweep is named: which value to use as baseline, what stopping rule, what (if any) code change.

## Runtime budget

- If the 20k checkpoint is saved: diagnostic checks take ~5-10 minutes total.
- If re-training is required: ~20-40 minutes for the single lambda=0 run + diagnostics.

## Out of scope

Do **not** run another lambda sweep as part of this diagnostic. Do **not** try address_dim=4 or gamma=0.5. Those decisions depend on what this diagnostic reveals — running them blind risks generating another set of results that compare against a broken baseline and waste another hour.

## Notes for the executing agent

- The "worse than uniform random" check is the load-bearing observation that motivated this diagnostic. Loss > ln(vocab_size) means the model is assigning probability mass to wrong characters in a way that's worse than ignoring its inputs entirely. If you find this is actually the case after independent recompute, that's the surprising fact that needs to be explained — don't normalize past it.
- If the samples at 20k look like verbatim training-set text (the model memorized exact sequences), report this explicitly. That'd be diagnostic of a particular kind of overfit (memorization rather than feature collapse).
