# Sidon Address-Subspace — k=2 Validation Specification

## Hypothesis

SGD can train a small dedicated subspace (the "address subspace") of token embeddings to satisfy a pairwise Sidon distinctness property — for all distinct unordered token pairs `{a,b} ≠ {c,d}`, the address-subspace sums `e_a^addr + e_b^addr` and `e_c^addr + e_d^addr` are separated by at least margin `gamma` — without significantly degrading language modeling perplexity.

Falsifiable form: at lambda values where the Sidon regularizer converges (pairwise separation > gamma for >99% of distinct pairs), validation perplexity stays within 5% of the lambda=0 baseline on Tiny Shakespeare.

## Setup

**Codebase**: NanoGPT (https://github.com/karpathy/nanoGPT) cloned as the base. Modifications kept minimal: add an address-subspace partition of the embedding, an L_sidon loss term, and per-run evaluation hooks.

**Datasets**:
- Char-level Tiny Shakespeare (vocab ~65). Pairs are exhaustively enumerable (~2k pairs).
- BPE Tiny Shakespeare (vocab ~10k). Pairs sampled (~50M pairs total — sample 10k per training step).

**Model** (small enough to iterate fast):
- d_model = 256
- n_layers = 6
- n_heads = 8
- d_ff = 1024
- block_size = 256
- Standard NanoGPT defaults otherwise

**Embedding partition**:
- Total dim 256
- Address subspace: dims [0, 2) — first 2 dimensions
- Semantic subspace: dims [2, 256) — remaining 254
- The partition is fixed by construction; gradients flow to both regions, but L_sidon only sees the address subspace.

**Held fixed across runs**: model architecture, data, optimizer (AdamW, lr=3e-4, weight_decay=0.1), batch size (64), training steps (5000), seed set (3 seeds: 1337, 1338, 1339).

**Varied across runs**: lambda ∈ {0, 0.01, 0.1, 1.0}.

## Procedure

1. Clone NanoGPT into `~/Code/Sidon/nanogpt/`. Verify baseline training runs.

2. Add `sidon.py` containing:
   - `address_subspace_indices = slice(0, 2)`
   - `def l_sidon(embedding_table, num_samples=10000, gamma=1.0, k=2):` — samples `num_samples` pairs of distinct unordered token pairs `((a,b), (c,d))` where `{a,b} ≠ {c,d}`, computes `dist = ||e_a[addr] + e_b[addr] - e_c[addr] - e_d[addr]||_2`, returns `mean(max(0, gamma - dist))`. For char-level Tiny Shakespeare, exhaustively enumerate instead of sampling.

3. Modify NanoGPT's training loop to compute `loss = loss_lm + lambda * l_sidon(model.transformer.wte.weight)` at each step.

4. Run the lambda sweep on char-level Tiny Shakespeare first (smaller, faster). Then repeat on BPE Tiny Shakespeare if char-level lands clean.

5. After training, run `eval.py` for each run:
   - Compute `val_perplexity` on held-out 10% split.
   - Compute Sidon satisfaction rate: for all distinct pair-of-pairs, fraction where the address-subspace sum distance exceeds gamma.
   - Compute pair recovery accuracy: build a table of `(a,b) → sum(a,b)[addr]` for all distinct unordered pairs. For each pair, perturb the sum with N(0, 0.01) noise (representing downstream usage), query nearest-neighbor in the table, measure top-1 accuracy.

6. Write per-run results to JSON. Write phase README summarizing the sweep.

## Metrics (per run, JSON format)

```json
{
  "config": {
    "lambda": 0.1,
    "address_dim": 2,
    "gamma": 1.0,
    "vocab": "char",
    "seed": 1337,
    "steps": 5000
  },
  "training": {
    "final_train_loss": 1.234,
    "final_lm_loss": 1.230,
    "final_sidon_loss": 0.040,
    "loss_curve": ["..."],
    "sidon_curve": ["..."]
  },
  "evaluation": {
    "val_perplexity": 4.21,
    "val_perplexity_baseline_ratio": 1.02,
    "sidon_satisfaction_rate": 0.997,
    "pair_recovery_accuracy_clean": 1.000,
    "pair_recovery_accuracy_noisy_0.01": 0.989
  }
}
```

## Expected outcomes

- **lambda = 0 (baseline)**: val_perplexity = baseline. Sidon satisfaction rate: somewhere in 0.3-0.7 (natural separation due to embedding diversity, but not guaranteed). Pair recovery: low without the constraint.
  - *Interpretation*: confirms baseline behavior. If satisfaction is already >0.95, the address dim is too generous or the vocab is too small for the constraint to be meaningful.

- **lambda = 0.01**: val_perplexity essentially unchanged. Sidon satisfaction improves modestly (0.7-0.9 range). Pair recovery improves but not to ceiling.
  - *Interpretation*: regularizer is too weak to fully shape geometry. Useful to confirm gradient pressure exists in the right direction.

- **lambda = 0.1**: val_perplexity within 5% of baseline. Sidon satisfaction >0.99. Pair recovery >0.99.
  - *Interpretation*: the load-bearing hypothesis is confirmed. SGD finds Sidon geometry at low LM cost.

- **lambda = 1.0**: val_perplexity may degrade (10%+). Sidon satisfaction near 1.0.
  - *Interpretation*: regularizer dominates. Confirms strong-form trainability; reveals the LM/Sidon tradeoff frontier.

**Falsification conditions**:
- If no lambda achieves both `val_perplexity_baseline_ratio < 1.05` AND `pair_recovery_accuracy > 0.99`, the hypothesis fails at this configuration. Try address_dim = 4 before declaring failure. If still failing, the proposal needs structural revision.

## Files to create

```
~/Code/Sidon/
├── nanogpt/                              # cloned, lightly modified
│   ├── model.py                          # original
│   ├── train.py                          # modified: + L_sidon term
│   ├── sidon.py                          # NEW: L_sidon implementation
│   └── eval.py                           # NEW: Sidon + recovery evaluation
├── data/
│   ├── tiny_shakespeare_char.bin
│   └── tiny_shakespeare_bpe.bin
├── experiments/
│   └── k2_sidon_validation/
│       ├── plan.md                       # this spec, copied in
│       ├── runs/
│       │   ├── char_lambda0.0_seed1337.json
│       │   ├── char_lambda0.01_seed1337.json
│       │   ├── ... (12 char runs: 4 lambda × 3 seeds)
│       │   └── bpe_lambda0.1_seed1337.json  # if char results land
│       ├── README.md                     # phase summary
│       └── plots/
│           ├── perplexity_vs_lambda.png
│           ├── recovery_vs_lambda.png
│           └── tradeoff_frontier.png
└── requirements.txt
```

## Success criteria

The phase is complete when:
1. All 12 char-level runs (4 lambdas × 3 seeds) finish with results JSON written.
2. The README in `experiments/k2_sidon_validation/` summarizes the sweep with:
   - A table of `lambda → (val_ppl_ratio, sidon_rate, recovery_acc)` aggregated across seeds (mean ± std).
   - One sentence per lambda saying whether the predicted outcome matched.
   - A bottom-line conclusion: hypothesis confirmed / failed / inconclusive.
3. The three plots are generated.

BPE-level runs are conditional: only proceed if char-level shows `lambda=0.1` matches the predicted outcome.

## Runtime budget

- Single char-level run on 5070 Ti: ~10-20 min for 5000 steps at this model size.
- 12 char runs: ~3-4 hours.
- BPE runs (12 if proceeding): ~6-8 hours.
- Total budget: 1 day of GPU time, runnable as background sessions.

If the executing agent finds steps taking dramatically longer, stop and report — likely indicates a configuration issue.

## Stretch

If main result lands clean and time permits:

1. **k=3 Sidon**: extend `l_sidon` to sample triples. Address dim still 2 (information budget tight at vocab 10k but should fit). Validate that pairwise + triple-wise Sidon co-exist.

2. **Positional notation order encoding**: add a single dim for `tok_a · V + tok_b` (k=2) injected into the embedding at pool-feed time. Validate that downstream LM doesn't break when one dim of the input has very large magnitude. (This is testing the engineering of the positional-notation injection, not the Sidon claim.)

3. **Joint Sidon + positional notation recovery**: implement a minimal pool (just gated sum, no full V4 architecture) and check that the pool output supports both multiset recovery (via Sidon dims) and order recovery (via positional notation dim).

The stretch experiments validate the full ordered-pair recovery scheme, but the main experiment (Sidon at k=2) is the load-bearing one. Land that first; everything else builds on it.

## Notes

- The `gamma=1.0` margin is a guess. If the regularizer fails to converge at any lambda, try `gamma=0.5` before declaring failure — overly large margins can cause optimization issues at small address_dim.
- `address_dim=2` even though the information-theoretic minimum is ~1 dim for k=2. Margin matters during training; 1 dim might be too tight for SGD to find the geometry. If 2 dims work cleanly, dropping to 1 is a follow-up.
- Char-level Tiny Shakespeare has a vocab of ~65 — small enough that the pairwise Sidon constraint is computable exhaustively (~2000 distinct pairs). That's a feature: no sampling-strategy confound in the first experiment. BPE is the harder test that comes second.
