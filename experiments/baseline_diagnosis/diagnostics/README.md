# Baseline Diagnosis — Results

## Summary

**Most likely cause: Real catastrophic overfitting.**

The 4.7M parameter model memorized the 1M character training set over 20k steps, achieving
train loss 0.10 (train PPL ~1.1) while val loss reached 5.2 (val PPL ~182). This is NOT
an eval bug — the independent recompute confirms it. The model assigns concentrated probability
mass to memorized training continuations that don't match val text, which is penalized more
severely than uniform uncertainty, hence val PPL > 65.

**Evidence per check:**
- **Check A (samples)**: Model generates coherent pseudo-Shakespeare text — it HAS learned language, but from memorization
- **Check B (independent eval)**: All methods agree on val PPL ~176, confirming the eval pipeline is correct
- **Check C (data splits)**: Splits are clean — sequential 90/10 split, no leakage
- **Check D (curve shape)**: Train loss drops to 0.10 while val loss climbs past ln(65) — textbook catastrophic overfit

**Recommendation for next sweep:**
1. Use early stopping at the val-loss minimum (likely around steps 2k-5k based on the 5k-step sweep where val PPL was ~7.8)
2. OR reduce max_iters back to 5000 where the model hadn't yet overfit
3. OR add dropout (0.1-0.2) to prevent memorization at longer training
4. The 5k-step results (val PPL ~7.9) are the valid baseline — the 20k-step results should be discarded
