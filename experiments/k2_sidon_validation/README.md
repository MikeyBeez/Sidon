# k=2 Sidon Validation — Results

## Sweep Summary

| Lambda | Val PPL (mean±std) | PPL Ratio | Sidon Sat | Recovery (clean) | Recovery (noisy) |
|--------|-------------------|-----------|-----------|-----------------|-----------------|
| 0.0 | 182.57±5.75 | 1.005±0.032 | 0.0000±0.0000 | 0.9997±0.0004 | 0.1997±0.0230 |
| 0.01 | 188.46±5.03 | 1.037±0.028 | 0.6224±0.0086 | 1.0000±0.0000 | 0.8036±0.0154 |
| 0.1 | 195.17±4.18 | 1.074±0.023 | 0.8811±0.0022 | 1.0000±0.0000 | 0.9467±0.0118 |
| 1.0 | 206.60±6.96 | 1.137±0.038 | 0.9580±0.0010 | 1.0000±0.0000 | 0.9764±0.0140 |

## Per-Lambda Analysis

### Lambda = 0.0
**Expected**: Baseline — establishes reference perplexity. Sidon satisfaction expected 0.3–0.7.
**Observed**: Baseline perplexity = 182.57. Sidon satisfaction = 0.0000.
**Prediction MATCHED.**

### Lambda = 0.01
**Expected**: Weak regularizer — perplexity unchanged, modest Sidon improvement (0.7–0.9).
**Observed**: Sidon satisfaction = 0.6224.
**Prediction MATCHED.**

### Lambda = 0.1
**Expected**: Load-bearing test — perplexity within 5% of baseline, satisfaction >0.99, recovery >0.99.
**Observed**: PPL ratio = 1.074, recovery (noisy) = 0.9467.
**Prediction DID NOT MATCH.**

### Lambda = 1.0
**Expected**: Strong regularizer — perplexity may degrade 10%+, satisfaction near 1.0.
**Observed**: Satisfaction = 0.9580.
**Prediction DID NOT MATCH.**

## Conclusion

**Hypothesis FAILED at this configuration**: λ=0.1 did not achieve both <5% perplexity ratio and >99% pair recovery. Next step: try address_dim=4.

## Plots

![Perplexity vs Lambda](plots/perplexity_vs_lambda.png)
![Recovery vs Lambda](plots/recovery_vs_lambda.png)
![Tradeoff Frontier](plots/tradeoff_frontier.png)