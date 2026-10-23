# Leaderboard

Mean per item with 95% bootstrap intervals. Ranks are within each split; **FLIP** marks a run whose rank by the proxy differs from its rank by the true score.

## test_id

| Run | Proxy score | Proxy rank | True score | True rank | Fooled | |
|---|---|---:|---|---:|---:|---|
| hardened | 0.81 [0.74, 0.87] | 3 | 0.98 [0.95, 1.00] | 1 | 1% | **FLIP** |
| base | 0.95 [0.91, 0.98] | 2 | 0.72 [0.64, 0.81] | 2 | 20% |  |
| naive | 1.00 [1.00, 1.00] | 1 | 0.67 [0.58, 0.76] | 3 | 26% | **FLIP** |

## test_transfer

| Run | Proxy score | Proxy rank | True score | True rank | Fooled | |
|---|---|---:|---|---:|---:|---|
| hardened | 0.79 [0.71, 0.85] | 3 | 0.98 [0.96, 1.00] | 1 | 0% | **FLIP** |
| base | 0.95 [0.91, 0.99] | 2 | 0.70 [0.61, 0.79] | 2 | 22% |  |
| naive | 0.98 [0.95, 1.00] | 1 | 0.65 [0.55, 0.73] | 3 | 25% | **FLIP** |

## test_boss

| Run | Proxy score | Proxy rank | True score | True rank | Fooled | |
|---|---|---:|---|---:|---:|---|
| naive | 1.00 [1.00, 1.00] | 3 | 0.21 [0.13, 0.29] | 1 | 76% | **FLIP** |
| base | 1.00 [1.00, 1.00] | 1 | 0.17 [0.10, 0.23] | 2 | 80% | **FLIP** |
| hardened | 1.00 [1.00, 1.00] | 2 | 0.14 [0.08, 0.21] | 3 | 86% | **FLIP** |
