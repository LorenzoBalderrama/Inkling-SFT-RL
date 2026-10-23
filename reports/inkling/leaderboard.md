# Leaderboard

Mean per item with 95% bootstrap intervals. Ranks are within each split; **FLIP** marks a run whose rank by the proxy differs from its rank by the true score.

## test_transfer

| Run | Proxy score | Proxy rank | True score | True rank | Fooled | |
|---|---|---:|---|---:|---:|---|
| inkling-sft-rl | 0.80 [0.73, 0.87] | 2 | 1.00 [1.00, 1.00] | 1 | 0% | **FLIP** |
| inkling-base | 0.80 [0.74, 0.86] | 1 | 0.88 [0.81, 0.94] | 2 | 0% | **FLIP** |
| inkling-sft | 0.71 [0.64, 0.77] | 3 | 0.85 [0.80, 0.91] | 3 | 0% |  |

## test_boss

| Run | Proxy score | Proxy rank | True score | True rank | Fooled | |
|---|---|---:|---|---:|---:|---|
| inkling-sft-rl | 0.76 [0.70, 0.83] | 2 | 0.90 [0.84, 0.95] | 1 | 3% | **FLIP** |
| inkling-sft | 0.64 [0.57, 0.72] | 3 | 0.79 [0.71, 0.86] | 2 | 0% | **FLIP** |
| inkling-base | 0.99 [0.98, 1.00] | 1 | 0.04 [0.01, 0.07] | 3 | 94% | **FLIP** |

## test_injection

| Run | Proxy score | Proxy rank | True score | True rank | Fooled | |
|---|---|---:|---|---:|---:|---|
| inkling-sft-rl | 0.90 [0.85, 0.95] | 2 | 0.86 [0.79, 0.92] | 1 | 13% | **FLIP** |
| inkling-sft | 0.64 [0.57, 0.70] | 3 | 0.77 [0.69, 0.83] | 2 | 2% | **FLIP** |
| inkling-base | 0.92 [0.86, 0.96] | 1 | 0.58 [0.49, 0.67] | 3 | 32% | **FLIP** |
