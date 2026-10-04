| term b | Wikipedia | Reddit | UCI | Enron | LastFM |
|---|---|---|---|---|---|
| none (f alone, β=0) | 86.19±1.21 | 90.62±0.27 | 83.35±0.29 | 85.32±0.85 | 73.96±0.95 |
| log λ_u (SNAM)* | 91.23±0.09 | 91.79±0.13 | 88.18±0.04 | **89.04**±0.18 | 89.82±0.71 |
| log(1+n_u^K) | 88.31±0.23 | 91.75±0.16 | 83.35±0.29 | 86.50±0.78 | 82.27±0.85 |
| log(1+n_u^{K/2}) | 89.44±0.44 | 91.26±0.19 | 86.85±0.48 | 88.93±0.39 | 85.23±0.89 |
| −log(1+Δt_u) | 90.39±0.32 | 91.44±0.24 | **88.66**±0.07 | 87.47±0.47 | **89.93**±0.70 |
| MLP on log(1+n_u) | **91.34**±0.11 | **91.86**±0.13 | 88.33±0.06 | 87.84±0.43 | **89.93**±0.73 |

Transductive test AP (x100) under inductive negatives, mean ± std over 3 seeds; score f + β·b with b replaced on the same checkpoints, β chosen on validation (under random negatives every variant selects β = 0). * = main results. Bold = best.
