| term b | Wikipedia | Reddit | UCI | Enron | LastFM |
|---|---|---|---|---|---|
| *Transductive setting, historical negatives* | | | | | |
| none (f alone, β=0) | 88.71±0.80 | 82.74±1.44 | 91.47±0.52 | 84.04±2.43 | 84.60±1.20 |
| log λ_u (SNAM)* | 93.25±0.05 | 90.16±0.17 | 96.04±0.17 | 90.26±0.77 | 95.36±0.03 |
| log(1+n_u^K) | 89.03±0.54 | 85.25±0.61 | 91.47±0.52 | 84.03±2.43 | 87.15±0.54 |
| log(1+n_u^{K/2}) | 91.67±0.19 | 83.92±1.23 | 95.03±0.34 | 88.98±1.56 | 92.71±0.32 |
| −log(1+Δt_u) | 93.33±0.04 | 90.29±0.21 | 96.02±0.08 | 88.96±0.89 | 95.30±0.03 |
| MLP on log(1+n_u) | **93.42**±0.08 | **90.51**±0.15 | **96.22**±0.08 | **90.55**±0.88 | **95.61**±0.02 |
| *Inductive setting, historical negatives* | | | | | |
| none (f alone, β=0) | 77.78±1.86 | 67.54±2.74 | 84.08±0.35 | 78.56±2.85 | 78.38±0.70 |
| log λ_u (SNAM)* | **88.29**±0.13 | 79.20±0.34 | 89.03±0.05 | **85.57**±0.91 | 90.12±0.61 |
| log(1+n_u^K) | 78.47±1.67 | 71.19±1.46 | 84.08±0.35 | 78.96±2.76 | 79.72±0.23 |
| log(1+n_u^{K/2}) | 84.85±0.76 | 69.50±2.27 | 87.20±0.52 | 83.88±1.54 | 86.70±0.67 |
| −log(1+Δt_u) | 88.15±0.08 | 79.37±0.36 | **89.51**±0.03 | 83.83±0.57 | **90.50**±0.59 |
| MLP on log(1+n_u) | 88.10±0.15 | **79.65**±0.31 | 89.26±0.08 | 85.32±1.05 | 90.41±0.62 |

Test AP (x100), mean ± std over 3 seeds; score f + β·b with b replaced on the same checkpoints, β chosen on validation (under random negatives every variant selects β = 0). In the inductive setting the two samplers coincide, so only the historical protocol is shown. * = main results. Bold = best within a block.
