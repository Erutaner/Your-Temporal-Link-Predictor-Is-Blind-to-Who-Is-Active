| score | Wikipedia | Reddit | UCI | Enron | LastFM |
|---|---|---|---|---|---|
| DyGFormer† | 76.01±0.51 | 81.32±1.35 | 81.59±0.61 | 76.60±0.54 | 81.61 |
| TPNet | 81.55±4.10 | 81.02±1.31 | 86.34±0.80 | 80.79±1.68 | 87.74±0.50 |
| f only | 88.71±0.80 | 82.74±1.44 | 91.47±0.52 | 84.04±2.43 | 84.60±1.20 |
| log λ_u only | 90.67±0.01 | 80.65±0.01 | 91.77±0.01 | 83.59±0.00 | 93.19±0.00 |
| f + β·log λ_u | **93.25±0.05** | **90.16±0.17** | **96.04±0.17** | **90.26±0.77** | **95.36±0.03** |

Transductive test AP (x100), historical negative sampling, mean ± std over seeds (3 for our rows and DyGFormer†, 1 on LastFM; 5 otherwise). log λ_u alone = f zeroed, β = 1: the source's event rate with no pair information. † = our retrain with the corrected released code. Bold = best.
