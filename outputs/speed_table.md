# Speed benchmark: one exclusive RTX 5090, batch 200, median of 3 repeats

Released models: per-batch times from 300 training / up to 500 evaluation timed batches after a warm-up, multiplied by the number of batches of the split (extrapolated where the split is longer than the timed window); ours: full epoch and full test-split pass measured. Evaluation = random negatives, transductive test split.

## Test-split scoring time [s]

| model | UCI | Wikipedia | Enron | Reddit | LastFM | Flights |
|---|---|---|---|---|---|---|
| JODIE | 0.3 | 1.7 | 0.4 | 9.6 | 6.7 | 28.0 |
| DyRep | 0.5 | 2.2 | 0.8 | 11.6 | 10.1 | 33.5 |
| TGAT | 7.4 | 19.0 | 15.0 | 149.3 | 160.1 | 243.6 |
| TGN | 0.5 | 2.2 | 0.8 | 11.8 | 10.4 | 33.2 |
| CAWN | 14.3 | 26.1 | 22.2 | 174.3 | 565.5 | 477.2 |
| TCL | 0.5 | 1.3 | 1.0 | 9.4 | 10.9 | 16.6 |
| GraphMixer | 3.8 | 9.9 | 7.9 | 41.5 | 82.9 | 120.1 |
| DyGFormer | 2.3 | 5.9 | 13.0 | 28.4 | 159.7 | 210.0 |
| EdgeBank | 0.6 | 3.2 | 1.4 | 68.1 | 138.2 | 670.0 |
| TPNet | 0.5 | 1.3 | 0.9 | 5.2 | 10.1 | 15.1 |
| DSRD | 0.5 | 1.2 | 0.9 | 5.2 | 10.0 | 14.8 |
| ours | 1.0 | 1.0 | 1.2 | 3.9 | 5.9 | 2.5 |

## Training time per epoch [s]

| model | UCI | Wikipedia | Enron | Reddit | LastFM | Flights |
|---|---|---|---|---|---|---|
| JODIE | 2 | 8 | 4 | 48 | 48 | 138 |
| DyRep | 3 | 10 | 4 | 55 | 52 | 156 |
| TGAT | 33 | 79 | 69 | 576 | 701 | 978 |
| TGN | 5 | 15 | 10 | 76 | 109 | 221 |
| CAWN | 58 | 98 | 88 | 493 | 2173 | 1850 |
| TCL | 10 | 24 | 21 | 123 | 212 | 323 |
| GraphMixer | 17 | 39 | 34 | 188 | 328 | 501 |
| DyGFormer | 16 | 38 | 106 | 185 | 1136 | 1198 |
| EdgeBank | — | — | — | — | — | — |
| TPNet | 7 | 19 | 16 | 92 | 174 | 263 |
| DSRD | 10 | 22 | 18 | 106 | 193 | 248 |
| ours | 12 | 6 | 13 | 14 | 19 | 12 |

## Scored pairs per second during the test split

| model | UCI | Wikipedia | Enron | Reddit | LastFM | Flights |
|---|---|---|---|---|---|---|
| JODIE | 59100 | 28000 | 85679 | 20951 | 58064 | 20601 |
| DyRep | 38890 | 21386 | 49900 | 17392 | 38249 | 17199 |
| TGAT | 2426 | 2509 | 2505 | 1353 | 2424 | 2364 |
| TGN | 38466 | 21516 | 49449 | 17095 | 37197 | 17324 |
| CAWN | 1261 | 1824 | 1697 | 1159 | 686 | 1207 |
| TCL | 34998 | 35963 | 35814 | 21536 | 35464 | 34772 |
| GraphMixer | 4681 | 4786 | 4770 | 4865 | 4682 | 4795 |
| DyGFormer | 7925 | 8093 | 2883 | 7118 | 2430 | 2743 |
| EdgeBank | 30563 | 14853 | 26795 | 2968 | 2808 | 860 |
| TPNet | 39758 | 38006 | 39717 | 39046 | 38367 | 38182 |
| DSRD | 38326 | 38346 | 39739 | 39014 | 38714 | 38823 |
| ours | 18165 | 46756 | 30950 | 51495 | 65299 | 229771 |

## Peak GPU memory during evaluation [MB]

| model | UCI | Wikipedia | Enron | Reddit | LastFM | Flights |
|---|---|---|---|---|---|---|
| JODIE | 110 | 315 | 124 | 880 | 1005 | 1793 |
| DyRep | 127 | 317 | 147 | 882 | 1021 | 1795 |
| TGAT | 506 | 575 | 548 | 914 | 1315 | 1738 |
| TGN | 128 | 360 | 148 | 945 | 1022 | 1857 |
| CAWN | 1052 | 640 | 612 | 979 | 2824 | 2285 |
| TCL | 108 | 177 | 149 | 516 | 917 | 1340 |
| GraphMixer | 717 | 788 | 759 | 1123 | 1524 | 1950 |
| DyGFormer | 214 | 283 | 484 | 646 | 1638 | 1675 |
| EdgeBank | 0 | 0 | 0 | 0 | 0 | 0 |
| TPNet | 206 | 309 | 253 | 679 | 1046 | 1525 |
| DSRD | 602 | 1004 | 570 | 1443 | 1422 | 2370 |
| ours | 239 | 1402 | 129 | 3818 | 1114 | 2450 |

## Peak GPU memory during training [MB]

| model | UCI | Wikipedia | Enron | Reddit | LastFM | Flights |
|---|---|---|---|---|---|---|
| JODIE | 202 | 413 | 136 | 778 | 948 | 1626 |
| DyRep | 236 | 442 | 170 | 805 | 981 | 1654 |
| TGAT | 1959 | 2027 | 2000 | 2366 | 2768 | 3191 |
| TGN | 283 | 550 | 201 | 965 | 1022 | 1784 |
| CAWN | 4675 | 2429 | 2399 | 2773 | 10212 | 5727 |
| TCL | 779 | 1098 | 1071 | 1437 | 1839 | 2262 |
| GraphMixer | 1059 | 1292 | 1101 | 1303 | 1687 | 2292 |
| DyGFormer | 1012 | 1081 | 1586 | 1513 | 3210 | 2893 |
| EdgeBank | — | — | — | — | — | — |
| TPNet | 509 | 591 | 548 | 937 | 1319 | 1768 |
| DSRD | 1197 | 1895 | 1860 | 1917 | 2608 | 2825 |
| ours | 265 | 1463 | 217 | 4786 | 1447 | 4235 |

## Test-split time relative to ours (x slower)

| model | UCI | Wikipedia | Enron | Reddit | LastFM | Flights |
|---|---|---|---|---|---|---|
| JODIE | 0.3x | 1.7x | 0.4x | 2.5x | 1.1x | 11.2x |
| DyRep | 0.5x | 2.2x | 0.6x | 3.0x | 1.7x | 13.4x |
| TGAT | 7.5x | 18.8x | 12.4x | 38.1x | 26.9x | 97.2x |
| TGN | 0.5x | 2.2x | 0.6x | 3.0x | 1.8x | 13.3x |
| CAWN | 14.4x | 25.8x | 18.3x | 44.5x | 95.2x | 190.5x |
| TCL | 0.5x | 1.3x | 0.9x | 2.4x | 1.8x | 6.6x |
| GraphMixer | 3.9x | 9.8x | 6.5x | 10.6x | 13.9x | 47.9x |
| DyGFormer | 2.3x | 5.8x | 10.7x | 7.2x | 26.9x | 83.8x |
| EdgeBank | 0.6x | 3.2x | 1.2x | 17.4x | 23.3x | 267.4x |
| TPNet | 0.5x | 1.2x | 0.8x | 1.3x | 1.7x | 6.0x |
| DSRD | 0.5x | 1.2x | 0.8x | 1.3x | 1.7x | 5.9x |

