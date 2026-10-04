| stream | prefix | train edges | test positives | epoch time (s) | s per 100k edges | scoring time (s) | s per 100k positives |
|---|---|---|---|---|---|---|---|
| LastFM | 10% | 74k | 19k | 12.6 | 17.1 | 1.3 | 6.8 |
|  | 30% | 222k | 58k | 12.8 | 5.8 | 1.7 | 2.8 |
|  | 100% | 723k | 194k | 21.7 | 3.0 | 7.1 | 3.7 |
| Flights | 10% | 104k | 24k | 0.5 | 0.5 | 0.5 | 1.9 |
|  | 30% | 328k | 84k | 2.5 | 0.8 | 0.9 | 1.0 |
|  | 100% | 1.10M | 288k | 13.8 | 1.2 | 2.9 | 1.0 |

Prefixes of the stream, each split 70/15/15 on its own; one RTX 5090, median of 3 runs. LastFM keeps C = 192 training blocks at every length; Flights uses one block per time step (5 / 17 / 90).
