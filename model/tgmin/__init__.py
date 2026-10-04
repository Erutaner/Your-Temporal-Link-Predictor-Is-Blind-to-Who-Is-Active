# -*- coding: utf-8 -*-
"""tgmin: the scale-free continuous-time state trunk for temporal
interaction graphs with a closed-form ground process.

  tgmin/sfs.py      feeder (index structure of the stream), SFSModel (state
                    on the chunk clock, within-chunk records in scalar
                    channels, readout), GroundProcess (lambda_u by exact
                    likelihood, closed-form compensator), training and the
                    three evaluation protocols, brute-force oracles.
  tgmin/data.py     dataset loading, chunking, edge features, bundles.
  tgmin/readout.py  certified readout head.
  tgmin/vendor/     DyGLib's split, negative samplers and metrics (the
                    evaluation protocol; unmodified).
  scripts/train_sfs.py   train / evaluate one stream.
  tools/            oracle selftests, device / batch-invariance check,
                    checkpoint recheck."""
