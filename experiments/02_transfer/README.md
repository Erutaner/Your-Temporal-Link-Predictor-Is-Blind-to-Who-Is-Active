# Transfer of the activity term to other models

**Paper content:** the transfer table in the main text (four other models, five streams, transductive test AP under
historical negative sampling, mean and standard deviation over seeds).

## What is measured

Our score is `f(u, v, t) + beta * log lambda_u(t)`: a pair score from the trunk plus a term that only depends on how
active the source node has been.  `lambda_u` is a fitted intensity with K + 1 parameters (K time scales), fitted on
the training stream by maximum likelihood; `beta` is chosen on the validation split.

The question here: does the term help other models too?  We take another model as it is (its released code, its
released configuration, its own checkpoint), keep its per-edge scores, and add `beta * log lambda_u(t)` with the very
same `lambda_u` parameters our runs fitted (`../../results/transfer/ground_params.json`, one set per stream; the code calls the activity process the "ground process"); only
`beta` is chosen again, on that model's validation scores.  Nothing of the other model is retrained.  The table shows
each model alone and with the term, next to ours.

The four other models are TPNet (as released, with the same-timestamp correction described in `baselines/README.md`),
TGN, DyGFormer (fixed-length padding) and DSRD (batch-composition correction).  A dagger in the table marks a model
whose numbers come from our retraining rather than from its paper.

## Rebuild the table from the shipped results

    python make_table.py                 # outputs/table_transfer_small (transductive, historical negatives)
    python make_table_appendix.py        # outputs/table_transfer_appendix (the remaining protocols and settings)
    python make_table_tpnet_streams.py   # outputs/table_transfer_tpnet_streams (TPNet on the eight other streams, one seed)

The first reads `results/transfer/<name>_transfer/<stream>/seed<S>_transfer.json` and the evaluation files of our main runs and
writes `outputs/table_transfer_small.tex` and `.md`.

The shipped TPNet runs cover all thirteen streams (the table shows the five of the study); the other three models were
run on those five.  Each `seed<S>_transfer.json` holds, per setting and negative-sampling protocol, the other model's own test AP / AUC
(`backbone_only`), the whole `beta` grid (`grid`) and the validation-selected entry (`selected`).

## Rerun from the data

1. Make the evaluation copies of the three code trees (the originals in `baselines/` are left untouched):

       python apply_transfer_overlays.py --work_dir <work>

   The overlays add a per-edge dump of the scored logits and a clean validation pass for the models that carry
   state between splits (TGN's memory, TPNet's projection state, DSRD's node state); the released metric code is not
   changed.

2. Per model, train it with its released configuration and add the term (all three protocols, both settings):

       python run_transfer.py --backbone TPNet     --data_root <data_root> --work_dir <work>
       python run_transfer.py --backbone TGN       --data_root <data_root> --work_dir <work>
       python run_transfer.py --backbone DyGFormer --data_root <data_root> --work_dir <work>
       python run_transfer.py --backbone DSRD      --data_root <data_root> --work_dir <work>

   Defaults: the five streams of the table (`wikipedia, reddit, uci, enron, lastfm`) and seeds 0, 1, 2.  Add
   `--skip_training` to reuse a checkpoint already in `<work>/trees/<tree>/saved_models/`.  The cost is the other
   model's own training time (minutes for UCI and Wikipedia, up to a few hours per seed for LastFM with DyGFormer or
   TGN); the readout step itself takes a minute per run.  Results land in
   `<work>/results/<name>_transfer/<stream>/seed<S>_transfer.json`; copy the `<name>_transfer` folders over the ones in
   a copy of `results/transfer/` and rebuild with `REPRO_RESULTS` pointing at that copy (see the top-level README).

3. Only if you retrained our own model and want to transfer *its* fitted parameters instead of the shipped ones:

       python export_ground_params.py --ckpt_dir <out_dir of 01_main_comparison/run_main.py> --out my_params.json

   and pass `--ground_json my_params.json` to `run_transfer.py`.

## What to check

`backbone_only` in a new transfer file should agree with the model's own result file from `01_main_comparison`
(same checkpoint, same evaluation code), and `selected` for `beta = 0` is the same number.  The gain from the term is
`selected` minus `backbone_only`.
