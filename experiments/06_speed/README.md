# Speed and cost

**Paper content:** the LastFM speed figure in the main text (training time per epoch and scoring time of the test
split against test AP, twelve models); the full-page appendix figure with the same panels for UCI, Wikipedia,
Enron, Reddit and Flights; the three-panel cost figure (cost of our model against its own settings); the
stream-length table (LastFM and Flights cut to 10 / 30 / 100 % of their events).  `make_table_all_models.py` also
writes the underlying timing table of all twelve models.

## How the time was measured

One GPU (an RTX 5090) with nothing else running on it, batches of 200 events, three repeats, the median reported.

* **Released models** (JODIE, DyRep, TGAT, TGN, CAWN, TCL, GraphMixer, DyGFormer, EdgeBank, TPNet, DSRD): their own
  training and evaluation scripts, with a timing hook inserted by `build_timing_patches.py`.  Training: after a
  warm-up, 300 training batches are timed and the epoch time is 300-batch time scaled to the number of batches of
  an epoch.  Scoring: after a warm-up, up to 500 test batches (random negatives) are timed and scaled to the number
  of batches of the test split.  TPNet is timed as released; DSRD with the row correction; DyGFormer with the
  fixed-length padding.
* **Our model**: one full training epoch measured (the second of two) and one full scoring pass over the test
  split; no extrapolation.

The hook is inert unless the environment variables `SPEED_TRAIN_BATCHES` / `SPEED_EVAL_BATCHES` (released models)
or `SFS_SPEED` (ours) are set, so the patched trees behave like the originals otherwise.

In file names the three parts carry the labels S1 (all models: `results/speed/results.jsonl`), S2 (stream length) and
S3 (our settings); the last two share `results/speed/results_s23.jsonl` (field `exp`) and give `outputs/table_s2_stream_length`
and `outputs/fig_cost_s3`.

## Rebuild the figures and tables from the shipped results

    python make_table_all_models.py     # outputs/speed_table.md (per-model, per-stream seconds and memory)
    python make_figure_main.py          # outputs/fig_speed_lastfm.pdf / .png
    python make_figure_appendix.py      # outputs/fig_speed_appendix.pdf / .png
    python make_figure_cost.py          # outputs/fig_cost_s3.pdf / .png
    python make_table_stream_length.py  # outputs/table_s2_stream_length.tex / .md

Inputs: `results/speed/results.jsonl` (one record per model, stream, repeat and phase; `measure` holds the timed
seconds, ms per batch, the number of batches of the split and the peak GPU memory) and
`results/speed/results_s23.jsonl` (the same for our model's setting and stream-length runs).  The y axis of the
speed figures is the transductive test AP under historical negatives from the main tables.  The label positions in
the speed figures are set by hand in the scripts.

## Rerun on your GPU

1. Build the timed copies of the three released trees (written to `patched/`, the originals are untouched):

       python build_timing_patches.py

2. Time all twelve models on the six streams (three repeats; several hours in total, most of it the training
   phases of the slower released models):

       python run_all_models.py --data_root <data_root> --out_dir <s>

   `--models` and `--datasets` select subsets (`SFS` is our model); records are appended to `<s>/results.jsonl`.

3. Our model against its own settings and against the stream length (about two hours):

       python run_cost_curves.py --data_root <data_root> --out_dir <c>

   S3 runs our model on five streams with the number of timescales, the number of training blocks and the
   within-block updates changed one at a time (two epochs each, the second timed); S2 builds the 10 % and 30 %
   prefixes of LastFM and Flights under `<data_root>/processed_data/` and times them.  Records go to
   `<c>/results.jsonl`.

4. Copy `<s>/results.jsonl` to `speed/results.jsonl` and `<c>/results.jsonl` to `speed/results_s23.jsonl` in a
   copy of `results/` and rebuild with `REPRO_RESULTS` set.  Absolute times depend on the GPU; the figures show
   times relative to ours, which is the quantity to compare.
