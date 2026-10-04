# Main comparison

**Paper content:** the four main tables: test AP in the transductive setting (main text), test AUC-ROC in the
transductive setting, and AP and AUC-ROC in the inductive setting (appendix).  Each table has three blocks, one per
negative-sampling protocol (random, historical, inductive), thirteen streams per block, and average-rank rows.

## What is measured

Link prediction on thirteen event streams under the DyGLib protocol: chronological 70 / 15 / 15 split, 10 % of the
nodes held out for the inductive setting, the same seeds and samplers for every model.  All models train with
random negatives.  At test time each positive is paired with one negative drawn by one of three protocols:
random (a random destination), historical (a pair seen earlier in the stream), inductive (a pair never seen in
training).  AP and AUC-ROC per batch of 200 positives and their negatives, averaged.

Columns: nine models with the values from the TPNet paper (JODIE, TGN, CAWN, EdgeBank, GraphMixer, NAT, PINT,
DyGFormer, TPNet), GRN and DSRD retrained by us, and two columns of ours: the trunk alone (`beta = 0`) and the full
score with the activity term (`beta` chosen on validation).

A dagger marks a cell that comes from our own retraining because the released code had to be corrected (see
`baselines/README.md`):

* GRN and DSRD on every stream, five seeds;
* DyGFormer with fixed-length padding on the streams we retrained (Can. Parl. with five seeds; Wikipedia, Reddit,
  UCI and Enron with three; LastFM with one);
* TPNet on the four streams where the corrected code does not reproduce the paper (Can. Parl., US Legis.,
  UN Trade, UN Vote); `tpnet_gate.py` applies the rule: seed 0 of the corrected code was run on every stream, and five seeds
  where the seed-0 run fell outside the tolerance; a stream keeps its paper values if the mean over its runs is within
  1.0 AP point (or two paper standard deviations, whichever is larger) of the paper on all six AP cells, otherwise
  the retrained mean replaces the paper values.

Our numbers are mean and standard deviation over five seeds.

## Rebuild the tables from the shipped results

    python tpnet_gate.py      # writes results/baselines/tpnet_gate.json and prints the per-stream outcome
    python make_tables.py     # outputs/table_trans_ap.tex, table_trans_auc.tex, table_ind_ap.tex, table_ind_auc.tex, main_tables.md
    python make_table_settings.py   # outputs/table_training_settings.tex / .md: the per-stream training settings, read from common.py and the configs

Inputs: `results/main/<stream>/<stream>_final_run<seed>_eval.json` (ours), `results/baselines/summary.json`
(retrained baselines, built from the per-seed files by `collect_baselines.py`), `results/main/tpnet_paper*.json`
(values parsed from the TPNet paper) and `results/baselines/tpnet_gate.json`.  `results/baselines/TPNet_strict/` also
holds seeds 1 to 4 of some streams that keep their paper values; the tables do not use them.

Each of our evaluation files holds, per setting and protocol, the test AP / AUC for a grid of `beta` values
(`grid`) and the entry selected on validation (`selected`); `beta = 0` in the grid is the trunk alone.

## Rerun from the data

### Our model

    python run_main.py --data_root <data_root> --out_dir <out>

trains one run per (stream, seed) with the settings listed in `../common.py` (`MAIN_FLAGS`, `PATIENCE`) on top of
`model/configs/<stream>.json`, then evaluates it (both settings, all three protocols, the whole `beta` grid).
Defaults: all thirteen streams, seeds 0 to 4; `--datasets`, `--seeds`, `--device` narrow it down, `--dry_run` prints
the commands.  Early stopping watches the validation AP and AUC under random negatives; a run stops 100 epochs
after its last improvement (50 on LastFM and Flights).  On one GPU an epoch takes seconds on the small streams
(UCI, Wikipedia, Enron, MOOC, UN Trade, UN Vote, US Legis., Can. Parl.) and up to a few minutes on the large ones
(LastFM, Flights, Contact, Social Evo.), so expect well under an hour per seed for the small streams and a few
hours for the large ones.  Outputs: `<out>/<stream>/<stream>_final_run<seed>.pt` and `_eval.json`.

### Retrained baselines

    python run_baselines.py --model GRN       --data_root <data_root> --out_dir <b>
    python run_baselines.py --model DSRD      --data_root <data_root> --out_dir <b>
    python run_baselines.py --model DyGFormer --data_root <data_root> --out_dir <b> --datasets CanParl,wikipedia,reddit,uci,enron,lastfm
    python run_baselines.py --model TPNet     --data_root <data_root> --out_dir <b> --datasets CanParl,USLegis,UNtrade,UNvote
    python collect_baselines.py <b> --out <b>/summary.json

Each run uses the tree's own training script with its released configuration (`--load_best_configs`), then its
evaluation script for the other protocols; the three result files per run are copied to
`<b>/<name>/<stream>/seed<S>_{random,historical,inductive}.json`.  The cost is the released models' own training
time with their released early stopping: minutes per seed on the small streams, hours on the large ones (DSRD and
GRN on Contact, Flights and Social Evo. are the longest).  To recheck the TPNet rule on every stream, run TPNet with
`--seeds 0` on all thirteen and `tpnet_gate.py` on the result.

### Putting it together

Copy `results/` to a folder of your own, replace the files you regenerated (`main/<stream>/`, `baselines/<name>/`,
`baselines/summary.json`), and run `REPRO_RESULTS=<that folder> python make_all.py` from the package root.
