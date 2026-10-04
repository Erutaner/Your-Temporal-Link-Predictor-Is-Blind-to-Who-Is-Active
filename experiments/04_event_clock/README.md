# Number of training blocks and within-block updates

**Paper content:** the appendix table on the number of training blocks and the within-block updates (five streams,
both settings).

## What is measured

Training processes the stream in C contiguous blocks.  The node state is recomputed once per block (the block-start
state); a read inside a block decays that state to the query time and, when within-block updates are on, adds the
events of the same block that precede the query.  With the updates off, a read sees only the state at the block
start, so finer blocks (larger C) are the only way to see recent events.  The experiment retrains the model on the
grid

    within-block updates on / off  x  C = 48 / 96 / 192

with everything else as in the main runs.  The main configuration (updates on, the stream's own C: 96 for Wikipedia,
48 for Reddit, 192 for UCI, Enron and LastFM) is one of the cells and reuses the main checkpoints.  The table reports
transductive and inductive test AP under the three protocols, mean and standard deviation over three seeds, with and
without the source-activity term.

In file names this experiment carries the label `A2`: `results/ablation/A2_fr<0|1>_t<C>` (`fr1` / `fr0` = within-block
updates on / off, `t<C>` = the number of training blocks; the flags are `--fresh_records` and `--t_train`) and
`outputs/table_a2_appendix`.

## Rebuild the table from the shipped results

    python make_table.py         # outputs/table_a2_appendix.tex / .md

Inputs: `results/ablation/A2_fr<0|1>_t<C>/<stream>/<stream>_final_run<seed>_eval.json` and the main runs.

## Rerun from the data

    python run_event_clock.py --data_root <data_root> --out_dir <out>

trains and evaluates the five non-default cells per stream for seeds 0, 1, 2 (75 runs; `--datasets`, `--seeds`,
`--dry_run` as elsewhere).  Runs with C = 192 take longer per epoch than C = 48 (more block-start recomputations); the
cost figure of `06_speed` quantifies it.  Outputs go to `<out>/A2_fr<0|1>_t<C>/<stream>/`; copy them into a copy of
`results/ablation/` and rebuild with `REPRO_RESULTS` set.
