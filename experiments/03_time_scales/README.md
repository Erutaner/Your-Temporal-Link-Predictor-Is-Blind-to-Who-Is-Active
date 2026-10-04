# Number of timescales

**Paper content:** the small table on the number of timescales in the main text (UCI and Reddit) and the
corresponding appendix table (all five streams, both settings).

## What is measured

Our node state is a bank of exponentially decaying counts, one per timescale.  The scales form a geometric grid
that covers the range of inter-event times of the training stream, with a fixed number of scales per decade of that
range; the number of scales K therefore follows from the stream, not from tuning.  The main runs use two scales per
decade, capped at 16.  This experiment retrains the model with the grid changed and nothing else:

| variant | grid | flags on top of the main configuration |
|---|---|---|
| K = 1 | a single scale at the geometric middle of the covered range | `--kmin 1 --per_decade 0` |
| one per decade | one scale per decade | `--per_decade 1` |
| two per decade | the main configuration (the main checkpoints are reused) | none |
| four per decade | four scales per decade, cap raised to 32 | `--per_decade 4 --kmax 32` |

The tables report the resulting K per stream, transductive test AP under the three negative-sampling protocols
(and the inductive setting in the appendix table), mean and standard deviation over three seeds, full model (the
activity term included, `beta` chosen on validation).

In file names this experiment carries the label `A1` (`results/ablation/A1_K1`, `A1_pd1`, `A1_pd4`, where `pd` is the
number of rates per decade; `outputs/table_a1_*`).

## Rebuild the tables from the shipped results

    python make_table_main.py        # outputs/table_a1_small.tex / .md   (UCI and Reddit, stacked)
    python make_table_appendix.py    # outputs/table_a1_appendix.tex / .md (five streams, both settings)

Inputs: `results/ablation/A1_K1`, `A1_pd1`, `A1_pd4` (one `<stream>_final_run<seed>_eval.json` per run) and the main
runs in `results/main/` for the two-per-decade row.

## Rerun from the data

    python run_time_scales.py --data_root <data_root> --out_dir <out>

trains and evaluates the three non-default variants on the five streams (`wikipedia, reddit, uci, enron, lastfm`)
for seeds 0, 1, 2: 45 runs, each as long as the corresponding main run (see `01_main_comparison/README.md`).
`--variants`, `--datasets` and `--seeds` narrow it down; `--dry_run` prints the commands.  Outputs go to
`<out>/A1_<variant>/<stream>/`, the layout of `results/ablation/`; copy them into a copy of `results/ablation/` and
rebuild with `REPRO_RESULTS` set.

## What to check

The training log prints the grid (`decades=... K=...`) at the start; the K column of the tables must match it.
