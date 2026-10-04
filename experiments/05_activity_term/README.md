# The activity term: what it is and what it contributes

**Paper content:** the table comparing forms of the activity term (main text: the transductive setting under inductive
negatives; appendix: the remaining protocols) and the small decomposition table (main text).

## What is measured

The score is `f + beta * b`, with `f` the pair score of the trunk and `b` a term that depends only on the source node's
past activity.  In the main runs `b = log lambda_u(t)`, the log-intensity of a fitted multi-scale process.  Two
questions, both answered on the main checkpoints without any retraining:

1. **Does the form of `b` matter?**  `b` is replaced by simpler functions of the same decayed counts, `beta` is chosen
   again on validation, and the trunk is left untouched:

   | row | `b` | flag of `evaluate_sfs.py` |
   |---|---|---|
   | trunk alone | none (`beta = 0`) | (from the same files, `beta` fixed at 0) |
   | fitted intensity (main runs) | `log lambda_u` | default |
   | slowest count | log(1 + count at the slowest scale) | `--readout_term count_slow` |
   | middle count | log(1 + count at the middle scale) | `--readout_term count_mid` |
   | time since last event | -log(1 + time since the source's last event) | `--readout_term recency` |
   | small network | a two-layer network on the log counts, fitted on the training split | `--readout_term mlp` |

2. **How much of the result is the term alone?**  The decomposition table puts side by side: two reference models,
   the trunk alone, `log lambda_u` alone (the trunk zeroed, so the score carries no information about the pair, only
   about the source's activity; flag `--b_only 1`), and the full score.

Everything is transductive test AP, mean and standard deviation over three seeds, on five streams.

In file names this experiment carries the labels `A3` (the form of the term: `results/ablation/A3_*`,
`outputs/table_a3_*`) and `I1` (the term alone: `results/ablation/I1_bonly`, `outputs/table_i1_small`).

## Rebuild the tables from the shipped results

    python make_table_main.py --strategy inductive   # outputs/table_a3_transductive_inductive.tex / .md
    python make_table_appendix.py        # outputs/table_a3_appendix.tex / .md
    python make_table_decomposition.py   # outputs/table_i1_small.tex / .md

Inputs: `results/ablation/A3_*` and `results/ablation/I1_bonly` (one evaluation file per run) and the main runs;
the decomposition table also takes DyGFormer from `results/baselines/summary.json` and TPNet from the parsed paper
values in `results/main/`.

## Rerun from the data

Needs the main checkpoints (`01_main_comparison/run_main.py`, or your own):

    python run_activity_term.py --data_root <data_root> --ckpt_dir <out_dir of run_main.py> --out_dir <out>

evaluates the five variants on the five streams for seeds 0, 1, 2: 75 evaluation runs, each a minute or two on one
GPU, no training.  Outputs go to `<out>/<variant>/<stream>/<stream>_final_run<seed>_eval.json`; copy the variant
folders into a copy of `results/ablation/` and rebuild with `REPRO_RESULTS` set.
