# Which channels of the trunk carry the pair information

**Paper content:** the appendix table on the trunk's channels (five streams, both settings).

## What is measured

The trunk (the relation branch of the paper) scores a pair from three groups of inputs: the two node representations `[h_u; h_v]` built from the
multi-scale state, a set of pair features (exact decayed pair and endpoint counts at every scale, learnable
bilinear matches between the two states, a static node-identity term, and random-projection estimates of the
structural overlap of the two neighbourhoods), and a candidate-conditioned encoding of the source's last 20 events.
This experiment retrains the model with parts of the last two groups removed, everything else as in the main runs:

| row | removed | flags on top of the main configuration |
|---|---|---|
| no set encoder | the recent interaction encoding $c_{uv}(t)$ of the source's last 20 events | `--set_M 0` |
| no projection channels | the structural features from random projections, and with them the recent interaction encoding, which reads them | `--sketch 0 --set_M 0` |
| no count channels | the pair and endpoint counts $n_{uv}^k$, $n_u^k$, $n_v^k$ among the pair features (the source-activity term keeps its own counts) | `--count_channels 0` |
| bare trunk | all three: only `[h_u; h_v]`, the bilinear matches and the static term remain | `--sketch 0 --set_M 0 --count_channels 0` |

The table reports transductive and inductive test AP under the three negative-sampling protocols, mean and standard
deviation over three seeds, for the trunk alone and for the full score with the activity term (`beta` chosen on
validation), next to the main configuration.

In file names this experiment carries the label `A4` (`results/ablation/A4_noset`, `A4_nosketch`, `A4_nocount`,
`A4_bare`; `outputs/table_a4_appendix`).

## Rebuild the table from the shipped results

    python make_table.py         # outputs/table_a4_appendix.tex / .md

Inputs: `results/ablation/A4_*/<stream>/<stream>_final_run<seed>_eval.json` and the main runs.

## Rerun from the data

    python run_trunk_channels.py --data_root <data_root> --out_dir <out>

trains and evaluates the four variants on the five streams (`wikipedia, reddit, uci, enron, lastfm`) for seeds
0, 1, 2: 60 runs, each about as long as the corresponding main run or shorter (the variants without the projection
channels compute less).  `--variants`, `--datasets`, `--seeds` and `--dry_run` as elsewhere.  Outputs go to
`<out>/A4_<variant>/<stream>/`; copy the folders into a copy of `results/ablation/` and rebuild with `REPRO_RESULTS` set.
