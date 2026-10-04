# Our model

    tgmin/sfs.py        the node state (a bank of exponentially decaying counts on a grid of time scales, with the
                        partner and feature states that ride on it), the reads at a query time, training, the
                        fitted activity process lambda_u and its closed-form likelihood
    tgmin/data.py       loading a stream, chunking, the per-chunk tensors the trainer consumes
    tgmin/readout.py    the scoring head
    tgmin/checkpoint.py rebuilding a trained run from its checkpoint
    tgmin/vendor/       DyGLib's split, negative samplers and metrics, copied unmodified: the evaluation protocol
    scripts/train_sfs.py     train one run on one stream
    scripts/evaluate_sfs.py  evaluate a checkpoint under every setting and protocol, for a grid of beta
    scripts/preprocess.py    turn a raw event table into the ml_* files (only for streams of your own)
    tools/                   brute-force self-tests of the state and the reads, a device / batch-invariance check,
                             a checkpoint recheck, and the transfer readout used by experiments/02_transfer
    configs/<stream>.json    per-stream settings: block mode and count, and the training settings of the main runs

Names used in the code and what the paper calls them:

| in the code | in the paper |
|---|---|
| `tgmin`, *SFS* | the package and the model |
| *chunk*, `--t_train T` | block, the number of training blocks $C$ |
| *entering state* of a chunk | the block-start state |
| *fresh records* (`--fresh_records`) | within-block updates |
| *fresh state* (`--fresh_state`) | within-block updates of the partner-sum state ("partner updates") |
| *trunk*, score `f` | the relation branch (MTHR and CRM), the base score $f$ |
| *ground process* `lambda_u` (`GroundProcess`, `--ground`, `ground_params.json`) | SNAM, the source-activity term $\log\lambda_u$ |
| *scale*, `--per_decade`, `--kmax`, `--kmin` | timescale, the timescale grid |
| *set encoder* (`--set_M`) | the recent interaction encoding $c_{uv}(t)$ |
| *sketch* (`--sketch`) | the structural features from random projections |
| *count channels* (`--count_channels`) | the pair and endpoint counts $n_{uv}^k$, $n_u^k$, $n_v^k$ among the pair features |

The scripts are run from this folder or by absolute path; `experiments/*/run_*.py` build the exact commands.

## Training one run

    python scripts/train_sfs.py --config configs/uci.json --data_root <data_root> --out_dir <out> --run 0 --tag _final \
        --epochs 500 --patience 100 --print_every 5 <flags>

`--run` is the seed.  The configs hold the training settings of the main runs, so the command above without `<flags>`
is the main run of the stream, with one exception: Enron and Contact add `--fresh_state 1`.  The full flag list of
every stream is in `experiments/common.py` (`MAIN_FLAGS`; the run scripts pass it explicitly):

| flag | meaning |
|---|---|
| `--t_train T` | number of training blocks (streams with one block per time step take it from the config) |
| `--e_train 0/1` | learn the node table (1) or use fixed random node codes (0) |
| `--dropout`, `--wd`, `--lr` | dropout, weight decay, learning rate |
| `--neg_pool train/seen` | training negatives drawn from all training destinations or from those seen so far |
| `--fresh_state 0/1` | within-block events also update the partner-sum state (1) or only the counts (0) |
| `--set_M` | number of most recent events read by the recent interaction encoding |
| `--fresh_records 0/1` | within-block updates on (1) or off (0) |
| `--per_decade`, `--kmax`, `--kmin` | the timescale grid: timescales per decade of the covered range, and the bounds on K |

The grid of timescales covers the range from the smallest positive gap between two timestamps of the training
stream to its whole span, `per_decade` timescales per decade of that range, K = the resulting count clipped to
[kmin, kmax]; the main runs use two per decade and at most 16.  Nothing else about the model depends on the stream.

Every epoch prints the validation and test AP under the three protocols, with and without the activity term
(`+lambda`).  The checkpoint is saved at every epoch that improves the validation AP and AUC under random
negatives; training stops `--patience` epochs after the last improvement.

## Evaluating a checkpoint

    python scripts/evaluate_sfs.py --ckpt <out>/uci_final_run0.pt --config configs/uci.json --data_root <data_root> \
        --settings transductive,inductive --periods val,test --out <out>/uci_final_run0_eval.json

writes, per setting and protocol, the test AP / AUC for every `beta` of `--betas` (`grid`) and the entry whose
validation AP is best (`selected`).  `--readout_term` replaces the activity term by a simpler function of the same
counts and `--b_only 1` zeroes the trunk (`experiments/05_activity_term`).

## Tests

`tools/sfs_selftest.py` and `tools/sfs_selftest2.py` compare every state and read against brute-force sums over
the stream on a synthetic stream; `tools/sfs_device_check.py` checks that scoring on the GPU, on the CPU and in
batches of different sizes gives the same numbers; `tools/sfs_recheck_ckpt.py` recomputes a checkpoint's reported
numbers.  They take a minute or two each and need no data beyond the stream they are pointed at.
