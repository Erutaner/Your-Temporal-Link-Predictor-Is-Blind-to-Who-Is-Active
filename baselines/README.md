# Baseline code

Five folders, each a copy of a released code base.  Four of them carry corrections we had to make before the
released code could be compared with ours on equal terms; every correction is marked in the source with a short
comment (`LEAK FIX`, `STRICT LAW`, `STRICT-EARLIER-TIMESTAMP LAW`, `ROW-FIX`, `HOLDOUT ALIGNMENT`, `TRANSFER`; "law" is our word for the
rule that a query sees only strictly earlier events) and is explained below.  Nothing about
the models' architectures, losses, configurations or early stopping was changed; the released `--load_best_configs`
settings are used as published.  The upstream licenses are kept in each folder.

| folder | origin | used for |
|---|---|---|
| `DyGLib/` | DyGLib (Yu et al., NeurIPS 2023), the reference implementation of JODIE, DyRep, TGAT, TGN, CAWN, EdgeBank, TCL, GraphMixer and DyGFormer | the DyGFormer retraining of the main tables, the TGN and DyGFormer runs of the transfer experiment, the timing of nine models (EdgeBank included) |
| `TPNet_released/` | TPNet (Lu et al., NeurIPS 2024) exactly as released | the timing experiment (timed as released) |
| `TPNet_strict/` | TPNet with the same-timestamp correction | the TPNet retraining of the main tables, the TPNet runs of the transfer experiment |
| `GRN_honest/` | Graph Retention Networks (Chang et al., WWW 2026) with the corrections below | the GRN column of the main tables |
| `DSRD_honest/` | DSRD (Chang et al., 2026) with the corrections below | the DSRD column of the main tables, the DSRD runs of the transfer experiment, the timing experiment |
| `transfer_overlays/` | drop-in replacements of single files of the three trees above, used only by the transfer experiment | see `experiments/02_transfer/README.md` |

All five trees share one bookkeeping addition: `--seed_start`, so that the seeds of one stream can be run on
different machines (the released scripts ran seeds 0 .. `num_runs` - 1 in one process).

## The corrections

The protocol scores a query (u, v, t) against whatever happened strictly before t.  The released code of each
model below let something from t itself, or from later, reach the model.  The corrections close those paths and
nothing else.

### DyGLib

**DyGFormer: padding length as a side channel** (`models/DyGFormer.py`, `pad_sequences`).  DyGFormer scores the
positive pairs of a batch and the negative pairs in two separate calls.  The released code padded the neighbour
sequences of each call to the longest sequence present in that call, and the model averages over all patches,
padding included, without a mask.  The true endpoints and the sampled negatives have different history lengths, so
the two calls were padded to systematically different lengths and every embedding of a call shifted with it: the
padded length carried the label.  Now every call is padded to the configured maximum length; whenever a call
already contained a sequence of maximal length the outputs are identical to the released code.  On most streams
the released configuration saturates that maximum in both calls, so the two agree; Can. Parl. is the stream where
the released numbers were affected.

**JODIE, DyRep, TGN: messages from the same timestamp** (`models/MemoryModel.py`).  These models keep a raw message
per node after each batch and fold it into the memory when the node is next read.  Batches are 200 events, many
events share a timestamp, and timestamps run across batch boundaries, so a query at time t in one batch could read
a memory that already contained events at the same t from the previous batch.  Now messages are folded in only when
strictly earlier than the batch being scored; later ones stay pending.

**TGN: memory after the training stream** (`train_link_prediction.py`).  Needed by the transfer experiment only:
the saved checkpoint holds the memory after the validation split, so whenever early stopping saves an epoch the
memory as it was right after the training stream is saved next to it.  The released training flow is unchanged.

Two small compatibility edits without a marker: `torch.load(..., weights_only=False)` for current PyTorch and
`shutil.copytree` in place of the removed `distutils` in the preprocessing script.

### TPNet (`TPNet_strict/`)

**Walk matrices updated with same-timestamp events** (`models/TPNet.py`, the training and evaluation loops).  The
released code committed each batch's events into the temporal-walk matrices right after scoring the batch, so a
query in the next batch was scored against a state that already held events with its own timestamp (other
interactions of its endpoints at that time, on some streams the queried pair itself).  Observed events are now
buffered and committed, in the released 200-event granularity and through the unchanged update arithmetic, only
once every buffered event is strictly earlier than the batch about to be scored.  On streams without shared
timestamps across batches the two versions coincide; `experiments/01_main_comparison/tpnet_gate.py` decides per
stream whether the paper's numbers survive the correction (they do on nine streams; Can. Parl., US Legis.,
UN Trade and UN Vote are retrained).

Also: the 10 % held-out node sample is drawn in DyGLib's order (the released loader sorted the node set first and
therefore held out different nodes under the same seed), and `--test_interval_epochs` lets the per-epoch test
logging be skipped (zeros are logged instead; model selection on validation is unchanged).

### GRN (`GRN_honest/`)

The released training branch for GRN had several paths from the label to the model; the trainer, the evaluation
loop and the streaming-state module were rewritten around the unchanged retention arithmetic
(`models/GRN.py`, `models/GraphRetention.py`, `train_link_prediction.py`, `evaluate_models_utils.py`):

1. The positive pair of the training loss was computed for the source and the *sampled* destination, marked as a
   positive event, and the negative pair for the same nodes afterwards; the true destination never entered
   training as a positive.  The trainer now scores the true pair as positive and the sampled pair as negative.
2. The features of the edge being predicted were fed to the scoring pass (and the true edge's features to the
   negative pair).  Scoring now takes no edge features; they enter only when a positive event is written into
   the state.
3. Within a call, a query could see rows with the same timestamp, and the positive call wrote its events into the
   state before the negative call of the same batch read it.  Now pending events are committed only when strictly
   earlier than the batch, both calls read that same state, and inside a batch a query sees only strictly earlier
   rows.
4. Scoring a negative pair overwrote the last-interaction time of the sampled destination.  Negatives no longer
   write anything.
5. State backups returned references to tensors that were updated in place, so restoring a backup restored
   nothing: validation and the final evaluation ran on a state that had already consumed the validation (and
   test) events.  Backups are now copies, and the final evaluation rebuilds the state by streaming the training
   events through the loaded weights.
6. The released loader removed every validation / test edge touching a held-out node from the transductive sets;
   DyGLib keeps them.  Aligned to DyGLib.

The released repository had no evaluation script for a saved checkpoint under the historical and inductive
protocols; `evaluate_link_prediction.py` was added for that.  The node-classification script and the model files
of other architectures, which the GRN pipeline never imports, were removed.

### DSRD (`DSRD_honest/`)

**Batch rows merged across queries** (`models/DSRD.py`, `_build_batch_graph`).  The released code listed the batch's
centre nodes and all sampled neighbour occurrences, then mapped node ids to rows with a scatter in which the last
write wins.  Ids repeat in that list, so a centre whose id also appeared later (as a neighbour of another centre of
the same batch) lost its own row and its history was merged with another query's.  Positives and negatives are
scored in separate calls whose node lists differ, so the collision pattern was correlated with the label.  Now every
centre and every neighbour occurrence keeps its own row, and a query's subgraph depends only on its own history.

Also: the held-out node sample is drawn in DyGLib's order, the transductive validation / test sets keep the
held-out-node edges as in DyGLib, and the node-classification script was removed.  DSRD's data loader is DyGLib's.

## Transfer overlays

`transfer_overlays/<tree>/` holds replacement files for `DyGLib`, `TPNet_strict` and `DSRD_honest` that add two
things when an environment variable (`DYGLIB_DUMP_DIR`, or `TPNET_DUMP_DIR` for TPNet) is set: every evaluation
call writes its per-edge logits with the source ids and timestamps of the scored positives and negatives to an
`.npz` file, and the models that carry state between splits score the validation split from the state they had
right after the training stream (TGN from the saved memory, DSRD from a saved state that its overlaid training
script writes, TPNet by replaying the training stream through its parameter-free update).  Without the variable
the files behave exactly like the originals.  `experiments/02_transfer/apply_transfer_overlays.py` builds the
overlaid copies; the trees here are never modified in place.
