# Data

All experiments use the thirteen event streams of the DyGLib benchmark (Yu et al., NeurIPS 2023) in the processed
form that DyGLib distributes: Wikipedia, Reddit, MOOC, LastFM, Enron, Social Evo., UCI, Flights, Can. Parl.,
US Legis., UN Trade, UN Vote and Contact.  We did not touch the data; every model here, ours included, reads the
same three files per stream.

## Getting the files

    bash get_data.sh <data_root>                    # all thirteen streams
    bash get_data.sh <data_root> wikipedia uci      # only the streams named

downloads the zips from the public Zenodo record of the benchmark (record 7213796) and unpacks, for every stream,

    <data_root>/processed_data/<stream>/ml_<stream>.csv        the events: u, i, ts, label, idx (chronological)
    <data_root>/processed_data/<stream>/ml_<stream>.npy        edge features, one row per event (row 0 is padding)
    <data_root>/processed_data/<stream>/ml_<stream>_node.npy   node features, one row per node (row 0 is padding)

The stream folder names are the ones every script expects (`wikipedia`, `reddit`, `mooc`, `lastfm`, `enron`,
`SocialEvo`, `uci`, `Flights`, `CanParl`, `USLegis`, `UNtrade`, `UNvote`, `Contacts`).  The zips are about 5 GB in
total and the unpacked files about 6 GB.  If you already have a DyGLib `processed_data` folder, point `--data_root`
at its parent and nothing needs to be downloaded.

## Who reads what

* Our scripts take `--data_root <data_root>` and read `<data_root>/processed_data/<stream>/`.
* The released baseline code reads `./processed_data/<stream>/` relative to its own folder.  The runner scripts in
  `experiments/` create that link (`<tree>/processed_data -> <data_root>/processed_data`) before the first run; if
  you call a baseline script by hand, create the link (or copy the folder) yourself.

## Splits and protocol

Nothing about the split is configurable: every run uses DyGLib's chronological 70 / 15 / 15 split, its 10 % of
held-out nodes for the inductive setting, its seeds and its three negative samplers (random, historical,
inductive).  Our model uses DyGLib's own split, sampler and metric code, copied unmodified into
`model/tgmin/vendor/`.

## Statistics table

    python stats.py --data_root <data_root>

recomputes the statistics table of the appendix (`outputs/table_datasets.tex` and `.md`): the benchmark's standard
columns plus the range covered by the time-scale grid and the number of scales per stream, the share of events that
share a timestamp with another event, and the share of test-period events whose node pair occurred earlier.

## A stream of your own

`model/scripts/preprocess.py` turns a CSV with columns `u, i, ts, label[, f1, f2, ...]` into the three files above;
its docstring lists the conventions (integer node ids, chronological order, optional edge features).
