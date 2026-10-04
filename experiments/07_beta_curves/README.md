# Test AP as a function of the coefficient of the activity term

**Paper content:** the MOOC curve in the main text and the twelve-stream figure in the appendix.

## What is shown

For every main run the evaluation script scores the test split for a whole grid of `beta` values
(`score = f + beta * log lambda_u`, `beta` in 0, 0.5, 1, ..., see `betas` in the evaluation files) under the three
negative-sampling protocols.  The figures plot transductive test AP against `beta`, mean over the five seeds with a
one-standard-deviation band, and mark with a ring the `beta` the validation split selected (the value most seeds
chose).  Under random negatives the curve falls with `beta`: a random negative keeps the source node of the positive,
so both share the same `lambda_u` and the term can only add noise there.  Under historical and inductive negatives
the negatives have other sources and the term separates them.

## Rebuild the figures from the shipped results

    python make_figures.py --dataset mooc    # outputs/fig_beta_mooc.pdf / .png
    python make_figures.py --all             # outputs/fig_beta_appendix.pdf / .png (the twelve other streams)

Inputs: the `grid` block of `results/main/<stream>/<stream>_final_run<seed>_eval.json`.

## Rerun from the data

There is nothing separate to run: the grid is part of every main-run evaluation (`01_main_comparison/run_main.py`).
