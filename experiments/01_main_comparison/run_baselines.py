# -*- coding: utf-8 -*-
"""Retrain the four baselines whose released code we corrected, with their released configurations.

    python run_baselines.py --model TPNet|DyGFormer|GRN|DSRD --data_root <root> --out_dir <dir>
                            [--datasets ...] [--seeds 0,1,2,3,4] [--gpu 0] [--dry_run]

Trees: TPNet -> baselines/TPNet_strict, DyGFormer -> baselines/DyGLib (fixed-length padding), GRN -> baselines/GRN_honest,
DSRD -> baselines/DSRD_honest (see baselines/README.md for what was changed).  Per (stream, seed): the tree's training
script (random negatives, the released early stopping), then its evaluation script for the historical and inductive
protocols (DSRD also for random: its training script writes no result file).  The three result files are copied to
<out_dir>/<name>/<stream>/seed<S>_{random,historical,inductive}.json with name = TPNet_strict, DyGFormer_fixpad,
GRN_honest or DSRD_honest, the layout collect_baselines.py reads:

    python collect_baselines.py <out_dir> --out <out_dir>/summary.json
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import BASELINES, DATASETS, MAIN_SEEDS, sh, link_data  # noqa: E402

# model -> (tree, model name in the tree, extra flags, result folder name)
MODELS = {'TPNet': ('TPNet_strict', 'TPNet', '--prefix strict --use_random_projection', 'TPNet_strict'),
          'DyGFormer': ('DyGLib', 'DyGFormer', '', 'DyGFormer_fixpad'),
          'GRN': ('GRN_honest', 'GRN', '', 'GRN_honest'),
          'DSRD': ('DSRD_honest', 'DSRD', '', 'DSRD_honest')}


def result_files(model, tree_dir, ds, seed):
    """protocol -> the file the tree writes."""
    if model == 'TPNet':
        r = os.path.join(tree_dir, 'saved_results')
        return {'random': os.path.join(r, 'strict_link_%s_TPNet_seed%d.json' % (ds, seed)),
                'historical': os.path.join(r, 'strict_link_historical_%s_TPNet_seed%d.json' % (ds, seed)),
                'inductive': os.path.join(r, 'strict_link_inductive_%s_TPNet_seed%d.json' % (ds, seed))}
    name = MODELS[model][1]
    r = os.path.join(tree_dir, 'saved_results', name, ds)
    if model == 'DSRD':
        return {st: os.path.join(r, '%s_negative_sampling_DSRD_seed%d.json' % (st, seed)) for st in ('random', 'historical', 'inductive')}
    return {'random': os.path.join(r, '%s_seed%d.json' % (name, seed)),
            'historical': os.path.join(r, 'historical_negative_sampling_%s_seed%d.json' % (name, seed)),
            'inductive': os.path.join(r, 'inductive_negative_sampling_%s_seed%d.json' % (name, seed))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=sorted(MODELS))
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--datasets', default=','.join(k for k, _ in DATASETS))
    p.add_argument('--seeds', default=','.join(str(s) for s in MAIN_SEEDS))
    p.add_argument('--gpu', default='0')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    tree, name, extra, folder = MODELS[a.model]
    cwd = os.path.join(BASELINES, tree)
    py = sys.executable
    if not a.dry_run:
        link_data(cwd, a.data_root)
    evaluated = ('random', 'historical', 'inductive') if a.model == 'DSRD' else ('historical', 'inductive')
    for ds in a.datasets.split(','):
        for seed in [int(s) for s in a.seeds.split(',')]:
            common = ('%s --dataset_name %s --model_name %s --load_best_configs --num_runs 1 --seed_start %d --gpu %s'
                      % (extra, ds, name, seed, a.gpu)).strip()
            print('\n# %s %s seed %d' % (a.model, ds, seed), flush=True)
            sh('%s -u train_link_prediction.py %s --test_interval_epochs 100000' % (py, common), cwd=cwd, dry_run=a.dry_run)
            for st in evaluated:
                sh('%s -u evaluate_link_prediction.py %s --negative_sample_strategy %s' % (py, common, st), cwd=cwd, dry_run=a.dry_run)
            dst = os.path.join(a.out_dir, folder, ds)
            for st, f in result_files(a.model, cwd, ds, seed).items():
                print('copy %s -> %s' % (f, os.path.join(dst, 'seed%d_%s.json' % (seed, st))))
                if not a.dry_run:
                    os.makedirs(dst, exist_ok=True)
                    shutil.copy2(f, os.path.join(dst, 'seed%d_%s.json' % (seed, st)))


if __name__ == '__main__':
    main()
