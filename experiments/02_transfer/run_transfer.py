# -*- coding: utf-8 -*-
"""Add our activity term to another model's scores without retraining that model (the transfer experiment).

    python run_transfer.py --backbone TPNet|TGN|DyGFormer|DSRD --data_root <root> --work_dir <dir>
                           [--ground_json results/transfer/ground_params.json] [--datasets ...] [--seeds 0,1,2]
                           [--gpu 0] [--skip_training] [--dry_run]

Run apply_transfer_overlays.py --work_dir <dir> first.  Per (stream, seed):
  1. train the backbone with its released configuration inside <work_dir>/trees/<tree> (random negatives, the
     released early stopping); --skip_training reuses a checkpoint already in that tree's saved_models/;
  2. run the backbone's evaluation script once per negative-sampling protocol with the dump directory set; the
     overlay writes <work_dir>/dumps/<prefix>_<protocol>_<split>.npz for the validation, new-node validation,
     test and new-node test splits;
  3. model/tools/sfs_transfer_readout.py rebuilds log lambda_u from the parameters in --ground_json, adds
     beta * log lambda_u to the dumped logits, picks beta on the validation split and writes
     <work_dir>/results/<name>/<stream>/seed<S>_transfer.json (name = TPNet_transfer, TGN_transfer,
     DyGFormer_fixpad_transfer or DSRD_transfer; beta = 0 in the file is the backbone alone).
Copy the results/<name> folders into results/transfer/ and run make_table.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import MODEL, RESULTS, ABLATION_DATASETS, ABLATION_SEEDS, sh, link_data  # noqa: E402

# backbone -> (tree, model name in the tree, extra flags, dump env prefix, result folder)
BACKBONES = {
    'TPNet': ('TPNet_strict', 'TPNet', '--prefix strict --use_random_projection', 'TPNET', 'TPNet_transfer'),
    'TGN': ('DyGLib', 'TGN', '', 'DYGLIB', 'TGN_transfer'),
    'DyGFormer': ('DyGLib', 'DyGFormer', '', 'DYGLIB', 'DyGFormer_fixpad_transfer'),
    'DSRD': ('DSRD_honest', 'DSRD', '', 'DYGLIB', 'DSRD_transfer'),
}
PROTOCOLS = ('random', 'historical', 'inductive')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', required=True, choices=sorted(BACKBONES))
    p.add_argument('--data_root', required=True)
    p.add_argument('--work_dir', required=True)
    p.add_argument('--ground_json', default=os.path.join(RESULTS, 'transfer', 'ground_params.json'))
    p.add_argument('--datasets', default=','.join(ABLATION_DATASETS))
    p.add_argument('--seeds', default=','.join(str(s) for s in ABLATION_SEEDS))
    p.add_argument('--gpu', default='0')
    p.add_argument('--skip_training', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    tree, model, extra, envp, folder = BACKBONES[a.backbone]
    py = sys.executable
    cwd = os.path.join(a.work_dir, 'trees', tree)
    if not a.dry_run:
        link_data(cwd, a.data_root)
    dumps = os.path.join(a.work_dir, 'dumps')
    for ds in a.datasets.split(','):
        for seed in [int(s) for s in a.seeds.split(',')]:
            prefix = ('strict_%s_seed%d' if a.backbone == 'TPNet' else model + '_%s_seed%d') % (ds, seed)
            common = ('%s --dataset_name %s --model_name %s --load_best_configs --num_runs 1 --seed_start %d --gpu %s'
                      % (extra, ds, model, seed, a.gpu)).strip()
            print('\n# %s %s seed %d' % (a.backbone, ds, seed), flush=True)
            if not a.skip_training:
                sh('%s -u train_link_prediction.py %s --test_interval_epochs 100000' % (py, common), cwd=cwd, dry_run=a.dry_run)
            env = {envp + '_DUMP_DIR': dumps, envp + '_DUMP_PREFIX': prefix}
            for st in PROTOCOLS:
                sh('%s -u evaluate_link_prediction.py %s --negative_sample_strategy %s' % (py, common, st), cwd=cwd, env=env,
                   dry_run=a.dry_run)
            out = os.path.join(a.work_dir, 'results', folder, ds, 'seed%d_transfer.json' % seed)
            if not a.dry_run:
                os.makedirs(os.path.dirname(out), exist_ok=True)
            sh('%s -u -W ignore %s --dataset %s --config %s --data_root %s --ground_json %s --dumps_dir %s --prefix %s '
               '--betas 0,0.5,1,1.5,2,3,4 --device cuda:%s --out %s'
               % (py, os.path.join(MODEL, 'tools', 'sfs_transfer_readout.py'), ds, os.path.join(MODEL, 'configs', ds + '.json'),
                  a.data_root, a.ground_json, dumps, prefix, a.gpu, out), cwd=MODEL, dry_run=a.dry_run)


if __name__ == '__main__':
    main()
