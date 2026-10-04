# -*- coding: utf-8 -*-
"""Re-score the main checkpoints with the source-activity term replaced (and with the trunk switched off).

    python run_activity_term.py --data_root <root> --ckpt_dir <dir of 01_main_comparison> --out_dir <dir>
                                [--datasets ...] [--seeds 0,1,2] [--device cuda:0] [--dry_run]

No training: every run is one call of model/scripts/evaluate_sfs.py on <ckpt_dir>/<stream>/<stream>_final_run<seed>.pt.
Variants and the flag that selects them:
    A3_count_slow   b = log(1 + n_u^K), the slowest-scale count            --readout_term count_slow
    A3_count_mid    b = log(1 + n_u^{K/2}), the middle-scale count          --readout_term count_mid
    A3_recency      b = -log(1 + time since the source's last event)        --readout_term recency
    A3_mlp          b = a small network on log(1 + n_u), fitted on training --readout_term mlp
    I1_bonly        the trunk zeroed, score = log lambda_u alone            --b_only 1
The coefficient beta is chosen on validation exactly as for the main results.  Outputs:
<out_dir>/<variant>/<stream>/<stream>_final_run<seed>_eval.json; copy the folders into results/ablation/ and
run make_table_main.py, make_table_appendix.py and make_table_decomposition.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import MODEL, ABLATION_DATASETS, ABLATION_SEEDS, sh  # noqa: E402

VARIANTS = [('A3_count_slow', '--readout_term count_slow'), ('A3_count_mid', '--readout_term count_mid'),
            ('A3_recency', '--readout_term recency'),
            ('A3_mlp', '--readout_term mlp'), ('I1_bonly', '--b_only 1')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--ckpt_dir', required=True, help='the --out_dir given to 01_main_comparison/run_main.py')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--datasets', default=','.join(ABLATION_DATASETS))
    p.add_argument('--seeds', default=','.join(str(s) for s in ABLATION_SEEDS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    py = sys.executable
    for ds in a.datasets.split(','):
        for seed in [int(s) for s in a.seeds.split(',')]:
            ckpt = os.path.join(a.ckpt_dir, ds, '%s_final_run%d.pt' % (ds, seed))
            cfg = os.path.join(MODEL, 'configs', ds + '.json')
            for v, flag in VARIANTS:
                odir = os.path.join(a.out_dir, v, ds)
                out = os.path.join(odir, '%s_final_run%d_eval.json' % (ds, seed))
                cmd = ('%s -W ignore %s --ckpt %s --config %s --data_root %s --settings transductive,inductive --periods val,test '
                       '--device %s %s --out %s' % (py, os.path.join(MODEL, 'scripts', 'evaluate_sfs.py'), ckpt, cfg, a.data_root,
                                                    a.device, flag, out))
                print('\n# %s %s seed %d\n%s' % (v, ds, seed, cmd), flush=True)
                if a.dry_run:
                    continue
                os.makedirs(odir, exist_ok=True)
                sh(cmd, cwd=MODEL)


if __name__ == '__main__':
    main()
