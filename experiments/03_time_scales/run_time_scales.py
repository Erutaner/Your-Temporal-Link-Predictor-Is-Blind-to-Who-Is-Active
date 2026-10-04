# -*- coding: utf-8 -*-
"""Retrain our model with a different number of time scales K (the scale-grid ablation).

    python run_time_scales.py --data_root <root> --out_dir <dir> [--variants K1,pd1,pd4] [--datasets ...]
                              [--seeds 0,1,2] [--device cuda:0] [--dry_run]

Variants (everything else is the main configuration of the stream):
    K1    a single rate, placed at the geometric middle of the covered range      (--kmin 1 --per_decade 0)
    pd1   one rate per decade of the covered range                                 (--per_decade 1)
    pd4   four rates per decade, upper bound raised to 32                          (--per_decade 4 --kmax 32)
The main configuration (two per decade, at most 16) is the run of 01_main_comparison and is not repeated.
Outputs: <out_dir>/A1_<variant>/<stream>/<stream>_final_run<seed>{.pt,_eval.json}; copy the A1_* folders
into results/ablation/ and run make_table_main.py / make_table_appendix.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import MODEL, MAIN_FLAGS, PATIENCE, ABLATION_DATASETS, ABLATION_SEEDS, flag_string, sh  # noqa: E402

VARIANTS = {'K1': {'kmin': 1, 'per_decade': 0}, 'pd1': {'per_decade': 1}, 'pd4': {'per_decade': 4, 'kmax': 32}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--variants', default='K1,pd1,pd4')
    p.add_argument('--datasets', default=','.join(ABLATION_DATASETS))
    p.add_argument('--seeds', default=','.join(str(s) for s in ABLATION_SEEDS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    py = sys.executable
    for v in a.variants.split(','):
        for ds in a.datasets.split(','):
            for seed in [int(s) for s in a.seeds.split(',')]:
                flags = dict(MAIN_FLAGS[ds])
                flags.update(VARIANTS[v])
                cfg = os.path.join(MODEL, 'configs', ds + '.json')
                odir = os.path.join(a.out_dir, 'A1_' + v, ds)
                base = os.path.join(odir, '%s_final_run%d' % (ds, seed))
                train = ('%s -u -W ignore %s --config %s --data_root %s --out_dir %s --run %d --tag _final --epochs 500 --patience %d '
                         '--print_every 5 --device %s %s' % (py, os.path.join(MODEL, 'scripts', 'train_sfs.py'), cfg, a.data_root, odir,
                                                            seed, PATIENCE[ds], a.device, flag_string(flags)))
                evaluate = ('%s -W ignore %s --ckpt %s.pt --config %s --data_root %s --settings transductive,inductive --periods val,test '
                            '--device %s --out %s_eval.json' % (py, os.path.join(MODEL, 'scripts', 'evaluate_sfs.py'), base, cfg, a.data_root,
                                                                a.device, base))
                print('\n# A1_%s %s seed %d\n%s\n%s' % (v, ds, seed, train, evaluate), flush=True)
                if a.dry_run:
                    continue
                os.makedirs(odir, exist_ok=True)
                for cmd in (train, evaluate):
                    sh(cmd, cwd=MODEL)


if __name__ == '__main__':
    main()
