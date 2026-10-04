# -*- coding: utf-8 -*-
"""Retrain our model with channels of the trunk removed (the trunk-channel ablation).

    python run_trunk_channels.py --data_root <root> --out_dir <dir> [--variants noset,nosketch,nocount,bare]
                                 [--datasets ...] [--seeds 0,1,2] [--device cuda:0] [--dry_run]

Variants (everything else is the main configuration of the stream):
    noset     no candidate-conditioned set encoder over the source's last events      (--set_M 0)
    nosketch  no random-projection structure channels; the set encoder needs them,
              so it is off as well                                                    (--sketch 0 --set_M 0)
    nocount   the exact pair / endpoint count channels of the scoring head zeroed     (--count_channels 0)
    bare      all three removed                                                       (--sketch 0 --set_M 0 --count_channels 0)
Outputs: <out_dir>/A4_<variant>/<stream>/<stream>_final_run<seed>{.pt,_eval.json}; copy the A4_* folders into
results/ablation/ and run make_table.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import MODEL, MAIN_FLAGS, PATIENCE, ABLATION_DATASETS, ABLATION_SEEDS, flag_string, sh  # noqa: E402

VARIANTS = {'noset': {'set_M': 0}, 'nosketch': {'sketch': 0, 'set_M': 0}, 'nocount': {'count_channels': 0},
            'bare': {'sketch': 0, 'set_M': 0, 'count_channels': 0}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--variants', default=','.join(VARIANTS))
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
                odir = os.path.join(a.out_dir, 'A4_' + v, ds)
                base = os.path.join(odir, '%s_final_run%d' % (ds, seed))
                train = ('%s -u -W ignore %s --config %s --data_root %s --out_dir %s --run %d --tag _final --epochs 500 --patience %d '
                         '--print_every 5 --device %s %s' % (py, os.path.join(MODEL, 'scripts', 'train_sfs.py'), cfg, a.data_root, odir,
                                                            seed, PATIENCE[ds], a.device, flag_string(flags)))
                evaluate = ('%s -W ignore %s --ckpt %s.pt --config %s --data_root %s --settings transductive,inductive --periods val,test '
                            '--device %s --out %s_eval.json' % (py, os.path.join(MODEL, 'scripts', 'evaluate_sfs.py'), base, cfg, a.data_root,
                                                                a.device, base))
                print('\n# A4_%s %s seed %d' % (v, ds, seed), flush=True)
                if a.dry_run:
                    for cmd in (train, evaluate):
                        print(cmd)
                    continue
                os.makedirs(odir, exist_ok=True)
                for cmd in (train, evaluate):
                    sh(cmd, cwd=MODEL)


if __name__ == '__main__':
    main()
