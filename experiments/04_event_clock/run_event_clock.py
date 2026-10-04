# -*- coding: utf-8 -*-
"""Retrain our model with a different number of training chunks T, with and without the within-chunk records
(the event-clock ablation).

    python run_event_clock.py --data_root <root> --out_dir <dir> [--datasets ...] [--seeds 0,1,2]
                              [--device cuda:0] [--dry_run]

Cells: within-chunk records on / off (--fresh_records 1 / 0) x T in {48, 96, 192} (--t_train).  The cell
(on, the stream's own T) is the main configuration and is not repeated here.  Outputs:
<out_dir>/A2_fr<0|1>_t<T>/<stream>/<stream>_final_run<seed>{.pt,_eval.json}; copy the A2_* folders into
results/ablation/ and run make_table.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import MODEL, MAIN_FLAGS, PATIENCE, ABLATION_DATASETS, ABLATION_SEEDS, flag_string, sh  # noqa: E402

CHUNKS = (48, 96, 192)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--datasets', default=','.join(ABLATION_DATASETS))
    p.add_argument('--seeds', default=','.join(str(s) for s in ABLATION_SEEDS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    py = sys.executable
    for ds in a.datasets.split(','):
        for fr in (1, 0):
            for t in CHUNKS:
                if fr == 1 and t == MAIN_FLAGS[ds].get('t_train'):
                    continue                     # the main configuration
                for seed in [int(s) for s in a.seeds.split(',')]:
                    flags = dict(MAIN_FLAGS[ds])
                    flags.update({'fresh_records': fr, 't_train': t})
                    cfg = os.path.join(MODEL, 'configs', ds + '.json')
                    v = 'A2_fr%d_t%d' % (fr, t)
                    odir = os.path.join(a.out_dir, v, ds)
                    base = os.path.join(odir, '%s_final_run%d' % (ds, seed))
                    train = ('%s -u -W ignore %s --config %s --data_root %s --out_dir %s --run %d --tag _final --epochs 500 '
                             '--patience %d --print_every 5 --device %s %s'
                             % (py, os.path.join(MODEL, 'scripts', 'train_sfs.py'), cfg, a.data_root, odir, seed, PATIENCE[ds],
                                a.device, flag_string(flags)))
                    evaluate = ('%s -W ignore %s --ckpt %s.pt --config %s --data_root %s --settings transductive,inductive '
                                '--periods val,test --device %s --out %s_eval.json'
                                % (py, os.path.join(MODEL, 'scripts', 'evaluate_sfs.py'), base, cfg, a.data_root, a.device, base))
                    print('\n# %s %s seed %d\n%s\n%s' % (v, ds, seed, train, evaluate), flush=True)
                    if a.dry_run:
                        continue
                    os.makedirs(odir, exist_ok=True)
                    for cmd in (train, evaluate):
                        sh(cmd, cwd=MODEL)


if __name__ == '__main__':
    main()
