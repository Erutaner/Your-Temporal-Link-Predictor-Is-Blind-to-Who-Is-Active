# -*- coding: utf-8 -*-
"""Train and evaluate our model on the thirteen streams with the settings of the main results.

    python run_main.py --data_root <root> --out_dir <dir> [--datasets uci,mooc] [--seeds 0,1,2,3,4]
                       [--device cuda:0] [--dry_run]

One (stream, seed) = one training run followed by one evaluation run.  Training writes
<out_dir>/<stream>/<stream>_final_run<seed>.pt (the selected epoch) and evaluation writes
<out_dir>/<stream>/<stream>_final_run<seed>_eval.json with the test AP / AUC of both settings and all three
negative-sampling protocols, plus the readout coefficient grid.  Copy the _eval.json files into
results/main/<stream>/ (or point REPRO_RESULTS at <out_dir>'s parent layout) and run make_tables.py.

<root> must hold processed_data/<stream>/ml_<stream>.{csv,npy,_node.npy}; see ../../data/README.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import MODEL, DATASETS, MAIN_FLAGS, MAIN_SEEDS, PATIENCE, flag_string, main_run_files, sh  # noqa: E402


def commands(ds, seed, data_root, out_dir, device):
    py = sys.executable
    cfg = os.path.join(MODEL, 'configs', ds + '.json')
    odir = os.path.join(out_dir, ds)
    ckpt, ev = main_run_files(out_dir, ds, seed)
    train = ('%s -u -W ignore %s --config %s --data_root %s --out_dir %s --run %d --tag _final --epochs 500 --patience %d '
             '--print_every 5 --device %s %s' % (py, os.path.join(MODEL, 'scripts', 'train_sfs.py'), cfg, data_root, odir, seed,
                                                PATIENCE[ds], device, flag_string(MAIN_FLAGS[ds])))
    evaluate = ('%s -W ignore %s --ckpt %s --config %s --data_root %s --settings transductive,inductive --periods val,test '
                '--device %s --out %s' % (py, os.path.join(MODEL, 'scripts', 'evaluate_sfs.py'), ckpt, cfg, data_root, device, ev))
    return odir, train, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--datasets', default=','.join(k for k, _ in DATASETS))
    p.add_argument('--seeds', default=','.join(str(s) for s in MAIN_SEEDS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry_run', action='store_true', help='print the commands and exit')
    a = p.parse_args()
    for ds in a.datasets.split(','):
        for seed in [int(s) for s in a.seeds.split(',')]:
            odir, train, evaluate = commands(ds, seed, a.data_root, a.out_dir, a.device)
            print('\n# %s seed %d\n%s\n%s' % (ds, seed, train, evaluate), flush=True)
            if a.dry_run:
                continue
            os.makedirs(odir, exist_ok=True)
            for cmd in (train, evaluate):
                sh(cmd, cwd=MODEL)


if __name__ == '__main__':
    main()
