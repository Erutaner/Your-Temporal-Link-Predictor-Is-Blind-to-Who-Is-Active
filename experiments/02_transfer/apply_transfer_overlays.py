# -*- coding: utf-8 -*-
"""Make the copies of the three backbone trees that the transfer experiment evaluates from.

    python apply_transfer_overlays.py --work_dir <dir>

Copies baselines/DyGLib, baselines/TPNet_strict and baselines/DSRD_honest to <work_dir>/trees/<name> and drops the
files of baselines/transfer_overlays/<name>/ over them.  The overlays add, without touching the released metric
path: (1) a per-edge dump of the scored logits, with the source ids and timestamps of the positives and negatives,
whenever DYGLIB_DUMP_DIR (TPNET_DUMP_DIR for TPNet) is set; (2) a clean validation pass for the models that carry
state, since their checkpoints hold the state after the validation split: TGN scores validation from the memory
saved right after the training stream, TPNet by replaying the training stream into a fresh projection state, DSRD
from the state saved right after the training stream.  DSRD's overlay also patches its training script so that this
state is saved next to the checkpoint (the package's DyGLib training script already saves the memory of JODIE / DyRep / TGN).
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import BASELINES  # noqa: E402

TREES = ['DyGLib', 'TPNet_strict', 'DSRD_honest']
SKIP = shutil.ignore_patterns('__pycache__', 'saved_models', 'saved_results', 'logs', 'processed_data')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work_dir', required=True)
    a = p.parse_args()
    for name in TREES:
        dst = os.path.join(a.work_dir, 'trees', name)
        if os.path.exists(dst):
            print('exists, left as is:', dst)
            continue
        shutil.copytree(os.path.join(BASELINES, name), dst, ignore=SKIP)
        ov = os.path.join(BASELINES, 'transfer_overlays', name)
        for f in sorted(os.listdir(ov)):
            shutil.copy2(os.path.join(ov, f), os.path.join(dst, f))
            print('%s: %s replaced' % (name, f))


if __name__ == '__main__':
    main()
