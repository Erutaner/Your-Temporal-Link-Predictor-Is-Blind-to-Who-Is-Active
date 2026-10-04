# -*- coding: utf-8 -*-
"""Rebuild every table and figure of the paper from the result files in results/.

    python make_all.py

Writes outputs/*.tex, outputs/*.md and outputs/*.pdf|png, and refreshes results/baselines/tpnet_gate.json (the
per-stream outcome of the TPNet rule).  Needs numpy and matplotlib.  To rebuild from your own runs, set REPRO_RESULTS to a folder
with the same layout as results/ (see README.md).
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(ROOT, 'experiments')
STEPS = [
    ('01_main_comparison', 'tpnet_gate.py', []),
    ('01_main_comparison', 'make_tables.py', []),
    ('01_main_comparison', 'make_table_settings.py', []),
    ('02_transfer', 'make_table.py', []),
    ('02_transfer', 'make_table_appendix.py', []),
    ('02_transfer', 'make_table_tpnet_streams.py', []),
    ('03_time_scales', 'make_table_main.py', []),
    ('03_time_scales', 'make_table_appendix.py', []),
    ('04_event_clock', 'make_table.py', []),
    ('05_activity_term', 'make_table_main.py', ['--strategy', 'inductive']),
    ('05_activity_term', 'make_table_appendix.py', []),
    ('05_activity_term', 'make_table_decomposition.py', []),
    ('06_speed', 'make_table_all_models.py', []),
    ('06_speed', 'make_table_stream_length.py', []),
    ('06_speed', 'make_figure_main.py', []),
    ('06_speed', 'make_figure_appendix.py', []),
    ('06_speed', 'make_figure_cost.py', []),
    ('07_beta_curves', 'make_figures.py', ['--dataset', 'mooc']),
    ('07_beta_curves', 'make_figures.py', ['--all']),
    ('08_trunk_channels', 'make_table.py', []),
]


def main():
    os.makedirs(os.environ.get('REPRO_OUTPUTS', os.path.join(ROOT, 'outputs')), exist_ok=True)
    for folder, script, args in STEPS:
        print('== %s/%s %s' % (folder, script, ' '.join(args)), flush=True)
        rc = subprocess.call([sys.executable, os.path.join(E, folder, script)] + args, cwd=os.path.join(E, folder))
        if rc != 0:
            raise SystemExit('%s/%s failed (exit %d)' % (folder, script, rc))
    print('all tables and figures written')


if __name__ == '__main__':
    main()
