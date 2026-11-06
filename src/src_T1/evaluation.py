import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument('--name_dset', type=str, default='Preset1')
parser.add_argument('--name_model', type=str, default='ASTRA_Net_phase_1')
parser.add_argument('--name_ablation', type=str, default='')
parser.add_argument('--epoch_load', type=int, nargs='+', default=[-1])
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--n', type=int, default=0)
parser.add_argument('--dset_test', type=str, nargs='+', default=None)
parser.add_argument('--plot', type=lambda value: value.lower() == 'true', default=True)
args = parser.parse_args()

import os
import sys
path_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(path_src)
if path_src not in sys.path: sys.path.insert(0, path_src)

from src_T1.options.hyper_parameters import HP, load_local_HP
from src_T1.utils.utils_evaluation import evaluation, evaluation_audit
from src_T1.utils.utils_inference import load_kernel_info

hp = HP(name_dset=args.name_dset, name_model=args.name_model, name_ablation=args.name_ablation)
load_local_HP(hp)
print(f'Model: {hp.name}')
for epoch_load in args.epoch_load:
    epoch_tag = load_kernel_info(epoch_load, hp.name)[3]
    alpha = hp.alpha_evaluation if args.alpha is None else args.alpha
    evaluation(epoch_tag, hp.name, alpha=alpha, n=args.n, dset_test=args.dset_test, plot=args.plot)
    # evaluation_audit(epoch_tag, hp.name, alpha=alpha, n=args.n, dset_test=args.dset_test)
