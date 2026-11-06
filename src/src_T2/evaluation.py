import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument('--name_dset', type=str, default='Preset1')
parser.add_argument('--name_model', type=str, default='ASTRA_Net')
parser.add_argument('--name_ablation', type=str, default='')
parser.add_argument('--epoch_load', type=int, nargs='+', default=[-2])
parser.add_argument('--n', type=int, default=0)
parser.add_argument('--plot', type=lambda value: value.lower() == 'true', default=True)
args = parser.parse_args()

import os
import sys

path_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(path_src)
if path_src not in sys.path: sys.path.insert(0, path_src)

from src_T2.options.hyper_parameters import HP_T2, load_local_HP_T2
from src_T2.utils.utils_evaluation import evaluation_T2

hp = HP_T2(name_dset=args.name_dset, name_model=args.name_model, name_ablation=args.name_ablation)
load_local_HP_T2(hp)
print(f'Model: {hp.name}')
for epoch_load in args.epoch_load: evaluation_T2(epoch_load, hp.name, n=args.n, plot=args.plot)
