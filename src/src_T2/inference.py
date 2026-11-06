import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument('--name_dset', type=str, default='Preset1')
parser.add_argument('--name_model', type=str, default='ASTRA_Net')
parser.add_argument('--name_ablation', type=str, default='')
parser.add_argument('--epoch_load', type=int, nargs='+', default=[-2])
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--overlap_inference', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--n', type=int, default=0)
args = parser.parse_args()

import os
import sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_NUM)

import torch

path_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(path_src)
if path_src not in sys.path: sys.path.insert(0, path_src)

from src_T2.options.hyper_parameters import HP_T2, load_local_HP_T2
from src_T2.utils.utils_inference import inference_T2

print(torch.cuda.is_available(), ': ', torch.cuda.get_device_name(0), args.GPU_NUM)
hp = HP_T2(name_dset=args.name_dset, name_model=args.name_model, name_ablation=args.name_ablation)
load_local_HP_T2(hp)
print(f'Model: {hp.name}')
for epoch_load in args.epoch_load: inference_T2(epoch_load, hp.name, batch_size=args.batch_size, n=args.n, num_workers=args.num_workers, overlap_inference=args.overlap_inference)
