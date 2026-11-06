import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

import os
import sys
import shutil
import time
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_NUM)

import torch

path_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(path_src)
if path_src not in sys.path: sys.path.insert(0, path_src)

from common.dataset.split_idx import get_split_idx_T2
from common.utils.utils_postprocessing import largest_connected_component_26, postprocess_threshold
from src_T2.options.hyper_parameters import HP_T2, load_local_HP_T2
from src_T2.utils.track1 import load_track1_model, predict_track1_volume, save_track1_cache

hp_T2 = HP_T2()
load_local_HP_T2(hp_T2)
batch_size = hp_T2.batch_size_inference if args.batch_size == 0 else args.batch_size
print(f'Track-1 checkpoint: {hp_T2.path_track1_checkpoint}')
print(f'Cache path: {hp_T2.path_cache}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hp_T1, model = load_track1_model(hp_T2.path_track1_checkpoint, device)
split = get_split_idx_T2(hp_T2.dset_preset, use_extra_train=hp_T2.use_extra_train_T2)
list_entry = list(dict.fromkeys(split['train']+split['valid']+split['test']))

t0 = time.perf_counter()
if os.path.isdir(hp_T2.path_cache): shutil.rmtree(hp_T2.path_cache)
os.makedirs(hp_T2.path_cache, exist_ok=True)
for name_dset, patient in list_entry:
    for cache_name in ('Idx_CK_SK_T1.pt', 'CK_count_T1.pt'):
        path_derived = f'{hp_T2.path_data_train}/{name_dset}/{patient}/{cache_name}'
        if os.path.isfile(path_derived): os.remove(path_derived)
print(f'[{time.perf_counter()-t0:.2f}s] Old T2 cache and derived sampling statistics removed done', flush=True)

with torch.inference_mode():
    for idx, (name_dset, patient) in enumerate(list_entry):
        path_save = f'{hp_T2.path_cache}/{name_dset}/{patient}'
        path_probability = f'{path_save}/probability.h5'
        path_mask = f'{path_save}/mask.h5'
        print(f'[{idx+1:3d}/{len(list_entry):3d}] {name_dset}/{patient}')
        probability = predict_track1_volume(hp_T1, model, name_dset, patient, hp_T2.path_data, device, batch_size=batch_size, num_workers=args.num_workers)
        mask = largest_connected_component_26(postprocess_threshold(probability, hp_T2.track1_threshold))
        save_track1_cache(hp_T2.path_cache, name_dset, patient, probability, mask)

print('Track-1 cache generation finished.', flush=True)
