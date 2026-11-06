import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument('--idx_s', type=int, default=0)
parser.add_argument('--idx_n', type=int, default=20)
args = parser.parse_args()

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_NUM)
import sys
import numpy as np
import warnings
import SimpleITK as sitk
from pathlib import Path
from IPython.display import clear_output

import torch

path_src = Path(__file__).resolve().parents[1]
os.chdir(path_src)
if str(path_src) not in sys.path: sys.path.append(str(path_src))

from common.utils.utils_common import listdir, load_mha, make_dir, save_h5
from common.utils.utils_preprocessing import read_case_info
from src_T1.options.hyper_parameters import HP_Preprocessing

warnings.filterwarnings('ignore')

hp = HP_Preprocessing()
path_main = hp.path_main
path_raw = f'{path_main}/raw'
path_dataset = f'{path_main}/Dataset'
name_dataset_raw = 'ATM26_Track2'
meta = read_case_info(f'{path_raw}/ATM26_Track1/TrainBatch_CaseInfo.csv')
meta = np.array(meta)
meta_map = {str(row[0])[:7]:row for row in meta}

make_dir(path_dataset)
make_dir(f'{path_dataset}/ATM26_Shanghai')
make_dir(f'{path_dataset}/ATM26_LIDC')
make_dir(f'{path_dataset}/ATM26_EXACT09')

list_label = [i for i in listdir(f'{path_raw}/{name_dataset_raw}/labelsTr') if i.endswith('.mha')]
list_label = list_label[args.idx_s:args.idx_s+args.idx_n]
for n, label_file in enumerate(list_label):
    patient = label_file[:7]
    if patient not in meta_map:
        print(f'{patient}: no case info')
        continue
    row = meta_map[patient]
    idx_patient = str(row[3]).zfill(3)
    name_folder = None
    if 'Shanghai' in row[1]: name_folder = 'Shanghai'
    elif 'LIDC' in row[1]: name_folder = 'LIDC'
    elif 'EXACT' in row[1]: name_folder = 'EXACT'
    if name_folder is None:
        print(f'{patient}: invalid dataset name')
        continue
    save_folder = {'Shanghai':'ATM26_Shanghai', 'LIDC':'ATM26_LIDC', 'EXACT':'ATM26_EXACT09'}[name_folder]
    name_patient = f'{name_folder}_{idx_patient}'
    print(f'{n+1}/{len(list_label)}: {patient} -> {save_folder}/{name_patient}')

    _, CK = load_mha(f'{path_raw}/{name_dataset_raw}/labelsTr/{label_file}')
    CK = np.uint8(CK)

    path_patient = f'{path_dataset}/{save_folder}/{name_patient}'
    make_dir(path_patient)
    save_h5(f'{path_patient}/CK.h5', [CK], ['CK'])
    clear_output()

torch.cuda.empty_cache()
