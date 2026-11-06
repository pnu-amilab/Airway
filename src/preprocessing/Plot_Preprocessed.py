import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--idx_s', type=int, default=0)
parser.add_argument('--idx_n', type=int, default=1000000000)
parser.add_argument('--name_dset', type=str, default='')
parser.add_argument('--dset', type=str, default='')
args = parser.parse_args()

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

plt.style.use(plt.style.available[-3])
plt.rcParams['image.cmap'] = 'gray'

path_src = Path(__file__).resolve().parents[1]
os.chdir(path_src)
if str(path_src) not in sys.path: sys.path.append(str(path_src))

from common.utils.utils_common import listdir, load_h5, load_nii, make_dir
from src_T1.options.hyper_parameters import HP_Preprocessing

hp = HP_Preprocessing()
path_main = hp.path_main
path_dataset = f'{path_main}/Dataset'
path_fig_axial = f'{path_main}/Fig/Axial'
path_fig_projection = f'{path_main}/Fig/Projection'
make_dir(path_fig_axial)
make_dir(path_fig_projection)


def load_pt(path):
    if not os.path.exists(path): return None

    return torch.load(path, map_location='cpu')


def load_data(path_patient, name):
    path_h5 = f'{path_patient}/{name}.h5'
    if not os.path.exists(path_h5): return None

    return load_h5(path_h5, name)


def load_pm(path_patient):
    path_pm = f'{path_patient}/PM.nii.gz'
    if not os.path.exists(path_pm): return None
    _, PM = load_nii(path_pm)

    return PM


def get_fig_tag(name_dset, name_patient):
    if name_dset == 'ATM26_EXACT09': return f'ATM26_{name_patient}'
    if name_dset == 'EXACT09': return f'ATM22_{name_patient}'
    if name_dset == 'PNUYH' and name_patient.startswith('Patient_'): return f'PNUYH_{name_patient.split("_")[-1]}'
    if name_dset.startswith('ATM22_') or name_dset.startswith('ATM26_'):
        tag, name = name_dset.split('_', 1)
        if name_patient.startswith(f'{name}_'): return f'{tag}_{name_patient}'

    if name_patient.startswith(f'{name_dset}_'): return name_patient

    return f'{name_dset}_{name_patient}'


def get_pos(path_patient, MK, LM):
    MK_pos = load_pt(f'{path_patient}/MK_pos.pt')
    if MK_pos is not None and MK is not None: return MK_pos, MK
    LM_pos = load_pt(f'{path_patient}/LM_pos.pt')
    if LM_pos is not None and LM is not None: return LM_pos, LM>0

    return None, MK if MK is not None else LM>0


def crop_by_pos(arr, pos):
    if pos is None: return arr

    return arr[pos['idx_xm']:pos['idx_xM'], pos['idx_ym']:pos['idx_yM'], pos['idx_zm']:pos['idx_zM']]


def plot_patient(name_dset, name_patient):
    path_patient = f'{path_dataset}/{name_dset}/{name_patient}'
    fig_tag = get_fig_tag(name_dset, name_patient)

    CT = load_data(path_patient, 'CT')
    if CT is None: return False
    CV = load_data(path_patient, 'CV')
    LM = load_data(path_patient, 'LM')
    MK = load_data(path_patient, 'MK')
    SK = load_data(path_patient, 'SK')
    SM = load_data(path_patient, 'SM')
    MW = load_data(path_patient, 'MW')
    PM = load_pm(path_patient)

    k = int(len(CT)/2)
    zz = np.zeros_like(CT[k])
    imgs = [CT[k], CV[k] if CV is not None else zz, LM[k] if LM is not None else zz, PM[k] if PM is not None else zz, MK[k] if MK is not None else zz, SK[k] if SK is not None else zz, SM[k] if SM is not None else zz, MW[k] if MW is not None else zz]
    titles = ['CT', 'CV', 'LM', 'PM', 'MK', 'SK', 'SM', 'MW']
    cmaps = ['gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'jet', 'jet']
    plt.figure(figsize=(14, 7))
    for i in range(len(imgs)):
        plt.subplot(2, 4, i+1)
        plt.imshow(imgs[i], cmap=cmaps[i])
        plt.title(titles[i])
        plt.colorbar()
    plt.savefig(f'{path_fig_axial}/{fig_tag}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    if LM is None and MK is None: return True
    pos, mask = get_pos(path_patient, MK, LM)
    CT_temp = crop_by_pos(CT, pos)
    MK_temp = crop_by_pos(mask, pos)
    if len(CT_temp) == 0: return True
    k1 = int(len(CT_temp)*(1/2))
    k2 = int(len(CT_temp)*(3/5))
    imgs = [CT_temp[k1], MK_temp[k1], CT_temp[k2], MK_temp[k2], np.mean(CT_temp, axis=2), np.sum(MK_temp, axis=2)>0]
    plt.figure(figsize=(14, 14))
    for i in range(len(imgs)):
        plt.subplot(3, 2, i+1)
        plt.imshow(imgs[i], 'gray')
        plt.grid(c='y', lw=0.8, ls='--')
    plt.savefig(f'{path_fig_projection}/{fig_tag}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return True


list_entry = []
name_dset = args.dset if args.dset != '' else args.name_dset
list_dset = name_dset.split(',') if name_dset != '' else listdir(path_dataset)
for name_dset in list_dset:
    path_dset = f'{path_dataset}/{name_dset}'
    if not os.path.isdir(path_dset): continue
    for name_patient in listdir(path_dset):
        path_patient = f'{path_dset}/{name_patient}'
        if not os.path.isdir(path_patient): continue
        if not os.path.exists(f'{path_patient}/CT.h5'): continue
        list_entry.append([name_dset, name_patient])

idx_e = min(args.idx_s+args.idx_n, len(list_entry))
list_entry = list_entry[args.idx_s:idx_e]
for n, (name_dset, name_patient) in enumerate(list_entry):
    print(f'[{n+1:4.0f}/{len(list_entry):4.0f}] {name_dset}/{name_patient}')
    plot_patient(name_dset, name_patient)
