import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument('--idx_s', type=int, default=0)
parser.add_argument('--idx_n', type=int, default=20)
args = parser.parse_args()

import os
import sys
from pathlib import Path
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_NUM)

import gc
import cv2
import numpy as np
import nibabel as nib
import warnings
import matplotlib.pyplot as plt
from copy import deepcopy
from lungmask import LMInferer
from IPython.display import clear_output

import torch
import torch.nn as nn

plt.style.use(plt.style.available[-3])
plt.rcParams['image.cmap'] = 'gray'
warnings.filterwarnings('ignore')

path_src = Path(__file__).resolve().parents[1]
os.chdir(path_src)
if str(path_src) not in sys.path: sys.path.append(str(path_src))

from common.utils.utils_common import listdir, load_nii, make_dir, save_h5, save_nii
from src_T1.options.hyper_parameters import HP_Preprocessing
from common.utils.utils_preprocessing import *

torch.set_default_dtype(torch.float32)
GPU = torch.device('cuda')
CPU = torch.device('cpu')
print(torch.cuda.is_available(), ': ', torch.cuda.get_device_name(0), args.GPU_NUM)
torch.cuda.empty_cache()

hp = HP_Preprocessing()
for i in hp.__dict__: print(f'{i}: {hp.__dict__[i]}')

path_main = hp.path_main
path_raw = f'{path_main}/raw'
path_dataset = f'{path_main}/Dataset'
path_fig_axial = f'{path_main}/Fig/Axial'
path_fig_projection = f'{path_main}/Fig/Projection'
name_dataset = 'ATM22_ValidBatch'
name_folder = 'ATM22_Valid'
make_dir(path_dataset)
make_dir(f'{path_dataset}/{name_folder}')
make_dir(path_fig_axial)
make_dir(path_fig_projection)

model_expand = nn.Upsample(scale_factor=hp.scale_factor/1, mode='trilinear', align_corners=False)
model_reduce = nn.Upsample(scale_factor=1/hp.scale_factor, mode='trilinear', align_corners=False)
list_data = [_ for _ in listdir(f'{path_raw}/{name_dataset}') if 'ATM' in _]
idx_e = min(args.idx_s+args.idx_n, len(list_data))
for n in range(args.idx_s, idx_e):
    gc.collect()
    torch.cuda.empty_cache()
    patient = list_data[n][:-7]
    idx_patient = str(list_data[n][4:7]).zfill(3)
    sample_name = f'ATM22_Valid_{idx_patient}'
    name_patient = sample_name
    fig_tag = name_patient
    print(f'{n-args.idx_s+1}/{idx_e-args.idx_s}: {patient} -> {name_folder}/{name_patient}')

    print('    01.CT')
    CT_data, CT = load_nii(f'{path_raw}/{name_dataset}/{patient}.nii.gz')
    CT = np.float32(CT)

    print('    03.LM: Lung Mask')
    model_LM = LMInferer(modelname='LTRCLobes', fillmodel='R231')

    LM = model_LM.apply(CT)
    LM_binary = np.uint8(LM>0)

    print('    04.PM: Polygon LM')
    PM = polygon_all(deepcopy(LM_binary))

    print('    05.CV: CT Vessel')
    CT_w = np.clip(deepcopy(CT), -1000, -300)
    CV = LM_binary*CT_w
    CV[CV == 0] = -1000
    CV[CV >= -500] = 1
    CV[CV < -500] = 0

    CV_temp = []
    for k in range(len(CV)):
        cv = CV[k]
        lm = np.uint8(LM_binary[k]*255)
        C, _ = cv2.findContours(lm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = np.zeros_like(cv)
        cv2.drawContours(res, C, -1, (1), thickness=3)
        res = res*2 - 1
        res = res*(-1)
        res = (res+1)/2

        cv = res*cv
        CV_temp.append(cv)
    CV = np.array(CV_temp)

    print('    10.LM_pos')
    LM_x = np.sum(np.sum(LM_binary, axis=2), axis=1)>0
    LM_y = np.sum(np.sum(LM_binary, axis=2), axis=0)>0
    LM_z = np.sum(np.sum(LM_binary, axis=1), axis=0)>0

    idx_x = np.where(LM_x>0)[0]
    idx_xM = np.clip(idx_x.max(), 0, len(LM_x))
    idx_xm = np.clip(idx_x.min(), 0, len(LM_x))
    idx_y = np.where(LM_y>0)[0]
    idx_yM = np.clip(idx_y.max(), 0, len(LM_y))
    idx_ym = np.clip(idx_y.min(), 0, len(LM_y))
    idx_z = np.where(LM_z>0)[0]
    idx_zM = np.clip(idx_z.max(), 0, len(LM_z))
    idx_zm = np.clip(idx_z.min(), 0, len(LM_z))

    LM_pos = {'idx_xm':idx_xm,
              'idx_xM':idx_xM,
              'idx_ym':idx_ym,
              'idx_yM':idx_yM,
              'idx_zm':idx_zm,
              'idx_zM':idx_zM}

    print('    13.Meta Information')
    data = CT_data
    meta_info = {'sample_name':sample_name,
                 'pixdim':data.header['pixdim'],
                 'affine_tform':data.affine,
                 'shape':CT.shape,
                 'meta':patient}

    print('    15.Plotting')
    zz = np.zeros([512, 512], dtype=np.int16)
    k = int(len(CT)/2)
    imgs = [CT[k], CV[k], LM[k], PM[k], zz, zz, zz, zz]
    titles = ['CT', 'CV', 'LM', 'PM', 'MK', 'SK', 'SM', 'MW']
    cmaps = ['gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'jet', 'jet']
    plt.figure(figsize=(14, 7))
    for i in range(len(imgs)):
        plt.subplot(2, 4 ,i+1)
        plt.imshow(imgs[i], cmap=cmaps[i])
        plt.title(titles[i])
        plt.colorbar()
    plt.savefig(f'{path_fig_axial}/{fig_tag}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    CT_temp = CT[LM_pos['idx_xm']:LM_pos['idx_xM'], LM_pos['idx_ym']:LM_pos['idx_yM'], LM_pos['idx_zm']:LM_pos['idx_zM']]
    MK_temp = CT[LM_pos['idx_xm']:LM_pos['idx_xM'], LM_pos['idx_ym']:LM_pos['idx_yM'], LM_pos['idx_zm']:LM_pos['idx_zM']]
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

    print('    @@ Save @@')
    CT = np.int16(CT)
    LM = np.uint8(LM)
    PM = np.uint8(PM)
    CV = np.uint8(CV)
    LM_pos = LM_pos
    meta_info = meta_info
    path_patient = f'{path_dataset}/{name_folder}/{name_patient}'
    make_dir(path_patient)
    save_h5(f'{path_patient}/CT.h5', [CT], ['CT'])
    save_h5(f'{path_patient}/LM.h5', [LM], ['LM'])
    save_h5(f'{path_patient}/CV.h5', [CV], ['CV'])
    torch.save(LM_pos, f'{path_patient}/LM_pos.pt')
    torch.save(meta_info, f'{path_patient}/meta_info.pt')
    save_nii(f'{path_patient}/PM.nii.gz', PM, meta_info['pixdim'], meta_info['affine_tform'])
    clear_output()
