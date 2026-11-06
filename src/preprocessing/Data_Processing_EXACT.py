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
import pydicom
import warnings
import matplotlib.pyplot as plt
from copy import deepcopy
from lungmask import LMInferer
from scipy.ndimage import grey_dilation
from IPython.display import clear_output
from skimage.filters import gaussian
from skimage.morphology import skeletonize_3d

import torch
import torch.nn as nn

plt.style.use(plt.style.available[-3])
plt.rcParams['image.cmap'] = 'gray'
warnings.filterwarnings('ignore')

path_src = Path(__file__).resolve().parents[1]
os.chdir(path_src)
if str(path_src) not in sys.path: sys.path.append(str(path_src))

from common.utils.tree_parsing import large_connected_domain, tree_parse
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
name_dataset = 'TrainBatch2'
meta = np.array(read_case_info(f'{path_raw}/ATM22_TrainBatch2/{name_dataset}_CaseInfo.csv'))
make_dir(path_dataset)
make_dir(f'{path_dataset}/EXACT09')
make_dir(f'{path_fig_axial}')
make_dir(f'{path_fig_projection}')

model_expand = nn.Upsample(scale_factor=hp.scale_factor/1, mode='trilinear', align_corners=False)
model_reduce = nn.Upsample(scale_factor=1/hp.scale_factor, mode='trilinear', align_corners=False)
idx_e = min(args.idx_s+args.idx_n, len(meta))
for n in range(args.idx_s, idx_e):
    gc.collect()
    torch.cuda.empty_cache()
    patient = meta[n][0]
    sample_name = patient[:7]
    idx_patient = str(meta[n][3]).zfill(3)
    if 'Shanghai' in meta[n][1]: continue
    if 'LIDC'     in meta[n][1]: continue
    if 'EXACT'    in meta[n][1]: name_folder = 'EXACT'
    save_folder = 'EXACT09'
    name_patient = f'{name_folder}_{idx_patient}'
    fig_tag = f'ATM22_{name_patient}'
    print(f'{n-args.idx_s+1}/{idx_e-args.idx_s}: {patient} -> {save_folder}/{name_patient}')

    print('    01.CT')
    CT = []
    CT_loc = []
    for path_ct in listdir(f'{path_raw}/EXACT09/{meta[n][2]}'):
        ct_data = pydicom.read_file(f'{path_raw}/EXACT09/{meta[n][2]}/{path_ct}')
        ct = ct_data.pixel_array
        slope = ct_data.get((0x0028, 0x1053)).value
        intercept = ct_data.get((0x0028, 0x1052)).value
        ct = ct*slope + intercept
        CT.append(ct)
        CT_loc.append(ct_data.ImagePositionPatient[-1])
    CT = np.stack(CT)
    CT_loc = np.array(CT_loc)
    CT = CT[np.argsort(CT_loc)][::-1]

    print('    02.MK')
    MK_data, MK = load_nii(f'{path_raw}/ATM22_TrainBatch2/labelsTr/{meta[n][0]}.nii.gz')
    MK = np.float32(MK)
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

    print('    06.sk: skeleton')
    sk = skeletonize_3d(large_connected_domain(MK))
    sk = np.float32(sk>0)

    print('    07.SK: Skeleton')
    MK_expand = model_expand(torch.tensor(MK.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    SK_expand = skeletonize_3d(np.float32(MK_expand>0).squeeze())/255
    SK = model_reduce(torch.tensor(SK_expand.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    SK = np.clip(np.float32(SK>0).squeeze(), 0, 1)
    gc.collect()

    print('    08.SM: Soft Mask')
    SM = get_soft_label(MK, kernel_size=hp.kernel_size_SM, sigma=hp.sigma_SM)
    SM = np.clip(SM+SK, 0, 1)
    SM = SM*MK

    print('    09.MW: Mask Weight Map')
    MK_x = np.sum(MK, axis=0, keepdims=True)
    MK_y = np.sum(MK, axis=1, keepdims=True)
    MK_z = np.sum(MK, axis=2, keepdims=True)

    MK_x = MK_x*SK
    MK_y = MK_y*SK
    MK_z = MK_z*SK

    MW = np.stack([MK_x, MK_y, MK_z])
    MW = np.min(MW, axis=0)
    idx_background = MW==0
    MW = np.clip(MW, hp.MW_min, hp.MW_max)
    MW = ((1-hp.MW_alpha)/(hp.MW_max-hp.MW_min))*(MW-hp.MW_min) + hp.MW_alpha
    MW[idx_background] = 0

    MW = grey_dilation(MW, structure=np.ones([hp.kernel_Size_MW, hp.kernel_Size_MW, hp.kernel_Size_MW]))
    MW = gaussian(MW, sigma=1, mode='reflect')
    MW = MK*MW
    MW[MK==0] = 1/hp.MW_beta

    print('    10.Idx_SK: Index of SK')
    Idx_SK = np.where(SK==1)

    print('    11.LM_pos')
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

    print('    12.MK_pos')
    MK_x = np.sum(np.sum(MK, axis=2), axis=1)
    MK_y = np.sum(np.sum(MK, axis=2), axis=0)
    MK_z = np.sum(np.sum(MK, axis=1), axis=0)

    idx_x = np.where(MK_x>0)[0]
    idx_xM = np.clip(idx_x.max(), 0, len(MK_x))
    idx_xm = np.clip(idx_x.min(), 0, len(MK_x))
    idx_y = np.where(MK_y>0)[0]
    idx_yM = np.clip(idx_y.max(), 0, len(MK_y))
    idx_ym = np.clip(idx_y.min(), 0, len(MK_y))
    idx_z = np.where(MK_z>0)[0]
    idx_zM = np.clip(idx_z.max(), 0, len(MK_z))
    idx_zm = np.clip(idx_z.min(), 0, len(MK_z))

    MK_pos = {'idx_xM':idx_xM,
              'idx_xm':idx_xm,
              'idx_yM':idx_yM,
              'idx_ym':idx_ym,
              'idx_zM':idx_zM,
              'idx_zm':idx_zm}

    print('    13.Label Tree Parsing')
    MK_parsing, MK_gen, GK = tree_parse(MK)

    print('    14.Meta Information')
    data = MK_data
    meta_info = {'sample_name':sample_name,
                 'pixdim':data.header['pixdim'],
                 'affine_tform':data.affine,
                 'shape':CT.shape,
                 'meta':meta[n]}

    print('    15.Plotting')
    k = int(len(CT)/2)
    imgs = [CT[k], CV[k], LM[k], PM[k], MK[k], SK[k], SM[k], MW[k]]
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

    CT_temp = CT[MK_pos['idx_xm']:MK_pos['idx_xM'], MK_pos['idx_ym']:MK_pos['idx_yM'], MK_pos['idx_zm']:MK_pos['idx_zM']]
    MK_temp = MK[MK_pos['idx_xm']:MK_pos['idx_xM'], MK_pos['idx_ym']:MK_pos['idx_yM'], MK_pos['idx_zm']:MK_pos['idx_zM']]
    k1 = int(len(CT_temp)*(1/2))
    k2 = int(len(CT_temp)*(3/5))
    imgs = [CT_temp[k1], MK_temp[k1], CT_temp[k2], MK_temp[k2], np.mean(CT_temp, axis=2), np.sum(MK, axis=2)>0]
    plt.figure(figsize=(14, 14))
    for i in range(len(imgs)):
        plt.subplot(3, 2, i+1)
        plt.imshow(imgs[i], 'gray')
        plt.grid(c='y', lw=0.8, ls='--')
    plt.savefig(f'{path_fig_projection}/{fig_tag}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    print('    @@ Save @@')
    CT = np.int16(CT)
    MK = np.uint8(MK)
    GK = np.uint8(GK)
    LM = np.uint8(LM)
    PM = np.uint8(PM)
    CV = np.uint8(CV)
    SK = np.uint8(SK)
    sk = np.uint8(sk)
    SM = np.float16(SM)
    MW = np.float16(MW)
    Idx_SK = np.int16(Idx_SK)
    MK_gen = np.int16(MK_gen)
    MK_parsing = np.int16(MK_parsing)
    LM_pos = LM_pos
    MK_pos = MK_pos
    meta_info = meta_info
    path_patient = f'{path_dataset}/{save_folder}/{name_patient}'
    make_dir(path_patient)
    save_h5(f'{path_patient}/CT.h5', [CT], ['CT'])
    save_h5(f'{path_patient}/MK.h5', [MK], ['MK'])
    save_h5(f'{path_patient}/LM.h5', [LM], ['LM'])
    save_h5(f'{path_patient}/CV.h5', [CV], ['CV'])
    save_h5(f'{path_patient}/SK.h5', [SK], ['SK'])
    save_h5(f'{path_patient}/SM.h5', [SM], ['SM'])
    save_h5(f'{path_patient}/MW.h5', [MW], ['MW'])
    save_h5(f'{path_patient}/MK_parsing.h5', [MK_parsing], ['MK_parsing'])
    torch.save(Idx_SK, f'{path_patient}/Idx_SK.pt')
    torch.save(LM_pos, f'{path_patient}/LM_pos.pt')
    torch.save(MK_pos, f'{path_patient}/MK_pos.pt')
    torch.save(MK_gen, f'{path_patient}/MK_gen.pt')
    torch.save(meta_info, f'{path_patient}/meta_info.pt')
    save_nii(f'{path_patient}/GK.nii.gz', GK, data.header['pixdim'], data.affine)
    save_nii(f'{path_patient}/PM.nii.gz', PM, meta_info['pixdim'], meta_info['affine_tform'])
    save_nii(f'{path_patient}/MK_parsing.nii.gz', MK_parsing, data.header['pixdim'], data.affine)
    save_nii(f'{path_patient}/sk.nii.gz', sk, data.header['pixdim'], data.affine)
    clear_output()
