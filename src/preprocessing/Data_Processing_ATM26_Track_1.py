import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument('--idx_s', type=int, default=0)
parser.add_argument('--idx_n', type=int, default=20)
parser.add_argument('--LW', action='store_true')
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
import SimpleITK as sitk
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
from common.utils.utils_common import listdir, load_mha, make_dir, save_h5, save_nii
from src_T1.options.hyper_parameters import HP_Preprocessing
from common.utils.utils_preprocessing import *

torch.set_default_dtype(torch.float32)
GPU = torch.device('cuda')
CPU = torch.device('cpu')
print(torch.cuda.is_available(), ': ', torch.cuda.get_device_name(0), args.GPU_NUM)
torch.cuda.empty_cache()

hp = HP_Preprocessing()
hp.use_LW = hp.use_LW or args.LW
for i in hp.__dict__: print(f'{i}: {hp.__dict__[i]}')

path_main = hp.path_main
path_raw = f'{path_main}/raw'
path_dataset = f'{path_main}/Dataset'
path_fig_axial = f'{path_main}/Fig/Axial'
path_fig_projection = f'{path_main}/Fig/Projection'
name_dataset_raw = 'ATM26_Track1'
meta = read_case_info(f'{path_raw}/{name_dataset_raw}/TrainBatch_CaseInfo.csv')
meta = np.array(meta)
meta_map = {str(row[0]).replace('.mha', ''):row for row in meta}
make_dir(path_dataset)
make_dir(f'{path_dataset}/ATM26_Shanghai')
make_dir(f'{path_dataset}/ATM26_LIDC')
make_dir(f'{path_dataset}/ATM26_EXACT09')
make_dir(f'{path_fig_axial}')
make_dir(f'{path_fig_projection}')

model_expand = nn.Upsample(scale_factor=hp.scale_factor/1, mode='trilinear', align_corners=False)
model_reduce = nn.Upsample(scale_factor=1/hp.scale_factor, mode='trilinear', align_corners=False)
list_image = [i for i in listdir(f'{path_raw}/{name_dataset_raw}/imagesTr') if i.endswith('.mha')]
list_image = list_image[args.idx_s:args.idx_s+args.idx_n]
for n, image_file in enumerate(list_image):
    gc.collect()
    torch.cuda.empty_cache()
    patient = image_file[:-4]
    sample_name = patient[:7]
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
    fig_tag = f'ATM26_{name_patient}'
    print(f'{n+1}/{len(list_image)}: {patient} -> {save_folder}/{name_patient}')
    print('    01.CT')
    CT_data, CT = load_mha(f'{path_raw}/{name_dataset_raw}/imagesTr/{image_file}')
    CT = np.float32(CT)

    print('    02.MK')
    _, MK = load_mha(f'{path_raw}/{name_dataset_raw}/labelsTr/{patient[:7]}.mha')
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

    if hp.use_LW:
        print('    09b.LW: Local Imbalance Weight Map')
        LW = get_local_imbalance_weight(MK)

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
    spacing = CT_data.GetSpacing()
    origin = CT_data.GetOrigin()
    direction = CT_data.GetDirection()
    pixdim = np.array([1, spacing[2], spacing[1], spacing[0], 0, 0, 0, 0], dtype=np.float32)
    affine_tform = np.eye(4, dtype=np.float32)
    affine_tform[0, 0] = spacing[2]
    affine_tform[1, 1] = spacing[1]
    affine_tform[2, 2] = spacing[0]
    affine_tform[:3, 3] = np.array(origin[::-1], dtype=np.float32)
    meta_info = {'sample_name':sample_name,
                 'pixdim':pixdim,
                 'affine_tform':affine_tform,
                 'shape':CT.shape,
                 'spacing':spacing,
                 'origin':origin,
                 'direction':direction,
                 'meta':row}

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
    if hp.use_LW: save_h5(f'{path_patient}/LW.h5', [LW], ['LW'])
    save_h5(f'{path_patient}/MK_parsing.h5', [MK_parsing], ['MK_parsing'])
    torch.save(Idx_SK, f'{path_patient}/Idx_SK.pt')
    torch.save(LM_pos, f'{path_patient}/LM_pos.pt')
    torch.save(MK_pos, f'{path_patient}/MK_pos.pt')
    torch.save(MK_gen, f'{path_patient}/MK_gen.pt')
    torch.save(meta_info, f'{path_patient}/meta_info.pt')
    save_nii(f'{path_patient}/GK.nii.gz', GK, meta_info['pixdim'], meta_info['affine_tform'])
    save_nii(f'{path_patient}/PM.nii.gz', PM, meta_info['pixdim'], meta_info['affine_tform'])
    save_nii(f'{path_patient}/MK_parsing.nii.gz', MK_parsing, meta_info['pixdim'], meta_info['affine_tform'])
    save_nii(f'{path_patient}/sk.nii.gz', sk, meta_info['pixdim'], meta_info['affine_tform'])
    clear_output()
