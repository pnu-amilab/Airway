import os
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from common.dataset.split_idx import get_dset_preset, get_split_idx
from common.dataset.sliding_window import make_centers
from common.utils.checkpoint_compat import load_options
from common.utils.utils_common import minMax_scaling, make_dir, save_nii
from src_T1.options.hyper_parameters import HP, load_local_HP
from src_T1.utils.utils_model import load_model


def load_kernel_info(epoch, kernel):
    hp_base = HP()
    load_local_HP(hp_base)
    hp = load_options(f'{hp_base.path_Res}/{kernel}/options.pt')
    hp.name = kernel
    hp.path_res = f'{hp.path_Res}/{kernel}'
    hp.path_model = f'{hp.path_res}/model'
    hp.path_inference = f'{hp.path_res}/inference'
    hp.path_evaluation = f'{hp.path_res}/evaluation'
    history_path = f'{hp.path_model}/history.pt'
    history = torch.load(history_path, map_location='cpu', weights_only=False) if os.path.exists(history_path) else []
    epoch_load = epoch
    epoch_tag = epoch
    if epoch == -1:
        epoch_load = -1
        epoch_tag = hp.epochs
    elif epoch == -2:
        epoch_load = -2
        epoch_tag = 'best'

    return hp, history, epoch_load, epoch_tag


def _get_edge_mask(shape, edge=65):
    res = np.ones(shape, dtype=np.float32)
    res[:, :edge, :] = 0
    res[:, -edge:, :] = 0
    res[:, :, :edge] = 0
    res[:, :, -edge:] = 0

    return res


def make_pe_axes_training_exact(shape, lm_pos):
    axes = []
    for axis, key_m, key_M in zip(range(3), ['idx_xm', 'idx_ym', 'idx_zm'], ['idx_xM', 'idx_yM', 'idx_zM']):
        denominator = lm_pos[key_M]-lm_pos[key_m]
        if denominator == 0: denominator = 1
        position = ((np.arange(shape[axis])-lm_pos[key_m])/denominator).astype(np.float32)
        theta = (2*position-1)*np.pi*0.5
        div_term = np.exp(np.arange(0, 2, 2)*-(np.log(10000.0)/2))
        axes.append((np.sin(theta[:, np.newaxis]*div_term)[:, 0].astype(np.float32), np.cos(theta[:, np.newaxis]*div_term)[:, 0].astype(np.float32)))

    return axes


class Dataset_AW_inference(Dataset):

    def __init__(self, hp, patient, name_dset=None):
        self.hp = hp
        self.window = hp.size_patch//2
        path = f'{hp.path_data}/{name_dset}/{patient}'
        with h5py.File(f'{path}/CT.h5', 'r') as f: self.data_CT = f['CT'][:]
        self.data_LM = None
        if hp.use_LM:
            with h5py.File(f'{path}/LM.h5', 'r') as f: self.data_LM = np.float32(f['LM'][:]>0)
        self.data_CV = None
        if hp.use_CV:
            with h5py.File(f'{path}/CV.h5', 'r') as f: self.data_CV = f['CV'][:]
        self.data_PE = torch.load(f'{path}/LM_pos.pt', map_location='cpu', weights_only=False) if hp.use_PE else None
        self.shape = self.data_CT.shape
        self.pe_axes = make_pe_axes_training_exact(self.shape, self.data_PE) if hp.use_PE else None
        self.list_idx_x = make_centers(self.shape[0], hp.size_patch, hp.overlap_inference)
        self.list_idx_y = make_centers(self.shape[1], hp.size_patch, hp.overlap_inference)
        self.list_idx_z = make_centers(self.shape[2], hp.size_patch, hp.overlap_inference)
        self.list_idx = np.array([[x, y, z] for x in self.list_idx_x for y in self.list_idx_y for z in self.list_idx_z], dtype=np.int32)
        self.m = hp.m
        self.M = hp.M

    def __len__(self):
        return len(self.list_idx)

    def __getitem__(self, idx):
        x, y, z = self.list_idx[idx]
        w = self.window
        xs, xe = x-w, x+w
        ys, ye = y-w, y+w
        zs, ze = z-w, z+w
        X_data = np.zeros([self.hp.dim_in, self.hp.size_patch, self.hp.size_patch, self.hp.size_patch], dtype=np.float32)
        ct = np.clip(self.data_CT[xs:xe, ys:ye, zs:ze], self.m, self.M)
        X_data[0] = minMax_scaling(ct, m=self.m, M=self.M)
        l = 1
        if self.hp.use_LM:
            X_data[l] = self.data_LM[xs:xe, ys:ye, zs:ze]
            l += 1
        if self.hp.use_CV:
            X_data[l] = self.data_CV[xs:xe, ys:ye, zs:ze]
            l += 1
        if self.hp.use_PE:
            X_data[l] = self.pe_axes[0][0][xs:xe, None, None]
            X_data[l+1] = self.pe_axes[0][1][xs:xe, None, None]
            X_data[l+2] = self.pe_axes[1][0][None, ys:ye, None]
            X_data[l+3] = self.pe_axes[1][1][None, ys:ye, None]
            X_data[l+4] = self.pe_axes[2][0][None, None, zs:ze]
            X_data[l+5] = self.pe_axes[2][1][None, None, zs:ze]

        return torch.from_numpy(X_data), torch.tensor([x, y, z], dtype=torch.int32)


def _sliding_window_inference(test_set, model, hp, batch_size, num_workers, device):
    test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    size = hp.size_patch//2
    probability = np.zeros(test_set.shape, dtype=np.float32)
    weight_sum = np.zeros(test_set.shape, dtype=np.float32)
    for x, idx in test_loader:
        x = x.to(device, non_blocking=True)
        p = model(x)[1][:, 0].detach().to(dtype=torch.float32).cpu().numpy()
        idx = idx.numpy()
        for batch_idx, i in enumerate(idx):
            xs, xe = i[0]-size, i[0]+size
            ys, ye = i[1]-size, i[1]+size
            zs, ze = i[2]-size, i[2]+size
            probability[xs:xe, ys:ye, zs:ze] += p[batch_idx]
            weight_sum[xs:xe, ys:ye, zs:ze] += 1
    probability = probability/np.maximum(weight_sum, 1e-6)
    probability = probability*_get_edge_mask(test_set.shape, edge=hp.inference_edge)

    return probability


def inference(epoch, kernel, batch_size=0, n=0, num_workers=None, dset_test=None, overlap_inference=32):
    hp, _, epoch_load, epoch_tag = load_kernel_info(epoch, kernel)
    hp.overlap_inference = overlap_inference
    print(f'[{str(epoch_tag):>4} epoch]: {hp.name} --inference | overlap={hp.overlap_inference} | stride={hp.size_patch-hp.overlap_inference}', flush=True)
    if batch_size in [None, 0]: batch_size = hp.batch_size_inference
    if num_workers is None: num_workers = min(hp.num_workers, 4)
    dset_preset = get_dset_preset(hp, dset_test=dset_test)
    list_entry = get_split_idx(dset_preset)['test']
    if n > 0: list_entry = list_entry[:n]
    device = torch.device(hp.device if torch.cuda.is_available() else 'cpu')
    model = load_model(hp, epoch_load=epoch_load).eval().to(device)
    make_dir(f'{hp.path_inference}/epoch_{epoch_tag}')
    with torch.inference_mode():
        for patient_idx, (name_dset, patient) in enumerate(list_entry):
            print(f'[{patient_idx+1:3.0f}/{len(list_entry):3.0f}]: {name_dset}/{patient:<13}')
            path = f'{hp.path_inference}/epoch_{epoch_tag}/{name_dset}/{patient}_p.nii.gz'
            test_set = Dataset_AW_inference(hp, patient, name_dset=name_dset)
            probability = _sliding_window_inference(test_set, model, hp, batch_size, num_workers, device)
            path_patient = f'{hp.path_data}/{name_dset}/{patient}'
            meta = torch.load(f'{path_patient}/meta_info.pt', map_location='cpu', weights_only=False)
            make_dir(f'{hp.path_inference}/epoch_{epoch_tag}/{name_dset}')
            save_nii(path, probability, meta['pixdim'], meta['affine_tform'])
    torch.cuda.empty_cache()
