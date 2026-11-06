import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common.dataset.sliding_window import make_centers
from common.dataset.split_idx import get_split_idx_T2
from common.dataset.utils_dataloader import Positional_Encoding
from common.utils.utils_common import make_dir, minMax_scaling, save_nii
from src_T2.utils.track1 import load_track1_cache
from src_T2.utils.utils_dataloader import one_hot_lobe_mask
from src_T2.utils.utils_io import load_kernel_info_T2
from src_T2.utils.utils_model import load_model_T2


def _build_axis_ownership(centers, window, length):
    ownership = {}
    ends = np.empty(len(centers), dtype=np.int32)
    for idx, center in enumerate(centers):
        start = 0 if idx == 0 else (centers[idx-1]+center)//2
        end = length if idx == len(centers)-1 else (center+centers[idx+1])//2
        patch_start = center-window
        ownership[int(center)] = (slice(start, end), slice(start-patch_start, end-patch_start))
        ends[idx] = end

    return ownership, ends


def _select_sparse_centers(mask, centers_x, centers_y, centers_z, ends_x, ends_y, ends_z):
    coordinates = np.argwhere(mask)
    if coordinates.shape[0] == 0: return np.empty((0, 3), dtype=np.int32)
    idx_x = np.searchsorted(ends_x, coordinates[:, 0], side='right')
    idx_y = np.searchsorted(ends_y, coordinates[:, 1], side='right')
    idx_z = np.searchsorted(ends_z, coordinates[:, 2], side='right')
    num_y = len(centers_y)
    num_z = len(centers_z)
    linear_idx = np.unique((idx_x*num_y+idx_y)*num_z+idx_z)
    active_x = linear_idx//(num_y*num_z)
    remainder = linear_idx%(num_y*num_z)
    active_y = remainder//num_z
    active_z = remainder%num_z
    centers_x = np.asarray(centers_x, dtype=np.int32)
    centers_y = np.asarray(centers_y, dtype=np.int32)
    centers_z = np.asarray(centers_z, dtype=np.int32)

    return np.stack([centers_x[active_x], centers_y[active_y], centers_z[active_z]], axis=1).astype(np.int32, copy=False)


class Dataset_Conditional_T2_Inference(Dataset):

    def __init__(self, hp, name_dset, patient, probability, mask):
        self.hp = hp
        self.window = hp.size_patch//2
        self.probability = np.asarray(probability, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=np.float32)
        self.shape = self.mask.shape

        self.list_idx_x = make_centers(self.shape[0], hp.size_patch, hp.overlap_inference)
        self.list_idx_y = make_centers(self.shape[1], hp.size_patch, hp.overlap_inference)
        self.list_idx_z = make_centers(self.shape[2], hp.size_patch, hp.overlap_inference)
        self.ownership_x, ends_x = _build_axis_ownership(self.list_idx_x, self.window, self.shape[0])
        self.ownership_y, ends_y = _build_axis_ownership(self.list_idx_y, self.window, self.shape[1])
        self.ownership_z, ends_z = _build_axis_ownership(self.list_idx_z, self.window, self.shape[2])
        self.list_idx = _select_sparse_centers(self.mask > 0, self.list_idx_x, self.list_idx_y, self.list_idx_z, ends_x, ends_y, ends_z)

        path = f'{hp.path_data}/{name_dset}/{patient}'
        with h5py.File(f'{path}/CT.h5', 'r') as f: self.data_CT = f['CT'][:]
        if hp.use_LM:
            with h5py.File(f'{path}/LM.h5', 'r') as f: self.data_LM = np.asarray(f['LM'][:], dtype=np.uint8)
        if hp.use_CV:
            with h5py.File(f'{path}/CV.h5', 'r') as f: self.data_CV = f['CV'][:]
        if hp.use_PE:
            self.data_PE = torch.load(f'{path}/LM_pos.pt', map_location='cpu', weights_only=False)
            self.pos_encoding = Positional_Encoding(size=hp.size_patch)

    def __len__(self):
        return len(self.list_idx)

    def __getitem__(self, idx):
        x, y, z = self.list_idx[idx]
        w = self.window
        crop = np.s_[x-w:x+w, y-w:y+w, z-w:z+w]
        X_data = np.zeros([self.hp.dim_in, self.hp.size_patch, self.hp.size_patch, self.hp.size_patch], dtype=np.float32)
        X_data[0] = minMax_scaling(np.clip(self.data_CT[crop], self.hp.m, self.hp.M), m=self.hp.m, M=self.hp.M)
        X_data[1] = self.probability[crop]
        X_data[2] = self.mask[crop]
        if self.hp.use_LM: X_data[3:8] = one_hot_lobe_mask(self.data_LM[crop], self.hp.num_lobes)
        if self.hp.use_CV: X_data[8] = self.data_CV[crop]
        if self.hp.use_PE: X_data[9:15] = self.pos_encoding.get_PE(self.data_PE, [x, y, z])

        return torch.from_numpy(X_data), torch.tensor([x, y, z], dtype=torch.int32)

    def get_ownership(self, center):
        return self.ownership_x[int(center[0])], self.ownership_y[int(center[1])], self.ownership_z[int(center[2])]


def _run_label_volume(test_set, model, hp, batch_size, num_workers, device):
    loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    label = np.zeros(test_set.shape, dtype=np.uint8)
    for x, idx in tqdm(loader, desc='T2', leave=False):
        x = x.to(device, non_blocking=True)
        with torch.inference_mode():
            _, logits, _ = model(x)
        prediction = (torch.argmax(logits, dim=1)+1).cpu().numpy().astype(np.uint8)
        idx = idx.numpy()
        for batch_idx, center in enumerate(idx):
            (gx, lx), (gy, ly), (gz, lz) = test_set.get_ownership(center)
            label[gx, gy, gz] = prediction[batch_idx][lx, ly, lz]

    return label*(test_set.mask > 0)


def inference_T2(epoch, kernel, batch_size=0, n=0, num_workers=4, overlap_inference=32):
    hp, _, epoch_load, epoch_tag = load_kernel_info_T2(epoch, kernel)
    hp.overlap_inference = overlap_inference
    print(f'[{str(epoch_tag):>4} epoch]: {kernel} --inference', flush=True)
    if batch_size == 0: batch_size = hp.batch_size_inference
    list_entry = get_split_idx_T2(hp.dset_preset, use_extra_train=hp.use_extra_train_T2)['test']
    if n > 0: list_entry = list_entry[:n]
    device = torch.device(hp.device if torch.cuda.is_available() else 'cpu')
    model = load_model_T2(hp, epoch_load=epoch_load).eval().to(device)
    path_epoch = f'{hp.path_inference}/epoch_{epoch_tag}'
    make_dir(path_epoch)

    for patient_idx, (name_dset, patient) in enumerate(list_entry):
        print(f'[{patient_idx+1:3d}/{len(list_entry):3d}] {name_dset}/{patient}')
        path_save = f'{path_epoch}/{name_dset}/{patient}_label.nii.gz'
        probability, mask = load_track1_cache(hp.path_cache, name_dset, patient)
        test_set = Dataset_Conditional_T2_Inference(hp, name_dset, patient, probability, mask)
        label = _run_label_volume(test_set, model, hp, batch_size, num_workers, device)
        path_patient = f'{hp.path_data}/{name_dset}/{patient}'
        meta = torch.load(f'{path_patient}/meta_info.pt', map_location='cpu', weights_only=False)
        make_dir(f'{path_epoch}/{name_dset}')
        save_nii(path_save, label, meta['pixdim'], meta['affine_tform'])
    torch.cuda.empty_cache()
