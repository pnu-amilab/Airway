import h5py
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from common.dataset.split_idx import get_split_idx
from common.utils.utils_common import minMax_scaling


def sampling_idx(shape, Idx_SK, s=32, p=0.50, var=20):
    if np.random.rand() <= p:
        center = [np.random.randint(s, shape[axis]-s) for axis in range(3)]
    else:
        while True:
            n = np.random.randint(len(Idx_SK[0]))
            center = [int(Idx_SK[axis][n])+np.random.randint(var, var+1) for axis in range(3)]
            if all(s <= center[axis] <= shape[axis]-s for axis in range(3)): break

    return tuple(center)

class Positional_Encoding():

    def __init__(self, size=64, d_model=6, alpha=0.5):
        self.size = size
        self.window = size//2
        self.d_model = d_model
        self.alpha = alpha

    def positional_encoding_1d(self, position, d_model):
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((position.shape[0], d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position[:, np.newaxis] * div_term)
        pe[:, 1::2] = np.cos(position[:, np.newaxis] * div_term)

        return pe

    def positional_encoding_3d(self, x, y, z, d_model):
        d_model = d_model//3
        pe_x = self.positional_encoding_1d(x, d_model)
        pe_y = self.positional_encoding_1d(y, d_model)
        pe_z = self.positional_encoding_1d(z, d_model)
        pe = np.concatenate((pe_x, pe_y, pe_z), axis=-1)

        return pe

    def get_PE(self, LM_pos, center):
        x_PE = ((np.arange(center[0]-self.window, center[0]+self.window)-LM_pos['idx_xm'])/(LM_pos['idx_xM']-LM_pos['idx_xm'])).astype(np.float32)
        y_PE = ((np.arange(center[1]-self.window, center[1]+self.window)-LM_pos['idx_ym'])/(LM_pos['idx_yM']-LM_pos['idx_ym'])).astype(np.float32)
        z_PE = ((np.arange(center[2]-self.window, center[2]+self.window)-LM_pos['idx_zm'])/(LM_pos['idx_zM']-LM_pos['idx_zm'])).astype(np.float32)
        xx, yy, zz = np.meshgrid(x_PE, y_PE, z_PE, indexing='ij')
        xx = ((2 * xx - 1) * np.pi * self.alpha).flatten()
        yy = ((2 * yy - 1) * np.pi * self.alpha).flatten()
        zz = ((2 * zz - 1) * np.pi * self.alpha).flatten()

        pe_3d = self.positional_encoding_3d(xx, yy, zz, self.d_model)
        pe_3d = pe_3d.reshape((self.size, self.size, self.size, self.d_model))
        pe_3d = np.transpose(pe_3d, [3, 0, 1, 2])

        return pe_3d


class Dataset_AW(Dataset):

    def __init__(self, hp, phase='train', p=0.50, n=0, max_open_files=16):
        self.hp = hp
        self.phase = phase
        self.max_open_files = max_open_files
        path_dataset = hp.path_data_train
        dataset_split_entry = get_split_idx(hp.dset_preset, full_train=hp.full_train if phase == 'train' else False)[phase]

        self.List_patient = sorted([f'{path_dataset}/{name_dset}/{patient}' for name_dset, patient in dataset_split_entry])
        if n!=0: self.List_patient = self.List_patient[:n]
        print(f'Num Patient ({phase}): {len(self.List_patient):3.0f}')

        self.idx_var = hp.skeleton_sampling_jitter
        self.s = hp.size_patch//2
        self.p = hp.random_sampling_ratio
        self.list_h5_path = []
        for path in self.List_patient:
            data_path = {'CT':f'{path}/CT.h5', 'MK':f'{path}/MK.h5', 'SK':f'{path}/SK.h5', 'MK_parsing':f'{path}/MK_parsing.h5'}
            if hp.use_LM: data_path['LM'] = f'{path}/LM.h5'
            if hp.use_CV: data_path['CV'] = f'{path}/CV.h5'
            if hp.use_MW: data_path['MW'] = f'{path}/MW.h5'
            if hp.use_LW: data_path['LW'] = f'{path}/LW.h5'
            self.list_h5_path.append(data_path)

        self.list_data_Idx_SK = [torch.load(f'{path}/Idx_SK.pt', map_location='cpu', weights_only=False) for path in self.List_patient]
        self.list_data_MK_pos = [torch.load(f'{path}/MK_pos.pt', map_location='cpu', weights_only=False) for path in self.List_patient]
        self.list_data_LM_pos = [torch.load(f'{path}/LM_pos.pt', map_location='cpu', weights_only=False) for path in self.List_patient] if hp.use_PE else None
        self.list_shape_data = [self._get_shape(path['CT'], 'CT') for path in self.list_h5_path]

        if hp.use_PE: self.pos_encoding = Positional_Encoding(size=hp.size_patch)
        self.n = len(self.list_h5_path)
        self.N = len(self.list_h5_path)*hp.alpha
        self.m = hp.m
        self.M = hp.M
        self.cache = None

    def __len__(self):
        return self.N

    def _get_shape(self, path, key):
        with h5py.File(path, 'r') as f:
            shape = f[key].shape

        return shape

    def _init_cache(self):
        if self.cache is None: self.cache = OrderedDict()

    def _get_file(self, idx):
        self._init_cache()
        if idx in self.cache:
            self.cache.move_to_end(idx)

            return self.cache[idx]

        data = {key:h5py.File(path, 'r', libver='latest', swmr=True) for key, path in self.list_h5_path[idx].items()}
        self.cache[idx] = data
        if len(self.cache) > self.max_open_files:
            _, old_data = self.cache.popitem(last=False)
            for f in old_data.values(): f.close()

        return data

    def __getitem__(self, idx):
        idx = idx % self.n
        data = self._get_file(idx)
        x_center, y_center, z_center = sampling_idx(self.list_shape_data[idx], self.list_data_Idx_SK[idx], self.s, p=self.p, var=self.idx_var)
        xs, xe = x_center-self.s, x_center+self.s
        ys, ye = y_center-self.s, y_center+self.s
        zs, ze = z_center-self.s, z_center+self.s
        crop = np.s_[xs:xe, ys:ye, zs:ze]

        X_data = np.zeros([self.hp.dim_in, self.hp.size_patch, self.hp.size_patch, self.hp.size_patch], dtype=np.float32)
        y_data = np.zeros([self.hp.dim_true, self.hp.size_patch, self.hp.size_patch, self.hp.size_patch], dtype=np.float32)
        ct = data['CT']['CT'][crop]
        ct = np.clip(ct, self.m, self.M)
        X_data[0] = minMax_scaling(ct, m=self.m, M=self.M)
        y_data[0] = data['MK']['MK'][crop]

        l = 1
        if self.hp.use_LM:
            X_data[l] = np.float32(data['LM']['LM'][crop]>0)
            l += 1
        if self.hp.use_CV:
            X_data[l] = data['CV']['CV'][crop]
            l += 1
        if self.hp.use_PE:
            X_data[l:l+6] = self.pos_encoding.get_PE(self.list_data_LM_pos[idx], [x_center, y_center, z_center])

        y_data[1] = data['SK']['SK'][crop]
        y_data[2] = data['MK_parsing']['MK_parsing'][crop]
        l = 3
        if self.hp.use_MW:
            y_data[l] = data['MW']['MW'][crop]
            l += 1
        if self.hp.use_LW:
            y_data[l] = data['LW']['LW'][crop] + 1
            l += 1

        X_data = torch.from_numpy(X_data)
        y_data = torch.from_numpy(y_data)

        return X_data, y_data

    def __del__(self):
        if self.cache is not None:
            for data in self.cache.values():
                for f in data.values(): f.close()
