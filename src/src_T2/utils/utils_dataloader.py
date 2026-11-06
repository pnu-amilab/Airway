import os
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from common.dataset.split_idx import get_split_idx_T2
from common.dataset.utils_dataloader import Positional_Encoding, sampling_idx
from common.utils.utils_common import minMax_scaling


def one_hot_lobe_mask(lm, num_lobes=5):
    return np.stack([lm == label for label in range(1, num_lobes+1)], axis=0).astype(np.float32)


class Dataset_AW_T2(Dataset):

    def __init__(self, hp, phase='train', n=0, max_open_files=16):
        self.hp = hp
        self.phase = phase
        self.max_open_files = max_open_files
        self.list_entry = get_split_idx_T2(hp.dset_preset, use_extra_train=hp.use_extra_train_T2, full_train=hp.full_train if phase == 'train' else False)[phase]
        if n != 0: self.list_entry = self.list_entry[:n]
        self.List_patient = [f'{hp.path_data_train}/{name_dset}/{patient}' for name_dset, patient in self.list_entry]
        print(f'Num Patient T2 ({phase}): {len(self.List_patient):3d}')

        self.idx_var = 20
        self.s = hp.size_patch//2
        self.list_h5_path = []
        for (name_dset, patient), path in zip(self.list_entry, self.List_patient):
            path_cache = f'{hp.path_cache}/{name_dset}/{patient}'
            self.list_h5_path.append({
                'CT':f'{path}/CT.h5', 'MK':f'{path}/MK.h5', 'SK':f'{path}/SK.h5', 'MK_parsing':f'{path}/MK_parsing.h5',
                'CK':f'{path}/CK.h5', 'MW':f'{path}/MW.h5',
                'T1P':f'{path_cache}/probability.h5', 'T1M':f'{path_cache}/mask.h5',
                **({'LM':f'{path}/LM.h5'} if hp.use_LM else {}),
                **({'CV':f'{path}/CV.h5'} if hp.use_CV else {})})

        self.list_data_Idx_SK = [torch.load(f'{path}/Idx_SK.pt', map_location='cpu', weights_only=False) for path in self.List_patient]
        self.list_data_LM_pos = [torch.load(f'{path}/LM_pos.pt', map_location='cpu', weights_only=False) for path in self.List_patient] if hp.use_PE else None
        self.list_shape_data = [self._get_shape(path['CT'], 'CT') for path in self.list_h5_path]
        self.list_data_Idx_CK_SK, class_count = self._load_class_sampling_info()
        class_weight_all = self._get_class_weight(class_count)
        self.class_weight = class_weight_all[1:]
        if phase == 'train':
            count_text = ', '.join([f'{idx}:{int(value)}' for idx, value in enumerate(class_count) if idx > 0])
            weight_text = ', '.join([f'{idx+1}:{self.class_weight[idx]:.3f}' for idx in range(len(self.class_weight))])
            print(f'Class voxel count T2: {count_text}')
            print(f'Class CCE weight T2: {weight_text}')

        self.pos_encoding = Positional_Encoding(size=hp.size_patch)
        self.n = len(self.list_h5_path)
        self.N = int(self.n*hp.alpha)
        self.m = hp.m
        self.M = hp.M
        self.cache = None

    def __len__(self):
        return self.N

    def _get_shape(self, path, key):
        with h5py.File(path, 'r') as f: return f[key].shape

    def _load_class_sampling_info(self):
        list_data_Idx_CK_SK = []
        class_count = np.zeros(self.hp.num_classes, dtype=np.int64)
        for patient_idx, path in enumerate(self.List_patient):
            path_idx = f'{path}/Idx_CK_SK_T1.pt'
            path_count = f'{path}/CK_count_T1.pt'
            if os.path.exists(path_idx) and os.path.exists(path_count):
                data_Idx_CK_SK = torch.load(path_idx, map_location='cpu', weights_only=False)
                count = np.asarray(torch.load(path_count, map_location='cpu', weights_only=False), dtype=np.int64)
            else:
                with h5py.File(self.list_h5_path[patient_idx]['CK'], 'r') as f: CK = np.asarray(f['CK'], dtype=np.uint8)
                with h5py.File(self.list_h5_path[patient_idx]['T1M'], 'r') as f: T1M = np.asarray(f['mask'], dtype=np.uint8)>0
                count = np.bincount(CK[T1M].reshape(-1), minlength=self.hp.num_classes).astype(np.int64)
                idx_sk = self.list_data_Idx_SK[patient_idx]
                idx_x = np.asarray(idx_sk[0], dtype=np.int64)
                idx_y = np.asarray(idx_sk[1], dtype=np.int64)
                idx_z = np.asarray(idx_sk[2], dtype=np.int64)
                labels_sk = CK[idx_x, idx_y, idx_z]
                inside = T1M[idx_x, idx_y, idx_z]
                data_Idx_CK_SK = {}
                for class_idx in range(1, self.hp.num_classes):
                    mask = (labels_sk == class_idx) & inside
                    if np.any(mask): data_Idx_CK_SK[class_idx] = (torch.from_numpy(idx_x[mask]), torch.from_numpy(idx_y[mask]), torch.from_numpy(idx_z[mask]))
                torch.save(data_Idx_CK_SK, path_idx)
                torch.save(torch.from_numpy(count), path_count)
            list_data_Idx_CK_SK.append(data_Idx_CK_SK)
            class_count += count[:self.hp.num_classes]

        return list_data_Idx_CK_SK, class_count

    def _get_class_weight(self, class_count):
        weight = np.ones(self.hp.num_classes, dtype=np.float32)
        foreground_count = class_count[1:].astype(np.float64)
        present = foreground_count > 0
        median = np.median(foreground_count[present])
        foreground_weight = np.ones_like(foreground_count)
        foreground_weight[present] = np.power(median/foreground_count[present], self.hp.class_weight_power)
        foreground_weight[present] /= np.mean(foreground_weight[present])
        foreground_weight = np.clip(foreground_weight, 1/self.hp.class_weight_max, self.hp.class_weight_max)
        weight[1:] = foreground_weight.astype(np.float32)

        return torch.from_numpy(weight)

    def _get_file(self, idx):
        if self.cache is None: self.cache = OrderedDict()
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
        patient_idx = idx%self.n
        data = self._get_file(patient_idx)
        x_center, y_center, z_center = sampling_idx(self.list_shape_data[patient_idx], self.list_data_Idx_SK[patient_idx], self.s, p=self.hp.random_sampling_ratio, var=self.hp.skeleton_sampling_jitter)
        xs, xe = x_center-self.s, x_center+self.s
        ys, ye = y_center-self.s, y_center+self.s
        zs, ze = z_center-self.s, z_center+self.s
        crop = np.s_[xs:xe, ys:ye, zs:ze]

        X_data = np.zeros([self.hp.dim_in, self.hp.size_patch, self.hp.size_patch, self.hp.size_patch], dtype=np.float32)
        y_data = np.zeros([self.hp.dim_true, self.hp.size_patch, self.hp.size_patch, self.hp.size_patch], dtype=np.float32)
        ct = np.clip(data['CT']['CT'][crop], self.m, self.M)
        X_data[0] = minMax_scaling(ct, m=self.m, M=self.M)
        X_data[1] = np.asarray(data['T1P']['probability'][crop], dtype=np.float32)
        X_data[2] = np.asarray(data['T1M']['mask'][crop], dtype=np.float32)
        if self.hp.use_LM:
            LM_data = np.asarray(data['LM']['LM'][crop], dtype=np.uint8)
            X_data[3:3+self.hp.num_lobes] = one_hot_lobe_mask(LM_data, self.hp.num_lobes)
        if self.hp.use_CV: X_data[8] = data['CV']['CV'][crop]
        if self.hp.use_PE: X_data[9:15] = self.pos_encoding.get_PE(self.list_data_LM_pos[patient_idx], [x_center, y_center, z_center])

        CK_data = np.asarray(data['CK']['CK'][crop], dtype=np.int64)
        y_data[0] = np.asarray(data['MK']['MK'][crop], dtype=np.float32)
        y_data[1] = np.asarray(data['SK']['SK'][crop], dtype=np.float32)
        y_data[2] = np.asarray(data['MK_parsing']['MK_parsing'][crop], dtype=np.float32)
        y_data[3] = np.asarray(data['MW']['MW'][crop], dtype=np.float32)

        return torch.from_numpy(X_data), torch.from_numpy(y_data), torch.from_numpy(CK_data)

    def __del__(self):
        cache = getattr(self, 'cache', None)
        if cache is not None:
            for data in cache.values():
                for f in data.values():
                    if f.id.valid: f.close()
