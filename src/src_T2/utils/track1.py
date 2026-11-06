import os

import h5py
import numpy as np
import torch

from common.utils.checkpoint_compat import load_options


def load_track1_model(path_checkpoint, device):
    from src_T1.utils.utils_model import load_model

    path_res = os.path.dirname(os.path.dirname(path_checkpoint))
    hp = load_options(f'{path_res}/options.pt')
    hp.path_model = os.path.dirname(path_checkpoint)
    model = load_model(hp, epoch_load=0)
    pack = torch.load(path_checkpoint, map_location='cpu', weights_only=False)
    state = pack['model'] if isinstance(pack, dict) and 'model' in pack else pack
    if any(key.startswith('module.') for key in state): state = {key.replace('module.', '', 1):value for key, value in state.items()}
    model.load_state_dict(state)

    return hp, model.eval().to(device)


def predict_track1_volume(hp, model, name_dset, patient, path_data, device, batch_size=4, num_workers=4):
    from src_T1.utils.utils_inference import Dataset_AW_inference, _sliding_window_inference

    hp.path_data = path_data
    hp.overlap_inference = 32
    test_set = Dataset_AW_inference(hp, patient, name_dset=name_dset)

    return _sliding_window_inference(test_set, model, hp, batch_size, num_workers, device)


def load_track1_cache(path_cache, name_dset, patient):
    path = f'{path_cache}/{name_dset}/{patient}'
    with h5py.File(f'{path}/probability.h5', 'r') as f: probability = np.asarray(f['probability'], dtype=np.float32)
    with h5py.File(f'{path}/mask.h5', 'r') as f: mask = np.asarray(f['mask'], dtype=np.uint8)

    return probability, mask


def save_track1_cache(path_cache, name_dset, patient, probability, mask):
    path = f'{path_cache}/{name_dset}/{patient}'
    os.makedirs(path, exist_ok=True)
    with h5py.File(f'{path}/probability.h5', 'w') as f: f.create_dataset('probability', data=np.asarray(probability, dtype=np.float16))
    with h5py.File(f'{path}/mask.h5', 'w') as f: f.create_dataset('mask', data=np.asarray(mask, dtype=np.uint8))
