from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from common.dataset.split_idx import get_split_idx_T2
from common.utils.utils_common import load_nii
from src_T2.utils.utils_io import load_kernel_info_T2


def plot_tree_T2(epoch, kernel, n=0, show=False):
    hp, _, _, epoch_tag = load_kernel_info_T2(epoch, kernel)
    list_entry = get_split_idx_T2(hp.dset_preset, use_extra_train=hp.use_extra_train_T2)['test']
    if n > 0: list_entry = list_entry[:n]
    path_fig = Path(hp.path_fig)/f'epoch_{epoch_tag}'

    for i, (name_dset, patient) in enumerate(list_entry):
        path_pred = Path(hp.path_inference)/f'epoch_{epoch_tag}'/name_dset/f'{patient}_label.nii.gz'
        _, prediction = load_nii(path_pred)
        with h5py.File(f'{hp.path_data}/{name_dset}/{patient}/CK.h5', 'r') as f: target = f['CK'][:]
        prediction = np.asarray(prediction, dtype=np.uint8)
        target = np.asarray(target, dtype=np.uint8)
        coords = np.where((prediction > 0) | (target > 0))
        if len(coords[0]) == 0: continue
        bounds = [(max(0, int(v.min())-5), min(prediction.shape[axis], int(v.max())+6)) for axis, v in enumerate(coords)]
        crop = tuple(slice(a, b) for a, b in bounds)
        p, y = prediction[crop], target[crop]

        fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
        for row, (plane, axis) in enumerate([('Axial', 0), ('Coronal', 1), ('Sagittal', 2)]):
            p_proj = np.any(p > 0, axis=axis)
            y_proj = np.any(y > 0, axis=axis)
            correct = np.any((p > 0) & (p == y), axis=axis)
            wrong_label = np.any((p > 0) & (y > 0) & (p != y), axis=axis)
            fp = np.any((p > 0) & (y == 0), axis=axis)
            fn = np.any((p == 0) & (y > 0), axis=axis)
            diff = np.zeros((*p_proj.shape, 3), dtype=np.float32)
            diff[correct] = [0.48, 0.78, 0.83]
            diff[wrong_label] = [0.68, 0.48, 0.92]
            diff[fp] = [1.00, 0.70, 0.23]
            diff[fn] = [0.94, 0.30, 0.33]

            for col, img in enumerate((y_proj, p_proj, diff)):
                ax = axes[row, col]
                ax.set_facecolor('#111827')
                if col < 2: ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                else: ax.imshow(diff, interpolation='nearest')
                ax.set_title(f'{plane}  |  '+('GT' if col == 0 else 'Prediction' if col == 1 else 'Difference'), fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)

        fig.suptitle(f'{kernel}  |  epoch {epoch_tag}  |  {name_dset}/{patient}', fontsize=13, fontweight='semibold')
        fig.legend(handles=[Patch(color='#7ac7d4', label='Correct label'), Patch(color='#ad7aeb', label='Wrong label'), Patch(color='#ffb33b', label='FP'), Patch(color='#f04d54', label='FN')], loc='lower center', ncol=4, frameon=False)
        out_dir = path_fig/name_dset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir/f'{patient}.png'
        fig.savefig(out_path, dpi=170, bbox_inches='tight', facecolor='white')
        if show: plt.show()
        plt.close(fig)
        print(f'[{i+1}/{len(list_entry)}] {out_path}')
