from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from common.utils.utils_common import get_result_tag, load_nii
from src_T1.utils.utils_history import _epoch_tag, _path_res


def plot_tree(epoch, kernel, alpha=0.20, target_dset=None, n=0, show=True):
    epoch = _epoch_tag(epoch, kernel)
    tag = get_result_tag(epoch, alpha)
    path_eval = _path_res(kernel)/'evaluation'/tag
    path_fig = _path_res(kernel)/'fig'/tag
    list_file = sorted(path_eval.glob('*/*_p_data.nii.gz'))
    if target_dset is not None:
        if isinstance(target_dset, str): target_dset = [target_dset]
        list_file = [path for path in list_file if path.parent.name in target_dset]
    if n > 0: list_file = list_file[:n]

    for i, path_p in enumerate(list_file):
        patient = path_p.name[:-len('_p_data.nii.gz')]
        path_y = path_p.with_name(f'{patient}_y_data.nii.gz')
        path_res = path_p.with_name(f'{patient}.pt')
        _, p_data = load_nii(path_p)
        _, y_data = load_nii(path_y)
        p_data = p_data > 0
        y_data = y_data > 0
        coords = np.where(p_data | y_data)
        if len(coords[0]) == 0: continue
        bounds = [(max(0, int(v.min())-5), min(p_data.shape[axis], int(v.max())+6)) for axis, v in enumerate(coords)]
        crop = tuple(slice(a, b) for a, b in bounds)
        p_data, y_data = p_data[crop], y_data[crop]

        res = torch.load(path_res, map_location='cpu', weights_only=False)
        fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
        planes = [('Axial', 0), ('Coronal', 1), ('Sagittal', 2)]
        for row, (plane, axis) in enumerate(planes):
            p = np.any(p_data, axis=axis)
            y = np.any(y_data, axis=axis)
            diff = np.zeros((*p.shape, 3), dtype=np.float32)
            diff[y & p] = [0.48, 0.78, 0.83]
            diff[y & ~p] = [0.94, 0.30, 0.33]
            diff[p & ~y] = [1.00, 0.70, 0.23]
            for col, img in enumerate((y, p, diff)):
                ax = axes[row, col]
                ax.set_facecolor('#111827')
                if col < 2:
                    ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                else:
                    ax.imshow(diff, interpolation='nearest')
                ax.set_title(f'{plane}  |  '+('GT' if col == 0 else 'Prediction' if col == 1 else 'Difference'), fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)

        scores = '    '.join(f'{key} {float(res[key]):.2f}' for key in ['DSC', 'PRE', 'TDR', 'BDR'] if key in res)
        fig.suptitle(f'{kernel}  |  epoch {epoch}  |  {path_p.parent.name}/{patient}\n{scores}', fontsize=13, fontweight='semibold')
        fig.legend(handles=[Patch(color='#7ac7d4', label='TP'), Patch(color='#f04d54', label='FN'), Patch(color='#ffb33b', label='FP')], loc='lower center', ncol=3, frameon=False)
        out_dir = path_fig/path_p.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir/f'{patient}.png'
        fig.savefig(out_path, dpi=170, bbox_inches='tight', facecolor='white')
        if show: plt.show()
        plt.close(fig)
        print(f'[{i+1}/{len(list_file)}] {out_path}')
