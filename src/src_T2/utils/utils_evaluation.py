from pathlib import Path

import h5py
import numpy as np
import torch

from common.dataset.split_idx import get_split_idx_T2
from common.utils.utils_common import load_nii, make_dir
from src_T2.utils.graph_metrics import graph_scores
from src_T2.utils.utils_io import load_kernel_info_T2
from src_T2.utils.utils_plot import plot_tree_T2
from src_T2.utils.voxel_metrics import multi_class_overlap, multi_class_cldice


def evaluation_T2(epoch, kernel, n=0, plot=True):
    hp, _, _, epoch_tag = load_kernel_info_T2(epoch, kernel)
    list_entry = get_split_idx_T2(hp.dset_preset, use_extra_train=hp.use_extra_train_T2)['test']
    if n > 0: list_entry = list_entry[:n]
    path_eval = f'{hp.path_evaluation}/epoch_{epoch_tag}'
    make_dir(path_eval)
    print(f'[{str(epoch_tag):>4} epoch]: {kernel} --evaluation', flush=True)
    len_title = max(38, *(len(f'[{i+1:3}/{len(list_entry):3}] {dset}/{patient}')+2 for i, (dset, patient) in enumerate(list_entry)))
    list_metrics = ['mDSC', 'mclDice', 'ACC', 'F1', 'SC', 'TD', 'TAcc']
    print(f'{"Patient":<{len_title}}'+''.join(f'{key:>9}' for key in list_metrics), flush=True)

    for patient_idx, (name_dset, patient) in enumerate(list_entry):
        path_patient = f'{hp.path_data}/{name_dset}/{patient}'
        _, prediction = load_nii(f'{hp.path_inference}/epoch_{epoch_tag}/{name_dset}/{patient}_label.nii.gz')
        with h5py.File(f'{path_patient}/CK.h5', 'r') as f: target = f['CK'][:]
        prediction = prediction.astype('uint8')
        target = target.astype('uint8')
        with h5py.File(f'{path_patient}/MK_parsing.h5', 'r') as f: parsing = f['MK_parsing'][:]
        with h5py.File(f'{path_patient}/SK.h5', 'r') as f: skeleton = f['SK'][:]
        path_graph = next((p for p in [Path(path_patient)/'graph.npy', Path(path_patient)/'airway_graph.npy'] if p.exists()), None)
        edges = np.load(path_graph, allow_pickle=False) if path_graph is not None else None
        means, per_class = multi_class_overlap(prediction, target)
        mclDice, clDice_class = multi_class_cldice(prediction, target, skeleton)
        graph = graph_scores(prediction, target, parsing, edges=edges)
        res = {'mDSC':means['DSC_class'], 'mclDice':mclDice, 'mIoU':means['IoU_class'],
               'mPRE':means['PRE_class'], 'mSEN':means['SEN_class'], 'mDice':means['DSC_class'],
               **per_class, 'clDice_class':clDice_class, **graph}
        make_dir(f'{path_eval}/{name_dset}')
        torch.save(res, f'{path_eval}/{name_dset}/{patient}.pt')
        row = f'[{patient_idx+1:3d}/{len(list_entry):3d}] {name_dset}/{patient}'
        print(f'{row:<{len_title}}'+''.join(f'{res[key]:9.4f}' for key in list_metrics)+f'  [{res["graph_source"]}]', flush=True)

    if plot: plot_tree_T2(epoch, kernel, n=n, show=False)
