import h5py
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

from src_T1.utils.metrics import *
from src_T1.utils.utils_plot import plot_tree
from common.dataset.split_idx import get_dset_preset, get_split_idx
from common.utils.checkpoint_compat import load_options
from src_T1.options.hyper_parameters import HP, load_local_HP
from common.utils.utils_common import get_result_tag, load_nii, make_dir, save_nii
from common.utils.utils_postprocessing import postprocess_threshold


def evaluation(epoch, kernel, alpha=0.20, n=0, dset_test=None, plot=True):
    hp_base = HP()
    load_local_HP(hp_base)
    hp = load_options(f'{hp_base.path_Res}/{kernel}/options.pt')
    type_eval = get_result_tag(epoch, alpha)
    print(f'[{str(epoch):>4} epoch / alpha {alpha:.2f}]: {hp.name} --evaluation', flush=True)
    make_dir(f'{hp.path_evaluation}/{type_eval}')
    dset_preset = get_dset_preset(hp, dset_test=dset_test)
    list_entry = get_split_idx(dset_preset)['test']
    if n > 0: list_entry = list_entry[:n]

    len_title = max(45, *(len(f'[{i+1:3}/{len(list_entry):3}]: {dset}/{patient:<13}')+2 for i, (dset, patient) in enumerate(list_entry)))
    list_metrics = ['DSC', 'PRE', 'TDR', 'BDR', 'SEN', 'MOS', 'WOS']
    print(f'{"Patient":<{len_title}}' + ''.join(f'{key:>9}' for key in list_metrics), flush=True)
    for patient_idx, (name_dset, patient) in enumerate(list_entry):
        row_str = f'[{patient_idx+1:3.0f}/{len(list_entry):3.0f}]: {name_dset}/{patient:<13}'
        print(f'{row_str:<{len_title}}', end='', flush=True)
        path_patient = f'{hp.path_data}/{name_dset}/{patient}'
        _, probability = load_nii(f'{hp.path_inference}/epoch_{epoch}/{name_dset}/{patient}_p.nii.gz')
        p_data = (probability > alpha).astype(np.uint8)
        _, t_data = load_nii(f'{path_patient}/MK_parsing.nii.gz')
        if name_dset in ['ATM22_Enhanced', 'ATM26_Enhanced']:
            with h5py.File(f'{path_patient}/SK.h5', 'r') as data: s_data = data['SK'][:]
        else:
            _, s_data = load_nii(f'{path_patient}/sk.nii.gz')
        with h5py.File(f'{path_patient}/MK.h5', 'r') as data: y_data = data['MK'][:]
        t_data = t_data.astype('int16')
        s_data = s_data.astype('int16')
        y_data = y_data.astype('int16')
        MK_pos = torch.load(f'{path_patient}/MK_pos.pt', map_location='cpu', weights_only=False)
        z0 = int(MK_pos['idx_xm'])
        p_data[:z0] = 0
        t_data[:z0] = 0
        s_data[:z0] = 0
        y_data[:z0] = 0
        p_data = postprocess_threshold(p_data, 0.5)
        labels, n_labels = ndimage.label(p_data, structure=ndimage.generate_binary_structure(3, 1))
        if n_labels:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            p_data = (labels == sizes.argmax()).astype(np.uint8)
        else:
            p_data = np.zeros_like(p_data, dtype=np.uint8)

        DSC = dice_coefficient_score_calculation(p_data, y_data)
        TDR = tree_length_calculation(p_data, s_data)
        _, _, BDR = branch_detected_calculation(p_data, t_data, s_data)
        PRE = precision_calculation(p_data, y_data)
        MOS = round((DSC+PRE+TDR+BDR)/4, 2)
        WOS = wos_score_calculation(DSC, PRE, TDR, BDR)
        SEN = sensitivity_calculation(p_data, y_data)
        SPE = specificity_calculation(p_data, y_data)
        res = {'DSC':DSC, 'PRE':PRE, 'TDR':TDR, 'BDR':BDR, 'MOS':MOS, 'WOS':WOS, 'SEN':SEN, 'SPE':SPE}

        meta = torch.load(f'{path_patient}/meta_info.pt', map_location='cpu', weights_only=False)
        make_dir(f'{hp.path_evaluation}/{type_eval}/{name_dset}')
        save_nii(f'{hp.path_evaluation}/{type_eval}/{name_dset}/{patient}_p_data.nii.gz', p_data, meta['pixdim'], meta['affine_tform'])
        save_nii(f'{hp.path_evaluation}/{type_eval}/{name_dset}/{patient}_y_data.nii.gz', y_data, meta['pixdim'], meta['affine_tform'])
        torch.save(res, f'{hp.path_evaluation}/{type_eval}/{name_dset}/{patient}.pt')
        print('\r'+f'{row_str:<{len_title}}'+''.join(f'{res[key]:9.2f}' for key in list_metrics), flush=True)

    if plot: plot_tree(epoch, hp.name, alpha=alpha, target_dset=dset_test, n=n, show=False)


def evaluation_audit(epoch, kernel, alpha=0.20, n=0, dset_test=None):
    hp_base = HP()
    load_local_HP(hp_base)
    hp = load_options(f'{hp_base.path_Res}/{kernel}/options.pt')
    dset_preset = get_dset_preset(hp, dset_test=dset_test)
    list_entry = get_split_idx(dset_preset)['test']
    if n > 0: list_entry = list_entry[:n]
    if not list_entry: raise ValueError(f'No test cases found for {dset_test}')

    configurations = [
        ('current', False, False, 26),
        ('SK_only', True, False, 26),
        ('SK_upper', True, True, 26),
        ('SK_LCC6', True, False, 6),
        ('legacy', True, True, 6),
    ]
    rows = []
    for i, (name_dset, patient) in enumerate(list_entry):
        print(f'[{i+1}/{len(list_entry)}] {name_dset}/{patient}', flush=True)
        path_patient = f'{hp.path_data}/{name_dset}/{patient}'
        _, probability = load_nii(f'{hp.path_inference}/epoch_{epoch}/{name_dset}/{patient}_p.nii.gz')
        _, t_data = load_nii(f'{path_patient}/MK_parsing.nii.gz')
        _, sk_data = load_nii(f'{path_patient}/sk.nii.gz')
        with h5py.File(f'{path_patient}/MK.h5', 'r') as data: y_data = data['MK'][:].astype(np.uint8)
        t_data = t_data.astype(np.int16)
        sk_data = (sk_data > 0).astype(np.uint8)
        if name_dset in ['ATM22_Enhanced', 'ATM26_Enhanced']:
            with h5py.File(f'{path_patient}/SK.h5', 'r') as data: SK_data = (data['SK'][:] > 0).astype(np.uint8)
        else:
            SK_data = sk_data
        if not (probability.shape == y_data.shape == t_data.shape == sk_data.shape == SK_data.shape):
            raise ValueError(f'Volume shapes mismatch: {name_dset}/{patient}')

        MK_pos = None
        for mode, use_SK, upper_rejection, connectivity in configurations:
            s_data = SK_data if use_SK else sk_data
            p_data = (probability > alpha).astype(np.uint8)
            if upper_rejection:
                if MK_pos is None:
                    MK_pos = torch.load(f'{path_patient}/MK_pos.pt', map_location='cpu', weights_only=False)
                z0 = int(MK_pos['idx_xm'])
                p_data[:z0] = 0
                y_eval = y_data.copy()
                s_eval = s_data.copy()
                t_eval = t_data.copy()
                y_eval[:z0] = 0
                s_eval[:z0] = 0
                t_eval[:z0] = 0
            else:
                y_eval, s_eval, t_eval = y_data, s_data, t_data

            p_data = sitk.GetArrayFromImage(sitk.BinaryFillhole(sitk.GetImageFromArray(p_data), fullyConnected=True)).astype(np.uint8)
            structure = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
            components, component_count = ndimage.label(p_data, structure=structure)
            if component_count:
                sizes = np.bincount(components.ravel())
                sizes[0] = 0
                p_data = (components == sizes.argmax()).astype(np.uint8)
            else:
                p_data = np.zeros_like(p_data)
            del components

            DSC = dice_coefficient_score_calculation(p_data, y_eval)
            PRE = precision_calculation(p_data, y_eval)
            TDR = tree_length_calculation(p_data, s_eval)
            with np.errstate(divide='ignore', invalid='ignore'):
                _, _, BDR = branch_detected_calculation(p_data, t_eval, s_eval)
            SEN = sensitivity_calculation(p_data, y_eval)
            MOS = round((DSC+PRE+TDR+BDR)/4, 2)
            WOS = wos_score_calculation(DSC, PRE, TDR, BDR)

            rows.append(dict(dataset=name_dset, patient=patient, mode=mode, DSC=DSC, PRE=PRE, TDR=TDR, BDR=BDR, SEN=SEN, MOS=MOS, WOS=WOS))

    metrics = ['DSC', 'PRE', 'TDR', 'BDR', 'SEN', 'MOS', 'WOS']
    means = {}
    for mode, _, _, _ in configurations:
        selected = [row for row in rows if row['mode'] == mode]
        means[mode] = {key: float(np.mean([row[key] for row in selected])) for key in metrics}

    print(f'\nEvaluation audit: {kernel}, epoch={epoch}, alpha={alpha}, N={len(list_entry)}')
    print(f'{"Mode":<14}{"DSC":>9}{"PRE":>9}{"TDR":>9}{"BDR":>9}{"SEN":>9}{"BDR delta":>12}')
    for mode, _, _, _ in configurations:
        values = means[mode]
        print(f'{mode:<14}{values["DSC"]:9.2f}{values["PRE"]:9.2f}{values["TDR"]:9.2f}{values["BDR"]:9.2f}{values["SEN"]:9.2f}{values["BDR"]-means["current"]["BDR"]:+12.2f}')
    print('\nPer-case BDR comparison (%)')
    print(f'{"Dataset/Patient":<37}' + ''.join(f'{mode:>12}' for mode, _, _, _ in configurations) + f'{"legacy Δ":>12}')
    for name_dset, patient in list_entry:
        case = {row['mode']: row for row in rows if row['dataset'] == name_dset and row['patient'] == patient}
        delta = case['legacy']['BDR'] - case['current']['BDR']
        print(f'{name_dset}/{patient:<{max(1, 37-len(name_dset)-1)}}' + ''.join(f'{case[mode]["BDR"]:12.2f}' for mode, _, _, _ in configurations) + f'{delta:+12.2f}')
