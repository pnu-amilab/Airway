import os

import torch

from common.utils.checkpoint_compat import load_options
from src_T2.options.hyper_parameters import HP_T2, load_local_HP_T2


def load_kernel_info_T2(epoch, kernel):
    hp_base = HP_T2()
    load_local_HP_T2(hp_base)
    path_res = f'{hp_base.path_Res}/{kernel}'
    hp = load_options(f'{path_res}/options.pt')
    hp.path_res = path_res
    hp.path_model = f'{path_res}/model'
    hp.path_inference = f'{path_res}/inference'
    hp.path_evaluation = f'{path_res}/evaluation'
    hp.path_fig = f'{path_res}/fig'

    history_path = f'{hp.path_model}/history.pt'
    history = torch.load(history_path, map_location='cpu', weights_only=False) if os.path.exists(history_path) else []
    epoch_load = epoch
    epoch_tag = epoch
    if epoch == -1:
        epoch_load = -1
        epoch_tag = hp.epochs
    elif epoch == -2:
        epoch_load = -2
        epoch_tag = history[-1].get('epoch_best', 'best') if len(history) > 0 else 'best'

    return hp, history, epoch_load, epoch_tag
