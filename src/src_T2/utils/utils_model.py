import os

import torch


def _load_state(path):
    pack = torch.load(path, map_location='cpu', weights_only=False)
    state = pack['model'] if isinstance(pack, dict) and 'model' in pack else pack
    if any(key.startswith('module.') for key in state): state = {key.replace('module.', '', 1):value for key, value in state.items()}

    return state


def _get_checkpoint_path(path_model, epoch_load):
    if epoch_load > 0: return f'{path_model}/model_{epoch_load}.pt'
    if epoch_load == -1: return f'{path_model}/last_point.pt'
    if epoch_load == -2: return f'{path_model}/model_best.pt'
    raise ValueError(f'Invalid epoch_load: {epoch_load}')


def _adapt_input_weight(source, target):
    if source.ndim != 5 or target.ndim != 5 or source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]: return None
    if target.shape[1] == 3 and source.shape[1] in [1, 9]:
        weight = torch.zeros_like(target)
        weight[:, 0] = source[:, 0]
        return weight
    if source.shape[1] != 9 or target.shape[1] != 15: return None
    weight = torch.zeros_like(target)
    weight[:, 0] = source[:, 0]
    weight[:, 8] = source[:, 2]
    weight[:, 9:15] = source[:, 3:9]

    return weight


def load_init_T2(model, hp):
    if hp.path_init_model == '': return model
    if not os.path.exists(hp.path_init_model): raise FileNotFoundError(hp.path_init_model)
    state = _load_state(hp.path_init_model)
    model_state = model.state_dict()
    load_state = {key:value for key, value in state.items() if key in model_state and model_state[key].shape == value.shape}
    key = 'e_conv_1.encoder.0.weight'
    if key in state and key in model_state:
        adapted = _adapt_input_weight(state[key], model_state[key])
        if adapted is not None: load_state[key] = adapted
    model.load_state_dict(load_state, strict=False)
    print(f'Load T1 ASTRA initialization: {hp.path_init_model}')

    return model


def load_model_T2(hp, epoch_load=None, load_init=False):
    from src_T2.networks.network_ASTRA_Net_conditional import ASTRA_Net_Conditional_T2
    model = ASTRA_Net_Conditional_T2(dim_in=hp.dim_in, dim_out=hp.num_classes-1, dim_base=hp.dim_base, class_dim=hp.class_dim,
                                    ega=hp.ega, ega_learnable=hp.ega_learnable, ega_recalibration=hp.ega_recalibration,
                                    lobe_bias_strength=hp.lobe_bias_strength)
    if epoch_load is None: epoch_load = hp.epoch_load
    if epoch_load != 0:
        path = _get_checkpoint_path(hp.path_model, epoch_load)
        state = _load_state(path)
        model.load_state_dict(state)
    elif load_init:
        model = load_init_T2(model, hp)

    return model
