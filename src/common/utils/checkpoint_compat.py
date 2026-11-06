import importlib
import sys

import torch


def register_legacy_modules():
    aliases = {
        'options':'src_T1.options',
        'options.hyper_parameters':'src_T1.options.hyper_parameters'}
    for old_name, new_name in aliases.items():
        if old_name not in sys.modules: sys.modules[old_name] = importlib.import_module(new_name)


def load_options(path):
    register_legacy_modules()
    hp = torch.load(path, map_location='cpu', weights_only=False)

    if not hasattr(hp, 'overlap_inference'): hp.overlap_inference = 32
    if not hasattr(hp, 'full_train'): hp.full_train = False
    if hasattr(hp, 'refresh_paths'): hp.refresh_paths(create_dirs=False)

    return hp
