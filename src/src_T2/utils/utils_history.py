from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.dataset.split_idx import SPLIT_IDX_T2
from src_T2.options.hyper_parameters import HP_T2


def _path_res(kernel):
    hp = HP_T2()
    hp.refresh_paths(create_dirs=False)

    return Path(hp.path_Res)/kernel


def _load_history(kernel):
    return torch.load(_path_res(kernel)/'model/history.pt', map_location='cpu', weights_only=False)


def _epoch_tag(epoch, kernel):
    if epoch >= 0: return epoch
    history = _load_history(kernel)
    if epoch == -1: return history[-1]['epoch']+1

    return history[-1]['epoch_best']


def plot_history(kernel, m=None, n=None, plot_loss=True, plot_rate=True):
    history = _load_history(kernel)[m:n]
    epochs = np.array([entry['epoch']+1 for entry in history])
    groups = [group for group, enabled in [('loss', plot_loss), ('rate', plot_rate)] if enabled]
    keys = {group:list(dict.fromkeys(key for entry in history for phase in ['Train', 'Valid'] for key in entry[phase][group])) for group in groups}
    ncols = max(len(keys[group]) for group in groups)
    fig, axes = plt.subplots(len(groups), ncols, figsize=(4*ncols, 3.3*len(groups)), squeeze=False)

    for i, group in enumerate(groups):
        for j, key in enumerate(keys[group]):
            ax = axes[i, j]
            for phase in ['Train', 'Valid']:
                values = np.array([float(entry[phase][group].get(key, np.nan)) for entry in history])
                ax.plot(epochs, values, linewidth=1, label=phase)
                if phase == 'Valid' and np.isfinite(values).any():
                    idx = np.nanargmin(values) if group == 'loss' else np.nanargmax(values)
                    ax.axvline(epochs[idx], linestyle='--', linewidth=0.8, alpha=0.6)
                    ax.set_title(f'{key}: {values[idx]:.4f} (e{epochs[idx]})')
            ax.set_xlabel('Epoch')
            ax.grid(alpha=0.3)
            if j == 0: ax.legend()
        for j in range(len(keys[group]), ncols): axes[i, j].axis('off')

    fig.suptitle(kernel)
    plt.tight_layout()
    plt.show()


def _load_evaluation(epoch, kernel):
    if epoch < 0 and not (_path_res(kernel)/'model/history.pt').exists(): return epoch, []
    epoch = _epoch_tag(epoch, kernel)
    path_eval = _path_res(kernel)/'evaluation'/f'epoch_{epoch}'
    rows = [(f'{path.parent.name}/{path.stem}', torch.load(path, map_location='cpu', weights_only=False)) for path in sorted(path_eval.glob('*/*.pt'))]

    return epoch, rows


def _metric_text(values, std=False):
    if len(values) == 0: return '-'
    mean = np.mean(values)
    if std: return f'{mean:.2f}±{np.std(values):.2f}'

    return f'{mean:.4f}'


def _load_preset(kernel):
    path = _path_res(kernel)/'options.pt'
    if path.exists():
        from common.utils.checkpoint_compat import load_options
        hp = load_options(path)
        if hp.dset_preset: return hp.name_dset, hp.dset_preset

    name_dset = kernel.split('_', 1)[0]
    presets = HP_T2().list_dset_preset

    return name_dset, presets.get(name_dset)


def _print_preset(kernels):
    presets = {}
    for kernel in kernels:
        if kernel == '--': continue
        name, preset = _load_preset(kernel)
        if preset is None:
            print(f'Preset {name}: unavailable')
            continue
        key = (name, tuple(preset['train']), tuple(preset['test']))
        presets[key] = preset

    for (name, _, _), preset in presets.items():
        datasets = list(dict.fromkeys(preset['train']+preset['test']+list(SPLIT_IDX_T2)))
        width = max(24, *(len(dset)+2 for dset in datasets))
        print(f'Preset: {name}')
        print('='*(width+16))
        print(f'{"Dataset":<{width}}{"Train":>8}{"Test":>8}')
        print('-'*(width+16))
        for dset in datasets:
            train = 'O' if dset in preset['train'] else 'X'
            test = 'O' if dset in preset['test'] else 'X'
            print(f'{dset:<{width}}{train:>8}{test:>8}')
        print('='*(width+16))
        print()


def _filter_rows(rows, target_dset):
    if target_dset is None: return rows
    if isinstance(target_dset, str): target_dset = [target_dset]

    return [(name, res) for name, res in rows if name.split('/', 1)[0] in target_dset]


def _dataset_names(rows, target_dset=None):
    if target_dset is not None: return [target_dset] if isinstance(target_dset, str) else list(target_dset)

    return sorted(set(name.split('/', 1)[0] for name, _ in rows))


def _mean_row(label, rows, list_metrics, len_name, width, std=False):
    values = [_metric_text([float(res[key]) for _, res in rows if key in res], std=std) for key in list_metrics]

    return f'{label:<{len_name}}' + ''.join(f'{value:>{width}}' for value in values)


def print_kernel(kernel, list_metrics=None, len_name=None, target_dset=None, by_dset=False, print_preset=False, std=False):
    epoch, alpha, kernel = kernel
    if list_metrics is None: list_metrics = ['mDSC', 'mclDice', 'ACC', 'F1', 'SC', 'TD', 'TAcc']
    epoch, rows = _load_evaluation(epoch, kernel)
    rows = _filter_rows(rows, target_dset)
    if print_preset: _print_preset([kernel])
    if target_dset is not None: print(f'Target dataset: {", ".join(target_dset) if isinstance(target_dset, list) else target_dset}')
    len_name = max(0 if len_name is None else len_name, len('Patient')+2, *(len(name)+2 for name, _ in rows))
    width = 14 if std else 10
    header = f'{"Patient":<{len_name}}' + ''.join(f'{key:>{width}}' for key in list_metrics)
    line = '-'*len(header)
    print(f'{kernel} | epoch {epoch} | {len(rows)} cases')
    if by_dset:
        for dset in _dataset_names(rows, target_dset):
            subset = _filter_rows(rows, dset)
            print(f'\n[{dset}] ({len(subset)} cases)')
            print('='*len(header))
            print(header)
            print(line)
            for name, res in subset:
                print(f'{name:<{len_name}}' + ''.join(f'{float(res[key]):{width}.4f}' if key in res else f'{"-":>{width}}' for key in list_metrics))
            print(line)
            print(_mean_row('Mean', subset, list_metrics, len_name, width, std=std))
            print('='*len(header))
    else:
        print('='*len(header))
        print(header)
        print(line)
        for name, res in rows:
            print(f'{name:<{len_name}}' + ''.join(f'{float(res[key]):{width}.4f}' if key in res else f'{"-":>{width}}' for key in list_metrics))
        print(line)
        print(_mean_row('Mean', rows, list_metrics, len_name, width, std=std))
        print('='*len(header))


def _total_datasets(list_name, target_dset=None):
    if target_dset is not None: return _dataset_names([], target_dset)

    datasets = set()
    for epoch, alpha, kernel in list_name:
        if kernel == '--': continue
        if epoch < 0 and not (_path_res(kernel)/'model/history.pt').exists(): continue
        epoch = _epoch_tag(epoch, kernel)
        path_eval = _path_res(kernel)/'evaluation'/f'epoch_{epoch}'
        if path_eval.exists(): datasets.update(path.name for path in path_eval.iterdir() if path.is_dir() and any(path.glob('*.pt')))

    return sorted(datasets)


def print_total(list_name, len_name=None, std=False, list_metrics=None, target_dset=None, by_dset=False, print_preset=False):
    if list_metrics is None: list_metrics = ['mDSC', 'mclDice', 'ACC', 'F1', 'SC', 'TD', 'TAcc']
    if print_preset: _print_preset([kernel for _, _, kernel in list_name])
    if target_dset is not None: print(f'Target dataset: {", ".join(target_dset) if isinstance(target_dset, list) else target_dset}', flush=True)
    len_name = max(0 if len_name is None else len_name, len('Model')+2, *(len(kernel)+2 for _, _, kernel in list_name if kernel != '--'))
    width = 14 if std else 10
    header = f'{"Model":<{len_name}}{"Epoch":>7}{"N":>5}' + ''.join(f'{key:>{width}}' for key in list_metrics)
    line = '-'*len(header)

    def print_table(label, target):
        if label is not None: print(f'\n[{label}]', flush=True)
        print('='*len(header), flush=True)
        print(header, flush=True)
        print(line, flush=True)

        for epoch, alpha, kernel in list_name:
            if kernel == '--':
                print(line, flush=True)
                continue

            e, rows = _load_evaluation(epoch, kernel)
            rows = _filter_rows(rows, target)
            metrics = [_metric_text([float(res[key]) for _, res in rows if key in res], std=std) for key in list_metrics]
            print(f'{kernel:<{len_name}}{e:7d}{len(rows):5d}' + ''.join(f'{value:>{width}}' for value in metrics), flush=True)

        print('='*len(header), flush=True)

    if by_dset:
        for dset in _total_datasets(list_name, target_dset): print_table(dset, dset)
    else:
        print_table(None, target_dset)
