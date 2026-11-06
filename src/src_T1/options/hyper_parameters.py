from dataclasses import dataclass, field
from pathlib import Path
import os

from common.utils.utils_common import make_dir


@dataclass
class HP_Preprocessing:
    path_main: str = field(default_factory=lambda: str(Path(os.environ.get('AIRWAY_DATA_ROOT', Path(__file__).resolve().parents[2].parent/'data'))))
    scale_factor: int = 2
    MW_min: int = 1
    MW_max: int = 8
    MW_alpha: int = 2
    MW_beta: int = 1
    kernel_Size_MW: int = 3
    kernel_size_SM: int = 5
    use_LW: bool = False
    sigma_SM: float = 1.0
    LM_inferer_batch_size: int = 4


@dataclass
class HP:
    name_dset: str = 'Preset1'
    name_group: str = 'Others'
    name_model: str = 'ASTRA_Net_phase_1'
    name_ablation: str = ''
    device: str = 'cuda'
    amp: bool = True
    run_tqdm: bool = True
    name: str = ''

    alpha: int = 10
    dset_preset: dict = field(default_factory=dict)
    list_dset_preset: dict = field(default_factory=lambda: {
        'Preset1':{'train':['ATM26_Shanghai', 'ATM26_LIDC'],
                   'test':['ATM26_Shanghai', 'ATM26_LIDC', 'ATM26_EXACT09', 'ATM26_Enhanced', 'PNUYH']}})

    use_LM: bool = True
    use_CV: bool = True
    use_PE: bool = True
    use_MW: bool = True
    use_LW: bool = False

    ega: bool = True
    ega_learnable: bool = True
    ega_recalibration: bool = True

    size_patch: int = 64
    dim_base: int = 16
    m: int = -1200
    M: int = 600

    num_workers: int = 8
    epochs: int = 2000
    epoch_load: int = 0
    batch_size: int = 27
    optimizer_lr: float = 1e-4
    optimizer_wd: float = 1e-2
    scheduler_step: int = 100
    scheduler_gamma: float = 0.8
    scheduler_warmup_ratio: float = 0.05
    save_cycle: int = 50
    batch_size_inference: int = 32
    overlap_inference: int = 32
    inference_edge: int = 65
    alpha_evaluation: float = 0.20
    clip_grad_norm: float = 3.0
    best_metric: str = 'DSC'
    full_train: bool = False

    random_sampling_ratio: float = 0.50
    skeleton_sampling_jitter: int = 20

    gul_alpha_start: float = 0.10
    gul_alpha_end: float = 0.20
    gul_gamma: float = 0.70
    bgr_alpha: float = 1e-5
    loss_schedule_warmup: float = 0.10

    def refresh_paths(self, create_dirs=True):
        self.name = f'{self.name_dset}_{self.name_group}_{self.name_model}'
        if self.name_ablation: self.name += f'_{self.name_ablation}'
        path_src = Path(__file__).resolve().parents[2]
        path_data = Path(os.environ.get('AIRWAY_DATA_ROOT', path_src.parent/'data'))
        path_train = Path(os.environ.get('AIRWAY_TRAIN_DATA_ROOT', path_data))
        path_res_root = Path(os.environ.get('AIRWAY_RES_ROOT', path_src.parent/'res'))
        self.path_main = str(path_src)
        self.path_data = str(path_data/'Dataset')
        self.path_data_train = str(path_train/'Dataset')
        self.path_Res = str(path_res_root/'T1')
        self.path_res = f'{self.path_Res}/{self.name}'
        self.path_model = f'{self.path_res}/model'
        self.path_inference = f'{self.path_res}/inference'
        self.path_evaluation = f'{self.path_res}/evaluation'
        self.path_fig = f'{self.path_res}/fig'
        self.path_temp = str(path_src/'cache'/self.name)
        self.path_logs = str(path_src/'logs'/'T1'/self.name)
        if create_dirs:
            for path in [self.path_Res, self.path_res, self.path_model, self.path_inference, self.path_evaluation, self.path_fig, self.path_temp, self.path_logs]: make_dir(path)


def load_local_HP(hp):
    if hp.name_model in ['UNet', 'ResUNet', 'WingsNet', 'FANN', 'TSE_Net', 'UDADS_Net']:
        hp.name_group = 'Others'
        hp.use_PE = False
        hp.use_LM = False
        hp.use_CV = False
        hp.use_MW = False
        hp.use_LW = False
        hp.amp = hp.name_model not in ['FANN', 'TSE_Net']
        hp.loss_weight_start = {'GUL':1.0, 'BCE':1.0}
        hp.loss_weight_end = hp.loss_weight_start.copy()
        hp.loss_weight = hp.loss_weight_start.copy()
        if hp.name_model in ['UNet', 'ResUNet']:
            hp.gul_alpha_start = 0.50
            hp.gul_alpha_end = 0.50
            hp.gul_gamma = 1.00
        if hp.name_model == 'FANN': hp.batch_size = 14
        if hp.name_model == 'WingsNet': hp.use_LW = True
        if hp.name_model in ['WingsNet', 'UDADS_Net']: hp.batch_size = 22
    elif hp.name_model == 'ASTRA_Net_phase_1':
        hp.name_group = 'Ours'
        hp.use_PE = True
        hp.use_LM = True
        hp.use_CV = True
        hp.use_MW = True
        hp.use_LW = False
        hp.amp = True
        hp.size_patch = 64
        hp.batch_size = 27
        hp.batch_size_inference = 32
        hp.overlap_inference = 32
        hp.save_cycle = 50
        hp.gul_alpha_start = 0.10
        hp.gul_alpha_end = 0.20
        hp.bgr_alpha = 1e-5
        hp.random_sampling_ratio = 0.50
        hp.loss_weight_start = {'GUL':1.0, 'BCE':1.0, 'CTL':1.0, 'BGR':1.0}
        hp.loss_weight_end = hp.loss_weight_start.copy()
        hp.loss_weight = hp.loss_weight_start.copy()
        hp.alpha_evaluation = 0.20
    elif hp.name_model == 'ASTRA_Net_phase_2':
        hp.name_group = 'Ours'
        hp.use_PE = True
        hp.use_LM = True
        hp.use_CV = True
        hp.use_MW = True
        hp.use_LW = False
        hp.alpha = 10
        hp.epochs = 200
        hp.batch_size = 10
        hp.optimizer_lr = 2e-6
        hp.scheduler_step = 10
        hp.amp = True
        hp.loss_weight = {'GUL':1.0, 'BCE':1.0, 'KDL':1.0, 'IDL':1.0}
        hp.name_teacher = f'{hp.name_dset}_Ours_ASTRA_Net_phase_1'
        hp.alpha_evaluation = 0.20
    else:
        raise ValueError(f'Unsupported model: {hp.name_model}')

    if 'CT' in hp.name_ablation:
        hp.use_LM = False
        hp.use_CV = False
        hp.use_PE = False
    if hp.name_model == 'ASTRA_Net_phase_2' and hp.name_ablation:
        hp.name_teacher += f'_{hp.name_ablation}'

    hp.dset_preset = hp.list_dset_preset[hp.name_dset]
    hp.refresh_paths()
    hp.dim_in = 1
    if hp.use_LM: hp.dim_in += 1
    if hp.use_CV: hp.dim_in += 1
    if hp.use_PE: hp.dim_in += 6
    hp.dim_true = 3
    if hp.use_MW or hp.use_LW: hp.dim_true += 1

    return hp
