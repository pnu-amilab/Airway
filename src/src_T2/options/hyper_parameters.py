from dataclasses import dataclass, field
from pathlib import Path
import os

from common.utils.utils_common import make_dir
from src_T1.options.hyper_parameters import HP


@dataclass
class HP_T2(HP):
    list_dset_preset: dict = field(default_factory=lambda: {
        'Preset1':{'train':['ATM26_Shanghai'], 'test':['ATM26_Shanghai']}})
    name_model: str = 'ASTRA_Net'
    num_classes: int = 21
    background_index: int = 0
    num_lobes: int = 5
    class_dim: int = 64
    branch_min_voxels: int = 4
    deep_supervision_weight: float = 0.15
    lobe_bias_strength: float = 0.75

    path_track1_checkpoint: str = ''
    path_init_model: str = ''
    path_cache: str = ''
    track1_threshold: float = 0.20
    use_extra_train_T2: bool = False

    random_sampling_ratio: float = 0.50
    skeleton_sampling_jitter: int = 20
    class_weight_power: float = 0.35
    class_weight_max: float = 3.0

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
        self.path_Res = str(path_res_root/'T2')
        self.path_res = f'{self.path_Res}/{self.name}'
        self.path_model = f'{self.path_res}/model'
        self.path_inference = f'{self.path_res}/inference'
        self.path_evaluation = f'{self.path_res}/evaluation'
        self.path_fig = f'{self.path_res}/fig'
        self.path_temp = str(path_src/'cache'/self.name)
        self.path_logs = str(path_src/'logs'/'T2'/self.name)
        if not self.path_track1_checkpoint: self.path_track1_checkpoint = str(path_res_root/'T1'/'Preset1_Ours_ASTRA_Net_phase_2'/'model'/'model_200.pt')
        if not self.path_init_model: self.path_init_model = self.path_track1_checkpoint
        if not self.path_cache: self.path_cache = str(path_data/'ATM_T2_Track1_Cache')
        if create_dirs:
            for path in [self.path_Res, self.path_res, self.path_model, self.path_inference, self.path_evaluation, self.path_fig, self.path_temp, self.path_logs]: make_dir(path)


def load_local_HP_T2(hp):
    if hp.name_model != 'ASTRA_Net': raise ValueError(f'Unsupported T2 model: {hp.name_model}')

    hp.name_group = 'Ours'
    hp.use_PE = True
    hp.use_LM = True
    hp.use_CV = True
    hp.use_MW = True
    hp.use_LW = False
    if 'CT' in hp.name_ablation:
        hp.use_LM = False
        hp.use_CV = False
        hp.use_PE = False
        hp.lobe_bias_strength = 0.0
    hp.amp = True
    hp.size_patch = 64
    hp.batch_size = 24
    hp.batch_size_inference = 16
    hp.epochs = 1000
    hp.save_cycle = 100
    hp.optimizer_lr = 2e-5
    hp.scheduler_step = 100
    hp.scheduler_gamma = 0.8
    hp.overlap_inference = 32
    hp.best_metric = 'Score'
    hp.loss_weight = {'CCE':1.0, 'HCE':0.50, 'HDL':0.15, 'BrCE':0.75, 'BrCon':0.20}
    hp.dset_preset = hp.list_dset_preset[hp.name_dset]
    hp.refresh_paths()
    hp.dim_in = 3 if 'CT' in hp.name_ablation else 1+1+1+hp.num_lobes+1+6
    hp.dim_true = 4

    return hp
