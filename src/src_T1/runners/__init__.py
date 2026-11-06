from src_T1.runners.runner_ASTRA_Net_phase_1 import Runner_ASTRA_Net_phase_1
from src_T1.runners.runner_ASTRA_Net_phase_2 import Runner_ASTRA_Net_phase_2
from src_T1.runners.runner_others_UNet import Runner_UNet
from src_T1.runners.runner_others_ResUNet import Runner_ResUNet
from src_T1.runners.runner_others_WingsNet import Runner_WingsNet
from src_T1.runners.runner_others_FANN import Runner_FANN
from src_T1.runners.runner_others_TSE_Net import Runner_TSE_Net
from src_T1.runners.runner_others_UDADS_Net import Runner_UDADS_Net


def get_runner(hp):
    runners = {'ASTRA_Net_phase_1':Runner_ASTRA_Net_phase_1,
               'ASTRA_Net_phase_2':Runner_ASTRA_Net_phase_2,
               'UNet':Runner_UNet, 'ResUNet':Runner_ResUNet, 'WingsNet':Runner_WingsNet,
               'FANN':Runner_FANN, 'TSE_Net':Runner_TSE_Net,
               'UDADS_Net':Runner_UDADS_Net}
    if hp.name_model not in runners: raise ValueError(f'Unsupported model: {hp.name_model}')

    return runners[hp.name_model](hp)
