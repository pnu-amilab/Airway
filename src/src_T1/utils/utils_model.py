import torch


def load_model(hp, epoch_load=0):
    if hp.name_model in ['ASTRA_Net_phase_1', 'ASTRA_Net_phase_2']:
        from src_T1.networks.network_ASTRA_Net import ASTRA_Net
        model = ASTRA_Net(dim_in=hp.dim_in, dim_base=hp.dim_base, ega=hp.ega, ega_learnable=hp.ega_learnable, ega_recalibration=hp.ega_recalibration)
    elif hp.name_model in ['UNet', 'ResUNet']:
        if hp.name_model == 'UNet':
            from src_T1.networks.network_others_UNet import UNet
            model = UNet(dim_in=hp.dim_in)
        else:
            from src_T1.networks.network_others_ResUNet import ResUNet
            model = ResUNet(dim_in=hp.dim_in)
    elif hp.name_model == 'WingsNet':
        from src_T1.networks.network_others_WingsNet import WingsNet
        model = WingsNet(size=hp.size_patch, dim_in=hp.dim_in, dim_base=16)
    elif hp.name_model == 'FANN':
        from src_T1.networks.network_others_FANN import FANN
        model = FANN(in_channel=hp.dim_in)
    elif hp.name_model == 'TSE_Net':
        from src_T1.networks.network_others_TSE_Net import TSE_Net
        model = TSE_Net(dim_in=hp.dim_in)
    elif hp.name_model == 'UDADS_Net':
        from src_T1.networks.network_others_UDADS_Net import UDADS_Net
        model = UDADS_Net()
    else:
        raise ValueError(f'Unsupported model: {hp.name_model}')

    if epoch_load > 0:
        model_state = torch.load(f'{hp.path_model}/model_{epoch_load}.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(model_state)
    elif epoch_load == -1:
        last_point = torch.load(f'{hp.path_model}/last_point.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(last_point['model'])
    elif epoch_load == -2:
        model_state = torch.load(f'{hp.path_model}/model_best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(model_state)

    return model
