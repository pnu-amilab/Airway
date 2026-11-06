from src_T2.runners.runner_classifier import Runner_Classifier_T2


def get_runner(hp):
    if hp.name_model != 'ASTRA_Net': raise ValueError(f'Unsupported T2 model: {hp.name_model}')

    return Runner_Classifier_T2(hp)


def get_runner_T2(hp):
    return get_runner(hp)
