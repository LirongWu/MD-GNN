from MDGNN import MDGNN


def get_model(param):

    if param['model'] == 'MDGNN':
        model = MDGNN(param)

    return model