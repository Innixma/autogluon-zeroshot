from pathlib import Path

from autogluon.common.loaders import load_json


if __name__ == '__main__':
    json_root = Path(__file__).parent.parent / 'data' / 'configs' / 'zeroshot'

    configs_zs = load_json.load(path=json_root / 'configs_zs_20221004.json')

    hyperparameters_dict = {}

    for name in configs_zs:
        config = configs_zs[name]
        hyperparameters = config['hyperparameters']
        model_type = config['model_type']
        print(config)

        if model_type not in hyperparameters_dict:
            hyperparameters_dict[model_type] = []

        hyperparameters_dict[model_type].append(hyperparameters)

    print(hyperparameters_dict)


