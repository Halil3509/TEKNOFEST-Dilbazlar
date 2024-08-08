import yaml


def get_configs():
    """
    Getting configuration file
    :return:
    """
    # Load YAML file
    with open('configs.yaml', 'r') as file:
        data = yaml.safe_load(file)

    return data
