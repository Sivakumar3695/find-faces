import os
from pyaml_env import parse_config


def load():
    config = parse_config(f'{os.getcwd()}/config.yml', default_value='')
    print(config)
    print(config['app_config']['db_url'])
    return config['app_config']
