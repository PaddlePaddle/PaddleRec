import os


def encode_value(v):
    return v


def decode_value(v):
    return v


def set_global_envs(yaml, envs):
    for k, v in yaml.items():
        envs[k] = encode_value(v)


def get_global_env(env_name):
    """
    get os environment value
    """
    if env_name not in os.environ:
        raise ValueError("can not find config of {}".format(env_name))

    v = os.environ[env_name]
    return decode_value(v)
