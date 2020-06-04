import hashlib
from pathlib import Path


def hash_name(name):
    return hashlib.md5(name.encode()).hexdigest()


MODELS_DIR = Path('models')
PARAMS_DIR = Path('params')
