import hashlib
import tensorflow as tf
from pathlib import Path


def hash_name(name):
    return hashlib.md5(name.encode()).hexdigest()


MODELS_DIR = Path('models')
PARAMS_DIR = Path('params')

def preapre_environment():
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        sys.stderr.write('Could not get GPU to run\n')
