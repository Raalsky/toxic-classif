import sys
import hashlib
import random
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
from pathlib import Path


def hash_name(name):
    return hashlib.md5(name.encode()).hexdigest()


MODELS_DIR = Path('models')
PARAMS_DIR = Path('params')
DATASET_DIR = Path('dataset')


def preapre_environment():
    random.seed(2501)
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        deprecation._PRINT_DEPRECATION_WARNINGS = False
    except:
        sys.stderr.write('Could not get GPU to run\n')
