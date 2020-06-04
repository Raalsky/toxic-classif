import sys
from .models import BertToxicClassifier


def preapre_environment():
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        sys.stderr.write('Could not get GPU to run\n')


ToxicClassifier = BertToxicClassifier
