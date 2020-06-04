import tensorflow as tf


def preapre_environment():
    physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)

    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # print('Tensorflow sets memory gropth to True')
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


preapre_environment()
