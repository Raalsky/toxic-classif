import tensorflow as tf
import transformers

from .base import ToxicClassifier


class BertToxicClassifier(ToxicClassifier):
    def __init__(self, load_weights: bool = False):
        super(BertToxicClassifier, self).__init__(
            model_name='TOX-1',
            tokenizer_cls=transformers.BertTokenizer,
            config_cls=transformers.BertConfig,
            embedding_model_cls=transformers.TFBertModel,
            pretrained_weights_name='bert-base-uncased',
            load_weights=load_weights
        )

    def classifier_architecture(self, input_layer):
        x = tf.keras.layers.Dense(256, activation="relu")(input_layer)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        return x
