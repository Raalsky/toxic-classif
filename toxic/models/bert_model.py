from toxic.models.base import ToxicClassifier

import tensorflow as tf
from transformers import BertTokenizer, BertConfig, TFBertModel

class BertToxicClassifier(ToxicClassifier):
    def __init__(self):
        ToxicClassifier.__init__(pretrained_weights_name='bert-base-uncased')

    def architecture(self):
        input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32,
                                          name="input_ids")
        input_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32,
                                            name="segment_ids")

        last_hidden_state, pooler_output = transformer_model(input_ids,
                                                             attention_mask=input_mask,
                                                             token_type_ids=segment_ids)

        x = tf.keras.layers.Dense(256, activation="relu")(pooler_output)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=[input_ids, input_mask, segment_ids],
                               outputs=x)

        for layer in model.layers[:4]:
            layer.trainable = False
