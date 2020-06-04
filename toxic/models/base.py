import tensorflow as tf
import numpy as np

from .utils import hash_name


class ToxicClassifier:
    def __init__(self,
                 model_name,
                 tokenizer_cls,
                 config_cls,
                 embedding_model_cls,
                 pretrained_weights_name,
                 load_weights=False,
                 load_params=False,
                 max_seq_length=32,
                 dropout=0.2,
                 attention_dropout=0.2,
                 threshold=0.5,
                 do_lower_case=True
                 ):
        self.model_name = model_name
        self.model_name_hash = hash_name(model_name)
        self.pretrained_weights_name = pretrained_weights_name
        self.tokenizer_cls = tokenizer_cls
        self.config_cls = config_cls
        self.embedding_cls = embedding_model_cls

        if load_params:
            self.load_params_from_file()

        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.threshold = threshold

        self.initialize_model()

        if load_weights:
            self.load_weights_from_file()

    def load_params_from_file(self):
        pass

    def load_weights_from_file(self):
        pass

    def initialize_model(self):
        self.config = self._get_embedding_config()
        self.tokenizer = self._init_tokenizer()
        self.model = self.architecture
        self.model.compile()
        self.model.load

    def _get_tokenizer(self):
        return self.tokenizer_cls.from_pretrained(
            self.pretrained_weights_name,
            do_lower_case=self.do_lower_case,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=True
        )

    def tokenize(self, sequences):
        return self.tokenizer.tokenize(sequences)

    def get_tokens(self, sequences):
        input_ids, input_masks, input_segments = [], [], []

        for sentence in sequences:
            inputs = self.tokenizer.encode_plus(sentence,
                                                add_special_tokens=True,
                                                max_length=self.max_seq_length,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_token_type_ids=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])

        return np.asarray(input_ids, dtype='int32'), \
               np.asarray(input_masks, dtype='int32'), \
               np.asarray(input_segments, dtype='int32')

    @property
    def embedding_config(self):
        config = self.config_cls(dropout=self.dropout,
                                 attention_dropout=self.attention_dropout)
        config.output_hidden_states = False
        return config

    @property
    def embedding_architecture(self):
        transformer_model = self.embedding_cls.from_pretrained(
            self.pretrained_weights_name,
            config=self.embedding_config)
        transformer_model.trainable = self.embedding_trainable
        return transformer_model

    @property
    def inputs(self):
        input_ids = tf.keras.layers.Input(shape=(self.max_seq_length,),
                                          dtype=tf.int32,
                                          name="input_ids")
        input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,),
                                           dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.max_seq_length,),
                                            dtype=tf.int32,
                                            name="segment_ids")

        return input_ids, input_mask, segment_ids

    @property
    def architecture(self):
        input_ids, input_mask, segment_ids = self.inputs

        last_hidden_state, pooler_output = self.embedding_architecture(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids)

        x = self.classifier_architecture(pooler_output)

        model = tf.keras.Model(inputs=[input_ids, input_mask, segment_ids],
                               outputs=x)

        return model

    def classifier_architecture(self, input_layer):
        x = tf.keras.layers.Dense(256, activation="relu")(input_layer)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return x

    def threshold_prediction(self, prediction):
        return (prediction > self.threshold)

    def raw_predict(self, sequences):
        # TODO: TTA
        return self.model.predict(self.get_tokens(sequences))

    def predict(self, sequences, precision: int = 1):
        return [
            {
                'score': raw_score.round(precision),
                'toxic': self.threshold_prediction(raw_score)
            }
            for raw_score in self.raw_predict(sequences)
        ]
