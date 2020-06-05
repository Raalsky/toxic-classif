import tensorflow as tf
import transformers
import numpy as np
import itertools
import requests
import nltk
import time
import json
import os


import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf

from .utils import hash_name, MODELS_DIR, PARAMS_DIR


class ToxicClassifierBase:
    def __init__(self,
                 model_name,
                 tokenizer_cls,
                 config_cls,
                 embedding_model_cls,
                 pretrained_weights_name,
                 load_weights=False,
                 load_params=False,
                 initialize_model=True,
                 max_seq_length=32,
                 dropout=0.2,
                 attention_dropout=0.2,
                 threshold=0.5,
                 do_lower_case=True,
                 trainable_embedding=False,
                 tta_fold=0
                 ):
        self.model_name = model_name
        self.model_name_hash = hash_name(model_name)
        self.pretrained_weights_name = pretrained_weights_name
        self.tokenizer_cls = tokenizer_cls
        self.config_cls = config_cls
        self.embedding_cls = embedding_model_cls
        self.model = None

        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.threshold = threshold
        self.trainable_embedding = trainable_embedding
        self.tta_fold = tta_fold

        self.learning_rate = 2e-6
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.loss = "binary_crossentropy"
        self.metrics = ["accuracy"]

        self.load_params_from_file_if_specified(load_params)
        self.initialize_model_if_specified(initialize_model)
        self.load_weights_from_file_if_specified(load_weights)

    def load_params_from_file_if_specified(self, load_params):
        if load_params:
            pass
        pass

    def load_weights_from_file_if_specified(self, load_weights):
        if load_weights:
            self.model.load_weights(str(
                MODELS_DIR / self.model_name_hash / 'weights.h5'
            ))

    def initialize_model_if_specified(self, initialize_model):
        if initialize_model:
            self.model = self.architecture()
            self.model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics)

    @property
    def tokenizer(self):
        return self.tokenizer_cls.from_pretrained(
            self.pretrained_weights_name,
            do_lower_case=self.do_lower_case,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=True
        )

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
        transformer_model.trainable = self.trainable_embedding
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
        raise NotImplementedError

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

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

    def _judge_prediction(self, prediction, precision: int = 3):
        score = np.asscalar(prediction)
        return {
            'score': round(score, precision),
            'toxic': score > self.threshold
        }

    def augment_one(self, sequence):
        return sequence

    def augment(self, sequence):
        return [sequence] + [self.augment_one(sequence)] * self.tta_fold

    def test_time_augment(self, sequences):
        return list(itertools.chain(*list(map(self.augment, sequences))))

    def score_after_tta(self, predictions):
        # Numpy magic aka averaging every "tta_fold" subarrays
        predictions = predictions[:, 0]
        ids = np.arange(len(predictions)) // (1 + self.tta_fold)
        return np.bincount(ids, predictions) / np.bincount(ids)

    def save(self):
        model_version = str(int(time.time()))
        path = MODELS_DIR / self.model_name_hash / model_version
        os.makedirs(path, exist_ok=True)
        print(f"Saving model into {path}")
        tf.saved_model.save(self.model, str(path))

    def raw_predict(self, sequences):

        return self.score_after_tta(
                    self.model.predict(
                        self.get_tokens(
                            self.test_time_augment(sequences)
                        )
                    )
                )

    def judge_predictions(self, predictions):
        return [
            self._judge_prediction(prediction)
            for prediction in predictions
        ]

    def predict(self, sequences):
        return self.judge_predictions(
            self.raw_predict(sequences)
        )

    def predict_from_api(self, host, sequences):
        ids, masks, segments = self.get_tokens(
            self.test_time_augment(sequences)
        )

        input_data_json = json.dumps({
            "signature_name": "serving_default",
            "instances": [
                {
                    "input_ids": ids[u].tolist(),
                    "input_mask": masks[u].tolist(),
                    "segment_ids": segments[u].tolist()
                }
                for u in range(len(ids))
            ]
        })

        response = requests.post(host + ':predict', data=input_data_json)
        response.raise_for_status()
        predictions = np.array(response.json()['predictions'])

        return self.judge_predictions(
            self.score_after_tta(predictions)
        )

    def load_dataset(self):



class BertToxicClassifier(ToxicClassifierBase):
    def __init__(self,
                 load_weights: bool = False,
                 tta_fold: int = 0,
                 initialize_model: bool = True
                 ):
        super(BertToxicClassifier, self).__init__(
            model_name='TOX-1',
            tokenizer_cls=transformers.BertTokenizer,
            config_cls=transformers.BertConfig,
            embedding_model_cls=transformers.TFBertModel,
            pretrained_weights_name='bert-base-uncased',
            load_weights=load_weights,
            initialize_model=initialize_model,
            tta_fold=tta_fold
        )
        try:
            self.tta_composition = naf.Sequential([
                naw.SynonymAug(aug_src='wordnet'),
                naw.RandomWordAug(),
                nac.KeyboardAug()
            ])
        except:
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')

    def augment_one(self, sequence):
        return self.tta_composition.augment(sequence)

    def classifier_architecture(self, input_layer):
        x = tf.keras.layers.Dense(256, activation="relu")(input_layer)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        return x
