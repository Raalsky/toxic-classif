from pathlib import Path

import numpy as np

class ToxicClassifier:
    def __init__(self, pretrained_weights_name, max_seq_length: int = 32):
        self.pretrained_weights_name = pretrained_weights_name
        self.tokenizer_cls = None
        self.do_lower_case = True
        self.max_seq_length = max_seq_length
        self.MODELS_DIR = Path('models')
        self.tokenizer = self._init_tokenizer()

    def _get_tokenizer(self):
        return self.tokenizer_cls.from_pretrained(self.pretrained_weights_name,
                                                  do_lower_case=self.do_lower_case,
                                                  add_special_tokens=True,
                                                  max_length=self.max_seq_length,
                                                  pad_to_max_length=True)

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

        return np.asarray(input_ids, dtype='int32'),\
               np.asarray(input_masks, dtype='int32'),\
               np.asarray(input_segments,dtype='int32')


    def _get_embedding_architecture(self):
        config = BertConfig(dropout=0.2, attention_dropout=0.2)
        config.output_hidden_states = False
        transformer_model = TFBertModel.from_pretrained(model_name,
                                                        config=config)
        return self.embed

    def _get_classifier_architecture(self):
        raise NotImplementedError
