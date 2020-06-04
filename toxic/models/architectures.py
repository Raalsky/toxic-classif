import transformers

from .base import ToxicClassifier


class BertToxicClassifier(ToxicClassifier):
    def __init__(self):
        ToxicClassifier.__init__(
            model_name='bartek-000',
            tokenizer_cls=transformers.BertTokenizer,
            config_cls=transformers.BertConfig,
            embedding_model_cls=transformers.TFBertModel,
            pretrained_weights_name='bert-base-uncased'
        )
