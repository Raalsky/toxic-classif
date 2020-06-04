import tensorflow as tf


class ToxicClassifier:
    def __init__(self):
        pass

    def predict(self, sequences):
        return {
            'scores': [
                {
                    'score': 0.75,
                    'toxic': True
                } for seq in sequences
            ]
        }
