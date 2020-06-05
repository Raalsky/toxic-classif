import os
import sys

from toxic.utils import preapre_environment
from toxic.models import ToxicClassifierBase, BertToxicClassifier


def server():
    port = int(sys.argv[1])
    model_name = sys.argv[2]
    os.system(f"docker run -it --rm -p {port}:8501 -v \"$PWD/models/{model_name}:/models/deploy\" -e MODEL_NAME=deploy tensorflow/serving")


def print_results(predictions):
    for pred in predictions:
        print(int(pred['toxic']))


def client():
    host = sys.argv[1] + '/v1/models/deploy'
    batch_size = int(sys.argv[2])

    dummy_classifier = BertToxicClassifier(initialize_model=False)

    batch = []

    for line in sys.stdin:
        data = line.rstrip()
        batch.append(data)

        if len(batch) >= batch_size:
            print_results(
                dummy_classifier.predict_from_api(host, batch)
            )
            batch = []

    print_results(
        dummy_classifier.predict_from_api(host, batch)
    )


def train():
    preapre_environment()

    cls = BertToxicClassifier()

    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = cls.load_datasets(refresh=True)

    cls.train(x_train, y_train, x_validation, y_validation)
