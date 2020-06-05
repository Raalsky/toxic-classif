import os
import sys

from toxic.models import ToxicClassifierBase,\
    BertToxicClassifier as ToxicClassifier


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

    dummy_classifier = ToxicClassifier(initialize_model=False)

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
    cls = ToxicClassifier()

    train, validation, test = ToxicClassifierBase.load_dataset()

    cls.train(train, validation)
