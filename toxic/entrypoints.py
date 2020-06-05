import os
import sys

import optuna
import neptune

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


def train(trial):
    max_seq_length = trial.suggest_categorical('max_seq_length', [32, 64, 128, 256, 512])
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    attention_dropout = trial.suggest_uniform('attention_dropout', 0.0, 0.5)
    trainable_embedding = trial.suggest_categorical('trainable_embedding', [False, True])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1e-4)

    cls = BertToxicClassifier(
        max_seq_length=max_seq_length,
        dropout=dropout,
        attention_dropout=attention_dropout,
        trainable_embedding=trainable_embedding,
        learning_rate=learning_rate
    )
    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = cls.load_datasets(refresh=False)

    neptune.create_experiment(name=cls.model_name_hash, params=trial.params)

    for tag in cls.tags:
        neptune.append_tag(tag)

    cls.train(x_train, y_train, x_validation, y_validation)
    test_loss, test_acc = cls.evaluate(x_test, y_test)

    neptune.send_metric('test_loss', test_loss)
    neptune.send_metric('test_acc', 100.0 * test_acc)

    return test_acc


def optimization():
    n_trials = int(sys.argv[1])
    preapre_environment()
    neptune.init('raalsky/toxic')
    study = optuna.create_study(
        study_name='toxicity',
        storage='sqlite:///studies.db',
        load_if_exists=True
    )

    study.optimize(train,
                   n_trials=n_trials,
                   timeout=3 * 60 * 60)

    print(study.best_params)
