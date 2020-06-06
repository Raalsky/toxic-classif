import os
import sys

import optuna
import neptune

from toxic.utils import preapre_environment, compress_directory
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
    dropout = trial.suggest_uniform('dropout', 0.0, 0.3)
    attention_dropout = trial.suggest_uniform('attention_dropout', 0.0, 0.2)
    trainable_embedding = trial.suggest_categorical('trainable_embedding', [False, True])
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-7, 1e-4)
    epochs = trial.suggest_int('epochs', 1, 3)
    batch_size = trial.suggest_int('batch_size', 16, 64, 8)
    pretrained_weights_name = trial.suggest_categorical('pretrained_weights_name', [
        'bert-base-uncased'
    ])

    cls = BertToxicClassifier(
        max_seq_length=max_seq_length,
        dropout=dropout,
        attention_dropout=attention_dropout,
        trainable_embedding=trainable_embedding,
        learning_rate=learning_rate,
        pretrained_weights_name=pretrained_weights_name,
        tta_fold=8
    )
    (x_train, y_train), (x_validation, y_validation) = cls.load_datasets(refresh=True)

    neptune.create_experiment(name=cls.model_name, params=trial.params)

    for tag in cls.tags:
        neptune.append_tag(tag)

    # Training
    cls.train(x_train, y_train,
              x_validation, y_validation,
              epochs=epochs,
              batch_size=batch_size
              )

    # Evaluation
    x_test, y_test = cls.load_dataset_dataframe('test')
    test_acc = cls.evaluate_with_tta(x_test, y_test)

    neptune.send_metric('test_acc', 100.0 * test_acc)
    neptune.send_text('path', cls.save())

    # neptune.send_artifact(compress_directory(model_path), 'model.tar.gz')

    return 1-test_acc


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
