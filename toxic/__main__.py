import os
import sys
import json
import requests


def server(port=6666, model_name='deploy'):
    os.system(f"docker run -it --rm -p {port}:8501 -v \"$PWD/models/{model_name}:/models/{model_name}\" -e MODEL_NAME={model_name} tensorflow/serving")


def fetch_scores_and_print(host, sequences):
    payload = {
        'sequences': sequences
    }

    response = requests.post(host, json=payload, timeout=30)

    assert response.status_code == 200

    results = json.loads(response.content)['results']

    for score in results:
        print(int(score['toxic']))


def client():
    host = sys.argv[1]
    batch_size = int(sys.argv[2])

    batch = []

    for line in sys.stdin:
        data = line.rstrip()
        batch.append(data)

        if len(batch) >= batch_size:
            fetch_scores_and_print(f"{host}/predict", batch)
            batch = []

    fetch_scores_and_print(f"{host}/predict", batch)
