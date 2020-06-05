from toxic.models import BertToxicClassifier as ToxicClassifier
from toxic.utils import preapre_environment
import sys
import json
import requests
from bottle import Bottle, request, static_file


def server():
    port = int(sys.argv[1])
    preapre_environment()
    classifier = ToxicClassifier(load_weights=True)

    app = Bottle()

    @app.route('/', method='GET')
    def static():
        return static_file('api.html', root='web/')

    @app.route('/predict', method='POST')
    def predict():
        return {
            'results': classifier.predict(request.json.get('sequences', []))
        }

    preapre_environment()
    app.run(host='0.0.0.0', port=port, reloader=True)


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
