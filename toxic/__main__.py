from toxic.models import BertToxicClassifier as ToxicClassifier
from toxic.utils import preapre_environment
from bottle import Bottle, request, static_file


def serve(port):
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


if __name__ == '__main__':
    serve(port=6666)
