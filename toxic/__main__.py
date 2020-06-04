import sys
from bottle import run, route, request, static_file
from toxic import ToxicClassifier, preapre_environment

classifier = ToxicClassifier()
port = int(sys.argv[1]) or 6754


@route('/', method='GET')
def static():
    return static_file('api.html', root='web/')


@route('/predict', method='POST')
def predict():
    return classifier.predict(request.json.get('sequences', []))


def serve():
    preapre_environment()
    run(host='0.0.0.0', port=port, reloader=True)


if __name__ == '__main__':
    serve()
