import sys
from bottle import run, route, request, static_file
from toxic import ToxicClassifier

classifier = ToxicClassifier()


@route('/', method='GET')
def static():
    return static_file('api.html', root='web/')


@route('/predict', method='POST')
def predict():
    return classifier.predict(request.json.get('sequences', []))


run(host='0.0.0.0', port=int(sys.argv[1]), reloader=True)
