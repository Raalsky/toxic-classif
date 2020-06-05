# Toxic Comments Classificator
```
python3.6 -m venv venv
source venv/bin/activate
python setup.py install
```

## Serving
### Client
`cat <sample-per-line>.csv | toxic_client http://<id>.ngrok.io <batch-size>`
### Server
`toxic_server <port> <model-name>`
For `model-name` specity `deploy`
