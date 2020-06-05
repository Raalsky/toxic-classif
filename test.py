# from toxic.models import BertToxicClassifier as ToxicClassifier

import requests
import numpy as np

# cls = ToxicClassifier(load_weights=True, tta_fold=0)

# text = "It’s mayors and county executives that appoint most police chiefs and negotiate collective bargaining agreements with police unions. It’s district attorneys and state’s attorneys that decide whether or not to investigate and ultimately charge those involved in police misconduct. Those are all elected positions. In some places, police review boards with the power to monitor police conduct are elected as well. Unfortunately, voter turnout in these local races is usually pitifully low, especially among young people – which makes no sense given the direct impact these offices have on social justice issues, not to mention the fact that who wins and who loses those seats is often determined by just a few thousand, or even a few hundred, votes."

# print(text)

# ids, masks, segments = cls.get_tokens([text])

# np.save("ids.npy", ids, allow_pickle=False)
# np.save("masks.npy", masks, allow_pickle=False)
# np.save("segments.npy", segments, allow_pickle=False)

ids = np.load("ids.npy")
masks = np.load("masks.npy")
segments = np.load("segments.npy")

# print(cls.predict([text]))

# cls.save()

# input_ids, input_mask, segment_ids

from pprint import pprint

SERVER_URL = 'http://0.0.0.0:8501/v1/models/deploy/metadata'

response = requests.get(SERVER_URL)
response.raise_for_status()
response = response.json()

pprint(response)

import json

input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": [
        {
            "input_ids": ids[u].tolist(),
            "input_mask": masks[u].tolist(),
            "segment_ids": segments[u].tolist()
        }
        for u in range(len(ids))
    ]
})


print(input_data_json)

SERVER_URL = 'http://localhost:8501/v1/models/deploy:predict'

response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status()
response = response.json()

print(response)