from toxic.models import BertToxicClassifier as ToxicClassifier

import requests
import numpy as np

cls = ToxicClassifier(load_weights=True, tta_fold=0)

texts = [
    "You're stupid",
    "I like you very very much"
]

# ids, masks, segments = cls.get_tokens(texts)

# np.save("ids.npy", ids, allow_pickle=False)
# np.save("masks.npy", masks, allow_pickle=False)
# np.save("segments.npy", segments, allow_pickle=False)

# ids = np.load("ids.npy")
# masks = np.load("masks.npy")
# segments = np.load("segments.npy")

print(cls.predict(texts))

cls.save()

# input_ids, input_mask, segment_ids

# from pprint import pprint
#
# SERVER_URL = 'http://0.0.0.0:8501/v1/models/deploy/metadata'
#
# response = requests.get(SERVER_URL)
# response.raise_for_status()
# response = response.json()
#
# pprint(response)
#
# import json
#
# input_data_json = json.dumps({
#     "signature_name": "serving_default",
#     "instances": [
#         {
#             "input_ids": ids[u].tolist(),
#             "input_mask": masks[u].tolist(),
#             "segment_ids": segments[u].tolist()
#         }
#         for u in range(len(ids))
#     ]
# })
#
#
# print(input_data_json)
#
# SERVER_URL = 'http://localhost:8501/v1/models/deploy:predict'
#
# response = requests.post(SERVER_URL, data=input_data_json)
# response.raise_for_status()
# response = response.json()
#
# print(response)