from toxic.models.architectures import BertToxicClassifier
cls = BertToxicClassifier(load_weights=True)
print(cls.predict([
    'You are so stupid'
]))
