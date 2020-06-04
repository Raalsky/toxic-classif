from toxic.models import BertToxicClassifier as ToxicClassifier

cls = ToxicClassifier(load_weights=True, tta_fold=8)

print(cls.predict([
    'You are so stupid'
]))
