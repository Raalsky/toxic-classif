from toxic import ToxicClassifier

cls = ToxicClassifier(load_weights=True)

print(cls.predict([
    'You are so stupid'
]))
