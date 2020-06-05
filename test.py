from toxic.models import BertToxicClassifier as ToxicClassifier

cls = ToxicClassifier(load_weights=True, tta_fold=0)

text = "It’s mayors and county executives that appoint most police chiefs and negotiate collective bargaining agreements with police unions. It’s district attorneys and state’s attorneys that decide whether or not to investigate and ultimately charge those involved in police misconduct. Those are all elected positions. In some places, police review boards with the power to monitor police conduct are elected as well. Unfortunately, voter turnout in these local races is usually pitifully low, especially among young people – which makes no sense given the direct impact these offices have on social justice issues, not to mention the fact that who wins and who loses those seats is often determined by just a few thousand, or even a few hundred, votes."

print(text)

print(cls.predict([text]))

cls.save()
