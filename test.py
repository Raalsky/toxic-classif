from toxic.models import BertToxicClassifier as ToxicClassifier

cls = ToxicClassifier(load_weights=True, tta_fold=8)

print(cls.predict([
    """
    - Tworzenie i utrzymywanie standardowych artefaktów zarządzania projektami, takich jak plan projektu, rejestr ryzyka i problemów, dziennik zmian, moduł do śledzenia budżetu, raporty z postępów itp.
    - Zarządzanie wymaganiami wprowadzonymi do technicznej platformy technicznej backlog, tj. API systemu
    """
]))
