import pandas as pd
import pickle
import os

# Pfad zum Modell dynamisch setzen
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "..", "model", "diagnose_model.pkl")

# Prüfen, ob Modell existiert
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

# Modell + Feature-Namen laden
with open(model_path, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    feature_names = data["features"]

# Beispiel-Eingabe als Dictionary
# Nur die Features angeben, die 1 oder 0 gesetzt werden sollen
input_dict = {
    "abdominal_pain": 1,
    "acidity": 0,
    "abnormal_menstruation": 0,
    "acute_liver_failure": 1,
    # ... weitere Features nach Bedarf
}

# Alle Features in der richtigen Reihenfolge füllen, fehlende auf 0 setzen
example_values = [input_dict.get(f, 0) for f in feature_names]

# DataFrame erstellen
example_df = pd.DataFrame([example_values], columns=feature_names)

# Vorhersage
prediction = model.predict(example_df)
print("Vorhersage:", prediction)






