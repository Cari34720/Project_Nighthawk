import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Pfad zur CSV-Datei dynamisch setzen (unabhängig vom aktuellen Arbeitsverzeichnis)
base_dir = os.path.dirname(__file__)  # Verzeichnis, in dem train.py liegt
csv_path = os.path.join(base_dir, "..", "data", "Testing.csv")

# Prüfen, ob die Datei existiert
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_path}")

# Daten laden
df = pd.read_csv(csv_path)

# Features und Ziel
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Modell trainieren
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Feature-Namen speichern
feature_names = X.columns.tolist()

# Pfad zum Speichern des Modells
model_path = os.path.join(base_dir, "..", "model", "diagnose_model.pkl")

# Modell + Features speichern
with open(model_path, "wb") as f:
    pickle.dump({"model": model, "features": feature_names}, f)

print(f"Modell erfolgreich gespeichert unter: {model_path}")




