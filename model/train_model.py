import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# --- Pfad anpassen ---
data_path = r"C:\01_Programme\Projekt_Nele\data\Testing.csv"

print(f"📂 Lade Daten aus: {data_path}")
df = pd.read_csv(data_path)

# --- Spalten prüfen ---
print("✅ Spalten geladen:", df.columns.tolist()[:10], "...")

# --- Features & Ziel ---
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# --- Split in Trainings- und Testdaten ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Modell trainieren ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Genauigkeit prüfen ---
accuracy = model.score(X_test, y_test)
print(f"🎯 Modell-Genauigkeit: {accuracy:.2%}")

# --- Modell speichern ---
model_data = {
    "model": model,
    "features": list(X.columns)
}

model_path = os.path.join(os.path.dirname(__file__), "diagnose_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"✅ Modell gespeichert unter: {model_path}")
