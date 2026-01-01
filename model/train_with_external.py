import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# — Pfade zu deinen Dateien —
# Dein Testing-Datensatz (oder vorhandener eigener Datensatz)
own_path = r"C:\01_Programme\Projekt_Nele\data\Testing.csv"
# Externer Datensatz, z. B. der Kaggle-Disease/Symptoms-Datensatz
external_path = r"C:\01_Programme\Projekt_Nele\data\training.csv"

print("📂 Lade eigenen Datensatz:", own_path)
df_own = pd.read_csv(own_path)
print("✅ Eigener Datensatz geladen, Form:", df_own.shape)

print("📂 Lade externen Datensatz:", external_path)
df_ext = pd.read_csv(external_path)
print("✅ Externer Datensatz geladen, Form:", df_ext.shape)

# — Spaltennamen harmonisieren (Kleinbuchstaben, Unterstriche) —
df_own.columns = [c.strip().lower().replace(" ", "_") for c in df_own.columns]
df_ext.columns = [c.strip().lower().replace(" ", "_") for c in df_ext.columns]

print("ℹ️ Eigene Spaltenbeispiele:", df_own.columns[:10])
print("ℹ️ Externe Spaltenbeispiele:", df_ext.columns[:10])

# — Zielspalte (Diagnose) — hier: prognosis (in deinem Testing.csv) —
target = "prognosis"
if target not in df_own.columns:
    raise ValueError(f"Zielspalte '{target}' nicht in dem eigenen Datensatz vorhanden.")
if target not in df_ext.columns:
    raise ValueError(f"Zielspalte '{target}' nicht in dem externen Datensatz vorhanden.")

# — Gemeinsam genutzte Merkmale (Symptome) bestimmen —
features_common = [c for c in df_own.columns if c != target and c in df_ext.columns]

print("🔍 Gemeinsame Features (Symptome):", len(features_common), "Spalten")
print(features_common[:20])

# Subsets mit gemeinsamen Merkmalen + Ziel
X_own = df_own[features_common]
y_own = df_own[target]

X_ext = df_ext[features_common]
y_ext = df_ext[target]

print("✅ Subsets erstellt:", X_own.shape, X_ext.shape)

# — Datasets kombinieren —
X_comb = pd.concat([X_own, X_ext], ignore_index=True)
y_comb = pd.concat([y_own, y_ext], ignore_index=True)

print("🔗 Kombinierter Datensatz:", X_comb.shape, y_comb.shape)
print("📊 Diagnoseverteilungen (kombiniert):")
print(y_comb.value_counts().head(10))

# — Split in Trainings- und Testdaten —
X_train, X_test, y_train, y_test = train_test_split(
    X_comb, y_comb, test_size=0.2, random_state=42, stratify=y_comb
)

print("🔧 Train/Test Split:", X_train.shape, X_test.shape)

# — Modell trainieren —
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# — Auf Testdaten prüfen →
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Genauigkeit auf Testdaten: {acc:.4f}")
print(classification_report(y_test, y_pred))

# — Modell + Feature-Namen speichern —
model_data = {"model": model, "features": features_common}
model_dir = r"C:\01_Programme\Projekt_Nele\model"
os.makedirs(model_dir, exist_ok=True)
model_file = os.path.join(model_dir, "diagnose_model.pkl")
with open(model_file, "wb") as f:
    pickle.dump(model_data, f)

print("💾 Modell gespeichert unter:", model_file)
