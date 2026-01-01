from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from datetime import datetime
import os
import json
import traceback
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from geopy.geocoders import Nominatim

app = Flask(__name__)

# --- Modellpfad ---
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "..", "model", "diagnose_model.pkl")

# --- Modell laden ---
with open(model_path, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    feature_names = data["features"]

# --- Google Sheet Funktion ---
def get_gsheet():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]

    json_path = os.path.join(os.path.dirname(__file__), "service_account.json")

    if not os.path.exists(json_path):
        print("❌ service_account.json nicht gefunden:", json_path)
        return None

    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(json_path, scope)
        client = gspread.authorize(creds)
    except Exception as e:
        print("❌ Fehler bei der Authentifizierung:")
        traceback.print_exc()
        return None

    sheet_id = "12cpUeHKpbVEOe_VODjUquNXo3SIbaR3t4mIsnGT_JpQ"
    try:
        sheet = client.open_by_key(sheet_id).sheet1
        return sheet
    except Exception as e:
        print("❌ Fehler beim Öffnen des Sheets:")
        traceback.print_exc()
        return None

# --- In Google Sheet speichern ---
def save_to_gsheet(input_data, prediction, probabilities, location=None):
    sheet = get_gsheet()
    if sheet is None:
        print("⚠️ Keine Verbindung zum Google Sheet – nichts gespeichert.")
        return

    row = [
        datetime.now().isoformat(),
        prediction,
        json.dumps(probabilities),
        json.dumps(input_data),
        json.dumps(location)
    ]

    try:
        sheet.append_row(row, value_input_option="USER_ENTERED")
        print("✅ Vorhersage erfolgreich ins Google Sheet geschrieben.")
    except Exception as e:
        print("❌ Fehler beim Schreiben ins Google Sheet:")
        traceback.print_exc()

# --- Daten aus Google Sheet laden ---
def load_gsheet_data():
    sheet = get_gsheet()
    if sheet is None:
        return []
    try:
        rows = sheet.get_all_values()
        headers = rows[0]
        data = rows[1:]
        records = []
        for row in data:
            try:
                location_data = json.loads(row[4]) if len(row) > 4 and row[4] else None
                if location_data and "lat" in location_data and "lon" in location_data:
                    records.append({
                        "timestamp": row[0],
                        "prediction": row[1],
                        "probabilities": json.loads(row[2]),
                        "input": json.loads(row[3]),
                        "location": location_data
                    })
            except Exception:
                continue
        return records
    except Exception as e:
        print("❌ Fehler beim Lesen des Sheets:")
        traceback.print_exc()
        return []

# --- Flask Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("🧩 POST-Request empfangen")

        try:
            data = request.get_json()
            symptoms = data.get("symptoms", [])
            location_name = data.get("location", "").strip()

            # --- DEBUG / Normalisierung / Matching ---
            def normalize_name(s):
                import re
                s2 = s.strip().lower()
                s2 = s2.replace("(", "").replace(")", "")
                s2 = s2.replace("/", "_").replace("-", "_")
                s2 = s2.replace(".", "").replace(",", "")
                s2 = re.sub(r"\s+", "_", s2)
                s2 = s2.replace("__", "_")
                return s2

            symptoms_normalized = [normalize_name(s) for s in symptoms]
            feature_names_norm = [normalize_name(f) for f in feature_names]

            matched = []
            unmatched = []
            symptom_set = set(symptoms_normalized)
            for fn, fn_norm in zip(feature_names, feature_names_norm):
                if fn_norm in symptom_set:
                    matched.append(fn)
            for s in symptoms_normalized:
                if s not in feature_names_norm:
                    unmatched.append(s)

            print("=== DEBUG INPUT ===")
            print("raw symptoms:", symptoms)
            print("normalized symptoms:", symptoms_normalized)
            print("matched features (first 20):", matched[:20])
            print("unmatched symptoms:", unmatched[:20])

            # --- Input dict für Modell auf Basis matched ---
            input_dict = {f: 1 if f in matched else 0 for f in feature_names}
            example_df = pd.DataFrame([input_dict], columns=feature_names)

            print("example_df head:")
            print(example_df.head().T.head(40))
            print("dtypes:", example_df.dtypes[:30])
            print("any nulls:", example_df.isnull().any().any())

            # --- Modellvorhersage ---
            prediction = model.predict(example_df)[0]
            probs = model.predict_proba(example_df)[0].tolist() if hasattr(model, "predict_proba") else []

            probabilities = [{"name": cls, "prob": prob} for cls, prob in zip(model.classes_, probs)]
            probabilities = sorted(probabilities, key=lambda x: x["prob"], reverse=True)[:3]

            # --- Geocoding (Location -> Koordinaten) ---
            latitude = longitude = None
            if location_name:
                try:
                    geolocator = Nominatim(user_agent="project_nighthawk")
                    loc = geolocator.geocode(location_name)
                    if loc:
                        latitude, longitude = loc.latitude, loc.longitude
                        print(f"✅ Standort gefunden: {location_name} -> {latitude}, {longitude}")
                    else:
                        print(f"⚠️ Standort nicht gefunden: {location_name}")
                except Exception:
                    traceback.print_exc()

            # --- In Google Sheet speichern ---
            save_to_gsheet(input_dict, prediction, probabilities,
                           {"name": location_name, "lat": latitude, "lon": longitude})

            return jsonify({
                "prediction": prediction,
                "probabilities": probabilities,
                "location": {"name": location_name, "lat": latitude, "lon": longitude}
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # GET → Seite rendern + historische Punkte laden
    records = load_gsheet_data()
    return render_template("form.html", records=json.dumps(records))

if __name__ == "__main__":
    app.run(debug=True)























