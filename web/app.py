from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Modellpfad
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "..", "model", "diagnose_model.pkl")

# Modell laden
with open(model_path, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    feature_names = data["features"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None

    if request.method == "POST":
        # Eingaben aus Formular holen
        input_dict = {key: int(request.form.get(key, 0)) for key in feature_names}

        # Alle Features in der richtigen Reihenfolge füllen
        example_values = [input_dict.get(f, 0) for f in feature_names]
        example_df = pd.DataFrame([example_values], columns=feature_names)

        # Vorhersage
        prediction = model.predict(example_df)[0]

        # Wahrscheinlichkeiten
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(example_df)[0]
            probabilities = list(zip(model.classes_, probs))

    return render_template(
        "form.html",
        features=feature_names,
        prediction=prediction,
        probabilities=probabilities
    )

if __name__ == "__main__":
    app.run(debug=True)




