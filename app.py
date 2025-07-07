from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the retrained 6‑feature model & scaler
model = joblib.load("xgb_top6_model.pkl")
scaler = joblib.load("scaler_top6.pkl")

# Exact order must match index.html loop!
top_features = [
    "Curricular units 2nd sem (approved)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (grade)",
    "Tuition fees up to date",
    "Scholarship holder"
]

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    proba = None
    error = None

    if request.method == "POST":
        try:
            # collect raw inputs
            vals = []
            for feat in top_features:
                vals.append(float(request.form[feat]))
            X = np.array([vals])
            # scale
            X_scaled = scaler.transform(X)
            # predict
            prediction = int(model.predict(X_scaled)[0])
            proba = float(model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        top_features=top_features,
        prediction=prediction,
        proba=proba,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
