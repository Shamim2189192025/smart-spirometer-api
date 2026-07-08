from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained ML model
model = joblib.load("xgboost_model.pkl")


@app.route("/")
def home():
    return "Smart Spirometer AI Server Running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = pd.DataFrame([{
            "Age": data["Age"],
            "Sex": data["Sex"],
            "Height": data["Height"],
            "Weight": data["Weight"],
            "Baseline_FVC_L": data["Baseline_FVC_L"],
            "Baseline_FEV1_L": data["Baseline_FEV1_L"],
            "Baseline_FEV1_FVC_Ratio": data["Baseline_FEV1_FVC_Ratio"],
            "Baseline_PEF_Ls": data["Baseline_PEF_Ls"]
        }])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        if prediction == 1:
            result = "Obstructive"
        else:
            result = "Normal"

        return jsonify({
            "Prediction": result,
            "Probability_Normal": round(float(probability[0]), 6),
            "Probability_Obstructive": round(float(probability[1]), 6)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)