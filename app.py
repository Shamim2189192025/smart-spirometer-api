@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("========== REQUEST RECEIVED ==========")

        data = request.get_json(force=True)
        print("JSON:", data)

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

        print(features)

        prediction = model.predict(features)[0]
        print("Prediction OK")

        probability = model.predict_proba(features)[0]
        print("Probability OK")

        result = "Obstructive" if prediction == 1 else "Normal"

        return jsonify({
            "Prediction": result,
            "Probability_Normal": float(probability[0]),
            "Probability_Obstructive": float(probability[1])
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 400