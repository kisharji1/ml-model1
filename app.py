from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# load model (pipeline saved by train_model.py)
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
    # Print expected feature names if available
    if hasattr(model, 'feature_names_in_'):
        print('Model expects features:', model.feature_names_in_)
    else:
        print('Model feature names not available. Check your training script.')
except Exception as e:
    model = None
    print(f"Model not found or failed to load: {e}")


@app.route("/")
def home():
    return render_template("delivery_form.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run training script first."}), 500

    data = request.get_json()
    # Dynamically build input for model
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
        print('Model expects features:', feature_names)
        try:
            # Build input dict from user data, fallback to 0 if missing
            input_dict = {k: float(data.get(k, 0)) for k in feature_names}
        except Exception as e:
            return jsonify({"error": "Invalid input format: " + str(e)}), 400
        df = pd.DataFrame([input_dict])
    else:
        # Fallback to previous keys
        try:
            region_encoded = float(data.get("region_encoded", 0))
            distance = float(data.get("distance_km", 0))
            weight = float(data.get("package_weight_kg", 0))
            cost = float(data.get("delivery_cost", 0))
            delivery_id = float(data.get("delivery_id", 0))
        except Exception as e:
            return jsonify({"error": "Invalid input format: " + str(e)}), 400
        df = pd.DataFrame([
            {
                "region": region_encoded,
                "distance_km": distance,
                "weight_kg": weight,
                "delivery_cost": cost,
                "delivery_id": delivery_id,
            }
        ])

    try:
        pred = model.predict(df)[0]
        # if model supports predict_proba
        try:
            proba = model.predict_proba(df)
            classes = getattr(model, 'classes_', None)
            if classes is not None:
                # find probability of predicted class
                idx = list(classes).index(pred)
                confidence = float(proba[0][idx])
            else:
                confidence = float(max(proba[0]))
        except Exception:
            confidence = 1.0

        # Map predicted value to delivery mode
        mode_map = {0: "same day", 1: "express", 2: "two days"}
        try:
            predicted_encoded = int(pred)
        except Exception:
            predicted_encoded = pred
        predicted_mode = mode_map.get(predicted_encoded, str(predicted_encoded))
        return jsonify({
            "predicted_mode_encoded": predicted_encoded,
            "predicted_mode": predicted_mode,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": "Prediction failed: " + str(e)}), 500


if __name__ == "__main__":
    # Run on port 5001 to avoid conflicts with other processes on 5000
    app.run(debug=True, port=5001)
