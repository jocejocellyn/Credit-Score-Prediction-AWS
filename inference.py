import json
import os
import joblib
import pandas as pd

class CreditScoreInference:
    def __init__(self):
        self.content_type = "application/json"
        self.class_names = ["Poor", "Standard", "Good"]
    
    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, "best_model.joblib")
        return joblib.load(model_path)

    def parse_input(self, request_body, request_content_type):
        if request_content_type != self.content_type:
            raise ValueError(f"Unsupported content type: {request_content_type}")
        payload = json.loads(request_body)

        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        raise ValueError("Input must be a JSON object")

    def predict(self, input_data, model):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        return {
            "prediction": int(prediction),
            "label": self.class_names[int(prediction)],
            "probabilities": {
                "Poor": float(probabilities[0]),
                "Standard": float(probabilities[1]),
                "Good": float(probabilities[2])
            }
        }

    def format_output(self, prediction, accept):
        if accept == self.content_type:
            return json.dumps(prediction), self.content_type
        raise ValueError(f"Unsupported accept type: {accept}")

inference = CreditScoreInference()

def model_fn(model_dir):
    return inference.load_model(model_dir)

def input_fn(request_body, request_content_type):
    return inference.parse_input(request_body, request_content_type)

def predict_fn(input_data, model):
    return inference.predict(input_data, model)

def output_fn(prediction, accept):
    return inference.format_output(prediction, accept)