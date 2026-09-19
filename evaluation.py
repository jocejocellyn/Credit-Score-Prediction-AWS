import os
import json
import joblib 
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tarfile
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "xgboost==3.0.5"
])

class CreditScoreEvaluator:
    def __init__(self):
        self.model_files = {
            "RandomForest": "RandomForest.joblib",
            "ExtraTrees": "ExtraTrees.joblib",
            "XGBoost": "XGBoost.joblib"
        }

    def evaluate(self, model_dir, x_test, y_test):
        results = []
        best_model = None
        best_model_name = None
        best_f1 = -1

        for name, filename in self.model_files.items():
            model_path = os.path.join(model_dir, filename)
            if not os.path.exists(model_path):
                print(f"⚠️ Missing model: {model_path}")
                continue

            model = joblib.load(model_path)
            predictions = model.predict(x_test)

            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, average="macro")
            rec = recall_score(y_test, predictions, average="macro")
            f1 = f1_score(y_test, predictions, average="macro")

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1
            })

            print(
                f"{name} | "
                f"Acc={acc:.4f} | "
                f"Prec={prec:.4f} | "
                f"Recall={rec:.4f} | "
                f"F1={f1:.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name

        if best_model is None:
            raise Exception("No valid models found")

        return results, best_model, best_model_name
    
    def save_best_model(self, model, model_name, model_dir):
        best_model_path = os.path.join(model_dir, "best_model.joblib")
        joblib.dump(model, best_model_path)

        print(f"✅ Best Model: {model_name}")
        print(f"✅ Saved: {best_model_path}")

    def save_report(self, results, best_model_name, output_dir):
        best_result = None

        for r in results:
            if r["Model"] == best_model_name:
                best_result = r
                break

        if best_result is None:
            raise Exception("Best model result not found")

        report_dict = {
            "multiclass_classification_metrics": {
                "accuracy": {"value": float(best_result["Accuracy"])},
                "precision": {"value": float(best_result["Precision"])},
                "recall": {"value": float(best_result["Recall"])},
                "f1_score": {"value": float(best_result["F1"])}
            }
        }

        output_file = os.path.join(output_dir, "evaluation.json")
        with open(output_file, "w") as f:
            json.dump(report_dict, f)
        print(f"✅ Evaluation report saved: {output_file}")

    def run(self):
        if os.environ.get("SM_CHANNEL_TEST") or os.path.exists("/opt/ml/processing"):
            model_dir = "/opt/ml/processing/model"
            test_path = "/opt/ml/processing/test/test.csv"
            output_dir = "/opt/ml/processing/evaluation"
        else:
            model_dir = "/home/ec2-user/SageMaker/Credit_score/model"
            test_path = "/home/ec2-user/SageMaker/Credit_score/test/test.csv"
            output_dir = "/home/ec2-user/SageMaker/Credit_score/eval"

        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(model_dir):
            print(f"❌ Missing model directory: {model_dir}")
            return

        print("Model directory:", model_dir)
        print("Files in model directory:")
        print(os.listdir(model_dir))

        tar_path = os.path.join(model_dir, "model.tar.gz")

        if os.path.exists(tar_path):
            print("Extracting model.tar.gz...")
            with tarfile.open(tar_path) as tar:
                tar.extractall(model_dir)

            print("After extract:")
            print(os.listdir(model_dir))

        if not os.path.exists(test_path):
            print(f"❌ Missing test file: {test_path}")
            return

        test_df = pd.read_csv(test_path)
        x_test = test_df.drop("Credit_Score", axis=1)
        y_test = test_df["Credit_Score"]

        results, best_model, best_model_name = self.evaluate(model_dir, x_test, y_test)
        self.save_best_model(best_model, best_model_name, model_dir)
        self.save_report(results, best_model_name, output_dir)

if __name__ == "__main__":
    CreditScoreEvaluator().run()