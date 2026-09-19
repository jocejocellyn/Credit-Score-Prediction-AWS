import os
import joblib
import pandas as pd
import numpy as np
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-r",
    "requirements.txt"
])

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

class ModelPreprocessor:
    def handle_invalid_values(self, df, rules):
        df = df.copy()

        for col, (lower, upper) in rules.items():
            if col in df.columns:
                df.loc[(df[col] < lower) | (df[col] > upper),col] = np.nan
        return df

    def get_transformer(self, x_train):
        num_features = (x_train.select_dtypes(include=['int64', 'float64']).columns.tolist())
        ordinal_features = ['Credit_Mix', 'Payment_of_Min_Amount']
        nominal_features = ['Month', 'Occupation', 'Payment_Behaviour']

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        nominal_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        ordinal_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=[
                ['Bad', 'Standard', 'Good'],
                ['No', 'Yes']
            ]))
        ])

        return ColumnTransformer([
            ('num', numeric_transformer, num_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_features)
        ])

class CreditScoreTrainer:
    def __init__(self):
        self.model_preprocessor = ModelPreprocessor()

    def run(self):
        train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/home/ec2-user/SageMaker/Credit_score/train")
        model_dir = os.environ.get("SM_MODEL_DIR", "/home/ec2-user/SageMaker/Credit_score/model")

        os.makedirs(model_dir, exist_ok=True)
        train_file = os.path.join(train_dir, "train.csv")

        if os.path.exists(train_file):
            df = pd.read_csv(train_file)
            x_train = df.drop("Credit_Score", axis=1)
            y_train = df["Credit_Score"]

            rules = {
                "Age": (18, 100),
                "Num_Bank_Accounts": (0, 50),
                "Num_Credit_Card": (0, 50),
                "Interest_Rate": (0, 100),
                "Num_of_Loan": (0, 50),
                "Delay_from_due_date": (0, 100),
                "Num_of_Delayed_Payment": (0, 100),
                "Changed_Credit_Limit": (0, 50),
                "Num_Credit_Inquiries": (0, 100),
                "Monthly_Balance": (-1000000, 1000000)
            }

            x_train = self.model_preprocessor.handle_invalid_values(x_train, rules)
            transformer = self.model_preprocessor.get_transformer(x_train)

            models = {
                "RandomForest": Pipeline([
                    ("preprocessing", transformer),
                    ("classifier", RandomForestClassifier(
                        n_estimators=300,
                        max_depth=40,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    ))
                ]),

                "ExtraTrees": Pipeline([
                    ("preprocessing", transformer),
                    ("classifier", ExtraTreesClassifier(
                        n_estimators=250,
                        max_depth=18,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    ))
                ]),

                "XGBoost": Pipeline([
                    ("preprocessing", transformer),
                    ("classifier", XGBClassifier(
                        objective="multi:softprob",
                        n_estimators=700,
                        max_depth=8,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=1.0,
                        random_state=42,
                        eval_metric="mlogloss"
                    ))
                ])
            }

            for name, model in models.items():
                print(f"Training {name}...")
                model.fit(x_train, y_train)

                output_file = os.path.join(model_dir, f"{name}.joblib")
                joblib.dump(model, output_file)
                print(f"✅ Saved: {output_file}")

            print("✅ All models trained successfully.")

        else:
            print(f"❌ Error: {train_file} not found!")
            print(f"Files found: {os.listdir(train_dir)}")

if __name__ == "__main__":
    CreditScoreTrainer().run()