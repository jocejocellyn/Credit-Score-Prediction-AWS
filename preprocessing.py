import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def drop_columns(self, df):
        return df.drop(columns=['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN'], errors='ignore')

    def clean_numeric_columns(self, df):
        numeric_cols = [
            'Age',
            'Annual_Income',
            'Num_of_Loan',
            'Num_of_Delayed_Payment',
            'Changed_Credit_Limit',
            'Outstanding_Debt',
            'Amount_invested_monthly',
            'Monthly_Balance'
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('_', ''), errors='coerce')
        return df

    def clean_missing_values(self, df):
        df.replace({
            "_": np.nan,
            "_______": np.nan,
            "!@9#%8": np.nan
        }, inplace=True)

        df['Payment_of_Min_Amount'] = (df['Payment_of_Min_Amount'].replace({'NM': 'No'}))
        return df

    def process_type_of_loan(self, df):
        def clean_loan(x):
            if pd.isna(x):
                return np.nan

            x = x.replace(', and ', ', ')
            loans = [i.strip() for i in x.split(',')]
            loans = list(dict.fromkeys(loans))
            return ', '.join(loans)

        df['Type_of_Loan'] = df['Type_of_Loan'].apply(clean_loan)
        df['Type_of_Loan'] = df['Type_of_Loan'].fillna('Unknown')
        loan_dummies = df['Type_of_Loan'].str.get_dummies(sep=', ')
        df = pd.concat([df, loan_dummies], axis=1)
        df.drop(columns=['Type_of_Loan'], inplace=True)
        
        return df

    def process_credit_history(self, df):
        def convert_history(x):
            try:
                year = int(x.split(' ')[0])
                month = int(x.split(' ')[3])
                return year * 12 + month
            except:
                return np.nan

        df['Credit_History_Age'] = (df['Credit_History_Age'].apply(convert_history))
        return df

    def encode_target(self, df):
        df['Credit_Score'] = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})
        return df

    def transform(self, df):
        df = df.copy()

        df = self.drop_columns(df)
        df = self.clean_numeric_columns(df)
        df = self.clean_missing_values(df)
        df = self.process_type_of_loan(df)
        df = self.process_credit_history(df)
        df = self.encode_target(df)

        return df

    def run(self):
        if os.environ.get("SM_CHANNEL_INPUT") or os.path.exists("/opt/ml/processing"):
            input_dir = "/opt/ml/processing/ingested"
            out_train = "/opt/ml/processing/train"
            out_test = "/opt/ml/processing/test"

        else:
            input_dir = "/home/ec2-user/SageMaker/Credit_score/ingested"
            out_train = "/home/ec2-user/SageMaker/Credit_score/train"
            out_test = "/home/ec2-user/SageMaker/Credit_score/test"

        os.makedirs(out_train, exist_ok=True)
        os.makedirs(out_test, exist_ok=True)

        input_file = os.path.join(input_dir, "data_B.csv")

        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            df = self.transform(df)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Credit_Score"])

            train_df.to_csv(os.path.join(out_train, "train.csv"), index=False)
            test_df.to_csv(os.path.join(out_test, "test.csv"), index=False)
            print("✅ Data preprocessing complete.")
        else:
            print(f"❌ Error: {input_file} not found!")

if __name__ == "__main__":
    DataPreprocessor().run()