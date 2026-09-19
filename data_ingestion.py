import os
import pandas as pd

class DataIngestion:
    def __init__(self):
        container_input = "/opt/ml/processing/input/Credit_score"
        container_output = "/opt/ml/processing/ingested/Credit_score"

        if os.path.exists(container_input):
            self.input_dir = container_input
            self.output_dir = container_output
        else:
            self.input_dir = "/home/ec2-user/SageMaker/Credit_score"
            self.output_dir = "/home/ec2-user/SageMaker/Credit_score/ingested"

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)

        input_file = os.path.join(self.input_dir, "data_B.csv")
        print(f"Looking for file at: {input_file}")

        if os.path.exists(input_file):
            df = pd.read_csv(input_file)

            output_file = os.path.join(self.output_dir, "data_B.csv")
            df.to_csv(output_file, index=False)

            print(f"✅ Success: Ingested data saved to {output_file}")
        else:
            print(f"❌ Error: {input_file} not found!")
            print(f"Container sees these files in {self.input_dir}: {os.listdir(self.input_dir)}")

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()