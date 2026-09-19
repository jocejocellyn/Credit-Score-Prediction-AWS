import os
import sagemaker

from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn

class CreditScorePipeline:
    def __init__(self):
        self.local_pipeline_session = LocalPipelineSession()

        try:
            self.role = sagemaker.get_execution_role()
        except:
            self.role = "LabRole"

        self.instance_type = "local"
        self.base_path = "/home/ec2-user/SageMaker/Credit_score"

        os.makedirs(f"{self.base_path}/ingested", exist_ok=True)
        os.makedirs(f"{self.base_path}/train", exist_ok=True)
        os.makedirs(f"{self.base_path}/test", exist_ok=True)
        os.makedirs(f"{self.base_path}/eval", exist_ok=True)
        os.makedirs(f"{self.base_path}/model", exist_ok=True)

        self.local_output_path = f"file:///home/ec2-user/SageMaker/Credit_score"
        self.local_ingested_data = f"{self.local_output_path}/ingested"
        self.local_train_data = f"{self.local_output_path}/train"
        self.local_test_data = f"{self.local_output_path}/test"
        self.local_eval_data = f"{self.local_output_path}/eval"

        self.sklearn_processor = SKLearnProcessor(
            framework_version="1.4-2",
            role=self.role,
            instance_type=self.instance_type,
            instance_count=1,
            sagemaker_session=self.local_pipeline_session
        )

        self.sklearn_estimator = SKLearn(
            entry_point="train.py",
            source_dir=".",
            dependencies=["requirements.txt"],
            role=self.role,
            instance_type=self.instance_type,
            framework_version="1.4-2",
            sagemaker_session=self.local_pipeline_session
        )

    def create_pipeline(self):
        step_ingest = ProcessingStep(
            name="CreditScoreIngest",
            processor=self.sklearn_processor,
            inputs=[
                ProcessingInput(
                    source=self.base_path,
                    destination="/opt/ml/processing/input/Credit_score"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="ingested_data",
                    source="/opt/ml/processing/ingested/Credit_score",
                    destination=self.local_ingested_data
                )
            ],
            code="data_ingestion.py"
        )

        step_preprocess = ProcessingStep(
            name="CreditScorePreprocess",
            processor=self.sklearn_processor,
            inputs=[
                ProcessingInput(
                    source=step_ingest.properties.ProcessingOutputConfig.Outputs["ingested_data"].S3Output.S3Uri,
                    destination="/opt/ml/processing/ingested"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/train",
                    destination=self.local_train_data
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/test",
                    destination=self.local_test_data
                )
            ],
            code="preprocessing.py"
        )

        step_train = TrainingStep(
            name="CreditScoreTrain",
            estimator=self.sklearn_estimator,
            inputs={"train": step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri}
        )

        evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

        step_eval = ProcessingStep(
            name="CreditScoreEval",
            processor=self.sklearn_processor,
            inputs=[
                ProcessingInput(
                    source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=self.local_eval_data
                )
            ],
            code="evaluation.py",
            property_files=[evaluation_report]
        )

        return Pipeline(
            name="CreditScore-Local-Workflow",
            steps=[
                step_ingest,
                step_preprocess,
                step_train,
                step_eval
            ],
            sagemaker_session=self.local_pipeline_session
        )

    def run(self):
        pipeline = self.create_pipeline()
        pipeline.upsert(role_arn=self.role)
        pipeline.start()

if __name__ == "__main__":
    CreditScorePipeline().run()