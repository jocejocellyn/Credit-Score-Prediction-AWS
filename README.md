# Credit-Score-Prediction-AWS
An AWS-based extension of the Credit Score Prediction project, developed to adapt the machine learning workflow to a cloud-based environment using **Amazon SageMaker**. The original machine learning workflow was first developed locally. This repository focuses on the AWS implementation, including data processing, model training, evaluation, model selection, and deployment.  
> **AWS Status:** The original AWS environment and deployed resources are no longer active. This repository is maintained to document the implementation and workflow used for the original deployment.

## AWS Workflow
Data Processing → Model Training → Model Evaluation → Model Selection → Deployment

## AWS Pipeline
The project uses Amazon SageMaker to organize the machine learning workflow into separate processing and training steps.  
The pipeline includes:
1. Data processing
2. Model training
3. Model evaluation
4. Model selection
5. Model deployment

The implementation uses SageMaker processing and training environments to run the machine learning workflow independently from the local environment.

## Models
The AWS implementation evaluates the same main machine learning approaches used in the local project:
* Random Forest
* Extra Trees
* XGBoost


The trained models are evaluated using:  
* Accuracy
* Precision
* Recall
* F1-Score

The final model is selected based on the evaluation results.

## Model Evaluation
The evaluation step loads the trained model artifact and test data, generates predictions, calculates evaluation metrics, and stores the results in an `evaluation.json` file. The selected model is then saved as a model artifact for deployment.

## SageMaker Pipeline
The SageMaker pipeline was orchestrated programmatically and includes separate stages for processing, training, evaluation, and model selection. The implementation also uses SageMaker's pipeline session to define and execute the workflow.

## Deployment
The trained model was originally deployed using AWS services as part of the cloud-based machine learning workflow. The original AWS deployment is no longer active. Therefore, the endpoint and deployed AWS application cannot currently be accessed. The source code remains available in this repository for documentation and demonstration of the implementation.
