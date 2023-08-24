# Dialogue Summarization with Toxicity Detoxification

This repository contains code for a dialogue summarization model with toxicity detoxification. The model utilizes various libraries such as Hugging Face Transformers, TRL (Transformer Reinforcement Learning library), and datasets from Hugging Face's datasets library. This code was used in conjunction with AWS SageMaker for deployment.

## Setup

To get started, follow these steps to set up the necessary dependencies:

\`\`\`bash
%pip install --upgrade pip
%pip install --disable-pip-version-check \
    torch==1.13.1 \
    torchdata==0.5.1 --quiet

%pip install \
    transformers==4.27.2 \
    datasets==2.11.0 --quiet
\`\`\`
Make sure to run these commands in your preferred environment before proceeding.

## Overview

The code is organized into several sections, each serving a specific purpose:

- Importing Libraries: Import necessary libraries including Hugging Face Transformers, datasets, and other custom modules.
- Loading and Preprocessing the Dataset: Defines functions to load and preprocess a dialogue dataset for training and testing the summarization model. The dataset is tokenized and split into train and test sets.
- Model Configuration: Defines the configuration for the various models used in the pipeline, including FLAN-T5, PEFT, PPO, and a sentiment analysis toxicity model.
- Toxicity Analysis: Demonstrates how to use the sentiment analysis model to analyze toxicity in text. It provides examples of how to get toxicity scores and probabilities.
- Evaluation Functions: Includes functions to evaluate toxicity using the toxicity model and generate summaries using the PPO-based summarization model.
- Fine-Tuning for Detoxification: Describes the process of fine-tuning the summarization model using PPO with toxicity detoxification. It demonstrates how to set up the PPOTrainer, generate summaries, compute rewards, and perform PPO optimization steps.
- Comparison of Summaries: Compares the toxicity scores and improvements of generated summaries before and after detoxification.

## How to Use

- Dataset Preparation: Modify the `model_name` and `huggingface_dataset_name` variables to match your desired transformer model and Hugging Face dataset name. Also, adjust the `input_min_text_length` and `input_max_text_length` to filter the dialogues based on their lengths.
- Toxicity Model: Set up the toxicity model by defining its name and loading the corresponding tokenizer.
- Evaluate Toxicity: Use the provided functions to evaluate the toxicity of text samples using the toxicity model.
- Detoxification and Summarization: Fine-tune the summarization model using PPO with the detoxification process. This section demonstrates how to generate summaries, compute rewards, and optimize the model using PPO steps.
- Compare Summaries: Compare the generated summaries before and after detoxification in terms of their toxicity scores and improvements.

## Important Notes

- This code assumes familiarity with Hugging Face Transformers, TRL, and datasets libraries. Ensure you have knowledge of their APIs to understand the code structure and usage.
- The code contains example dataset names, model names, and hyperparameters. You may need to adjust these values according to your specific use case and requirements.
- The provided code is meant for educational purposes and may require further adaptation for production-level deployment.
- This code was used in conjunction with AWS SageMaker for deployment.

## Running on AWS SageMaker

This code was specifically configured to run on AWS SageMaker. Below are the steps to get the model up and running in this environment.

### Prerequisites

- An AWS account
- The AWS CLI installed and configured
- SageMaker Python SDK

### Steps

1. **Open SageMaker:** Navigate to the AWS SageMaker console and create a new Jupyter Notebook instance.
  
2. **Clone Repository:** Once the instance is ready, open Jupyter Notebook and clone this repository into the instance.
  
3. **Install Dependencies:** Open a terminal within Jupyter Notebook or SageMaker Studio and navigate to the repository directory. Run the following command to install the necessary dependencies.
    ```bash
    %pip install -r requirements.txt
    ```

4. **Set Environment Variables:** If necessary, set environment variables within SageMaker for any sensitive data or configurations.
  
5. **Execute Notebook:** Navigate back to the Jupyter Notebook interface, open the notebook containing the code, and execute all cells.

6. **Training:** You can either train the model using SageMaker's built-in distributed training feature or run the training code in a notebook cell.
  
7. **Deployment:** Once the model is trained, you can deploy it using SageMaker's real-time endpoint functionality or batch transform feature.

8. **Monitoring:** AWS SageMaker provides various tools for monitoring the model's performance and usage, such as SageMaker Model Monitor and CloudWatch.

### Important Notes

- Make sure to stop the SageMaker instance when not in use to avoid incurring extra charges.
- Ensure that the instance type selected meets the resource requirements for the training and inference tasks.

