# Dialogue Summarization with Toxicity Detoxification

This repository contains code for a dialogue summarization model with toxicity detoxification. The model utilizes various libraries such as Hugging Face Transformers, TRL (Transformer Reinforcement Learning library), and datasets from Hugging Face's datasets library. This code was used in conjunction with AWS SageMaker for deployment.

## Setup

First, make sure you have AWS CLI configured and SageMaker Python SDK installed. If not, you can install the SDK using:

`bash
pip install sagemaker'

### Running Locally

To test the code locally, you can use:

\`\`\`bash
python your_script.py --arg1 value1 --arg2 value2
\`\`\`

Replace \`your_script.py\` with the name of your Python script and set the arguments (\`arg1\`, \`arg2\`, etc.) as needed.

## Deploying on AWS SageMaker using Script Mode

### Prerequisites

- An AWS account
- AWS CLI configured
- SageMaker Python SDK installed

### Steps

1. **Upload Data to S3:** If your dataset is not already in S3, upload it there.

    \`\`\`bash
    aws s3 cp your_dataset s3://your_bucket/your_dataset
    \`\`\`

2. **Create a SageMaker Session:**

    \`\`\`python
    import sagemaker

    sagemaker_session = sagemaker.Session()
    \`\`\`

3. **Upload Your Script:**

    Upload your Python script (\`your_script.py\`) to your S3 bucket.

    \`\`\`bash
    aws s3 cp your_script.py s3://your_bucket/your_script.py
    \`\`\`

4. **Set Script Mode Configuration:**

    \`\`\`python
    from sagemaker.pytorch import PyTorch

    estimator = PyTorch(entry_point='your_script.py',
                        role='your_role',
                        instance_count=1,
                        instance_type='ml.m5.large',
                        framework_version='1.8.1',
                        py_version='py36',
                        hyperparameters={
                            'arg1': 'value1',
                            'arg2': 'value2'
                        })
    \`\`\`

    Replace \`your_role\` with your SageMaker role ARN, and set any hyperparameters your script needs.

5. **Train Model:**

    \`\`\`python
    estimator.fit({'training': 's3://your_bucket/your_dataset'})
    \`\`\`

6. **Deploy Model:**

    \`\`\`python
    predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.large')
    \`\`\`

### Cleanup

After you're done, make sure to delete the endpoint to avoid incurring additional charges:

\`\`\`python
predictor.delete_endpoint()
\`\`\`

## Important Notes

- Make sure the SageMaker role has the necessary permissions.
- You may need to adjust instance types and counts based on your specific needs.

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

