
# Promise Evaluation Task - SemEval 2025

This repository contains the implementation of a model for the **Promise Verification Task** in **SemEval-2025** using state-of-the-art pre-trained language models such as **BERT**, **DeBERTa**, and **BART**. The task aims to verify the promises made in statements, classify the evidence presented, assess evidence quality, and predict the verification timeline. The approach is based on fine-tuning pre-trained transformers for multi-task classification.

## Project Overview

This project implements a **multi-label text classification** pipeline to evaluate the promises and supporting evidence within a given text. The goal is to assess:

1. **Promise Classification**: Identifying if a statement involves a promise.
2. **Evidence Classification**: Determining if a statement provides supporting evidence for the promise.
3. **Evidence Quality**: Assessing the quality and clarity of the evidence presented.
4. **Verification Timeline**: Estimating when the promise will be verified.

The model leverages the following transformer-based architectures:

- **BERT**
- **DeBERTa**
- **BART-CNN**

These models are fine-tuned for the specific subtasks, providing a robust framework for promise verification in natural language processing.

## Methodology

### 1. **Dataset Preparation**
   - **Data Loading and Preprocessing**: 
     - The dataset consists of text from English mainly.
     - Each text sample is preprocessed by tokenizing the text and encoding it into the format required by each pre-trained model.
     - Label encoding is performed for each subtask (Promise Classification, Evidence Classification, Evidence Quality, and Verification Timeline).
     - Data is split into training, validation, and test sets to ensure robust evaluation.

### 2. **Model Architecture**
   - **Pre-trained Transformer Models**: 
     - We employ **BERT**, **DeBERTa**, and **BART** for fine-tuning on the task.
     - For each subtask, a corresponding output layer is used to predict whether a certain class or label applies to the input text.
   - **Multi-Task Learning**: 
     - The model is trained for multiple subtasks simultaneously, each focusing on one aspect of promise verification.
     - Cross-entropy loss is used for classification tasks, while multi-label classification loss is employed for tasks involving multiple labels. Contrastive Loss was also used in certain tasks to allow the model to better learn.

### 3. **Training**
   - **Fine-tuning on Subtasks**: 
     - The models are fine-tuned on each subtask (Promise Classification, Evidence Classification, Evidence Quality, and Verification Timeline).
     - The training process involves optimizing the models using the Adam optimizer and learning rate scheduling for efficient convergence.

### 4. **Evaluation**
   - **Metrics**: 
     - **Accuracy**, **F1-Score**, **Precision**, and **Recall** are computed to evaluate model performance on each subtask.
     - The results of the baseline and final models are compared for each subtask to assess improvements.

## Requirements

To run this code, you need the following dependencies:

- **Python 3.7+**
- **PyTorch**
- **Transformers Library** (by Hugging Face)
- **scikit-learn**
- **pandas**
- **numpy**
- **matplotlib** (for plotting graphs)
