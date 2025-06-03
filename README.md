# MAPIE and Conformal Predictions with a LLM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FaustinPulveric/mapie_llm/blob/main/MAPIE_for_cosmosqa.ipynb)

## Overview

This notebook demonstrates how to use MAPIE for conformal predictions with a Large Language Model (LLM). The goal is to evaluate prediction sets for a multiple-choice question-answering task using conformal prediction techniques. This notebook is based on the work presented in [Benchmarking LLMs via Uncertainty Quantification](https://arxiv.org/abs/2401.12794). Parts of the code come from this [Github repo](https://github.com/smartyfh/LLM-Uncertainty-Bench).

## Key Components

- **Dataset**: The CosmosQA dataset, a benchmark for commonsense reasoning.
- **LLM**: The notebook utilizes the `Mistral-7B-Instruct-v0.3` model for predictions.
- **MAPIE for Conformal Prediction**: The `SplitConformalClassifier` from MAPIE is used to generate prediction sets with a given confidence level.

## Steps Covered

1. **Setup & Installation**
   - Clone the repository and install required dependencies.
   - Authenticate with Hugging Face Hub to access the LLM.

2. **Data Preprocessing**
   - Load and transform CosmosQA data into a format suitable for the model.

3. **Model Loading**
   - Load the `Mistral-7B` model and its tokenizer.
   - Define an `LLMClassifier` wrapper to make predictions in a structured format.

4. **Conformal Prediction with MAPIE**
   - Use `SplitConformalClassifier` to conformalize the model on a subset of the data.
   - Generate prediction sets with a 95% confidence level.

5. **Evaluation & Visualization**
   - Compute accuracy scores and coverage metrics.
   - Visualize the size distribution of prediction sets.
   - Plot accuracy per prediction set size.

## Results

- The LLM achieves an accuracy of approximately **86%** on the test set.
- Prediction sets provide calibrated uncertainty estimates, enhancing reliability in decision-making.
- The more uncertain the model is (i.e., the larger the prediction sets), the lower the accuracy.

## Conclusion

This notebook illustrates how conformal prediction techniques can be applied to LLMs for more trustworthy AI systems. The approach can be extended to other question-answering datasets and models to assess confidence in model predictions.

