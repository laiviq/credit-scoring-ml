# Credit Scoring ML

Machine learning project for credit risk classification and creditworthiness analysis.

## Project Overview

This project is based on my graduation thesis: "Development and Training of a Neural Network for Customer Creditworthiness Analysis".

The main goal of the project is to solve a binary classification task: predicting whether a client belongs to a higher or lower credit risk group based on tabular financial and personal data.

The project includes data preprocessing, model training, model evaluation and comparison of several machine learning approaches.

## Models

The following models were trained and compared:

- Logistic Regression
- CatBoost
- Multilayer Perceptron

Logistic Regression was used as a baseline interpretable model.  
CatBoost was used as a strong gradient boosting model for tabular data.  
MLP was used as a neural-network-based approach for classification.

## Metrics

The models were evaluated using the following metrics:

- ROC-AUC
- Recall
- Precision
- F1-score
- Logloss
- Accuracy

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- matplotlib
- joblib / pickle
- Streamlit

## Project Structure

text credit-scoring-ml/ ├── data/          # dataset files or sample data ├── models/        # trained models ├── outputs/       # metrics, plots and reports ├── src/           # source code for preprocessing, training and evaluation ├── app.py         # Streamlit application ├── README.md └── .gitignore 

## Features

- Data loading and preprocessing
- Train/test split
- Model training
- Model comparison
- Evaluation with classification metrics
- Saving trained models
- Basic Streamlit interface for demonstration

## Results

The project compares classical machine learning and neural-network-based approaches for credit scoring.

The final model selection is based on classification quality metrics, with special attention to ROC-AUC and Recall, since credit scoring tasks require careful risk detection.

## How to Run

Install dependencies:

bash pip install -r requirements.txt 

Run the Streamlit app:

bash streamlit run app.py 

Or run training/evaluation scripts from the src directory.

## Notes

This repository is intended for educational and portfolio purposes.  
If real or sensitive data was used during research, it is not published in this repository.
