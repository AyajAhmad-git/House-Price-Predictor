🏡 California House Price Prediction

A complete Machine Learning pipeline for predicting house prices using the California Housing Dataset.
This project includes data preprocessing, feature engineering, model training, and automated inference using a saved pipeline and model.

📌 Project Overview

This project builds a fully automated ML system that:

Loads and preprocesses the California housing dataset

Performs stratified sampling using income categories

Cleans and transforms numerical & categorical data

Trains a Random Forest Regressor

Saves the final model and preprocessing pipeline using Joblib

Loads the model for inference on new input data

Generates predictions and exports them to output.csv

The entire workflow is written in pure Python using commonly used ML libraries.

🚀 Features

Automated preprocessing pipeline (imputation, scaling, one-hot encoding)

Stratified sampling to maintain distribution of income

Random Forest model for robust predictions

Reusable saved model & pipeline for real-time predictions

Simple train→save→predict workflow

Works with any new input file (input.csv)

🧠 Technologies Used

Python

Pandas

NumPy

Scikit-Learn

Joblib

📂 Project Structure
├── housing.csv            # Dataset used for training
├── input.csv              # New data for prediction (user provided)
├── output.csv             # Generated prediction results
├── model.pkl              # Saved trained model
├── pipeline.pkl           # Saved preprocessing pipeline
└── main.py                # Main training + inference script

🏗️ How It Works
1️⃣ Training Phase (runs automatically if no model exists)

Reads housing.csv

Creates income categories for stratified sampling

Splits dataset into training data only

Prepares numerical & categorical features

Builds preprocessing pipeline using:

SimpleImputer

StandardScaler

OneHotEncoder

Trains RandomForestRegressor

Saves:

model.pkl

pipeline.pkl

2️⃣ Inference Phase (runs if model already exists)

Loads saved model & pipeline

Reads input.csv

Applies the same transformations

Predicts median_house_value

Saves predictions to output.csv

📘 Usage Instructions
🔧 1. Install Dependencies
pip install pandas numpy scikit-learn joblib

▶️ 2. Run the Script
python main.py

📤 3. Predict on New Data

Add new house data to input.csv

Run the script again

Check output.csv for predictions

📈 Model

The model used is a Random Forest Regressor due to its robustness, non-linearity handling, and strong performance on tabular datasets.

🤝 Contributions

Contributions, issues, and feature requests are welcome!

⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
