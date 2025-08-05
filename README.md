# Machine Learning Project - 🧠 Stroke Prediction


<p align="center">
  <img src="https://github.com/user-attachments/assets/64f454cd-037b-4e7d-88b6-1429d8418ab2" width="700" />
</p>

## 📌 Project Overview
This project develops a machine learning system to **predict stroke risk in patients** using clinical and demographic data. The analysis identifies key risk factors and creates a deployable **web application** for real-time stroke risk assessment. Using advanced feature engineering and model optimization, the system provides **clinically relevant predictions** to support healthcare professionals.

## 📊 Dataset Overview
The dataset contains records of **5,110 patients** with stroke outcomes derived from healthcare settings.

### **Features include:**
- **Demographics:** Age, Gender, Residence Type (Urban/Rural)
- **Medical History:** Hypertension, Heart Disease, BMI, Average Glucose Level
- **Lifestyle:** Smoking Status, Work Type, Marital Status
- **Target Variable:** Stroke (binary: 0 = No, 1 = Yes)

### **Dataset Characteristics:**
- 4.9% stroke incidence (highly imbalanced)
- ~4% missing BMI values
- Age range: 0.08 to 82 years
- Requires careful preprocessing and resampling (SMOTE)

## 🩺 Project Objective

Stroke is a **leading cause of death and long-term disability**. The goal of this project is to build a simple and interpretable machine learning model that predicts the likelihood of a patient having a stroke based on demographic, medical, and lifestyle data.

Early identification of high-risk individuals is crucial for:

- Flag patients who may be at higher risk
- Understand key modifiable risk factors
- Support early screening or further clinical investigation
- Demonstrate how basic clinical data can be used for predictive modeling
- Serve as a prototype for more advanced risk assessment tools

## 🔧 Methodology Overview

### 🔬 Data Processing & Feature Engineering
- BMI imputation (median)
- New clinical features:
  - Age groups (young, middle-aged, senior)
  - BMI categories (underweight, normal, overweight, obese)
  - Glucose levels grouped (normal, prediabetic, high)
  - Risk factor count and interaction terms
  - BMI/glucose ratio
- One-hot encoding and scaling
- SMOTE resampling for class imbalance

### 🧠 Models Trained

1. **Logistic Regression (Primary Model)**
   - Interpretable, simple
   - Tuned: C=0.1, L1 penalty, class weights
   - 7 best features selected for deployment

2. **Random Forest**
   - Ensemble method trained with 100 trees (default setting)
   - Used to explore non-linear patterns and feature importance

3. **XGBoost**
   - Gradient boosting for higher performance
   - Tuned via cross-validation

### ⚙️ Feature Selection & Optimization
- Recursive feature importance analysis
- Feature count reduced from 30 ➝ 7
- Only ~2% performance loss
- GridSearchCV used for hyperparameter tuning

### 📏 Evaluation Metrics
- **Primary**: Recall (maximize detection of stroke cases)
- **Secondary**: Precision, F1, ROC-AUC
- Stratified train-test split (80/20)
- 5-fold cross-validation

## 🚀 Deployment (instructions see below under 'Deployment Preparation')

Streamlit web application enables **real-time prediction** in a browser:
- Intuitive form to enter patient data
- Risk classification: Low, Moderate, High, Very High
- Probability score and prediction label
- Designed for easy clinical use

## 📁 Project Files and Structure

| File/Folder                        | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `Stroke_Prediction_ML_Dev.ipynb`       | Full notebook with EDA, statistical testing, model training, and evaluation |
| `Stroke_Prediction_ML_Deployment.ipynb` | Deployment-specific notebook to test predictions and load final artifacts   |
| `stroke_functions.py`            | All helper functions for preprocessing, visualization, and model evaluation |
| `streamlit_app.py`               | Web app built with Streamlit for real-time stroke risk prediction            |
| `final_model_artifacts/`         | Folder containing saved model, pipeline, and feature list (`.pkl`, `.json`) |
| `requirements.txt`               | List of Python packages needed to run the analysis and app                   |


## 📦 `final_model_artifacts/` Contents

This directory contains all saved components needed for inference and deployment:

- `final_model.pkl` — Final trained **Logistic Regression** model (with top 7 features)
- `preprocessing_pipeline.pkl` — Full **preprocessing pipeline** used during training (scaling, encoding, imputation)
- `feature_names.json` — List of final feature names used in the model input

These ensure consistent transformation and prediction during deployment.

## Prerequisites

To run the notebook and deployment app, make sure you have:

- **Python** version: `>=3.8`

### Required Python Libraries

- Data handling:
  - `pandas`
  - `numpy`
- Modeling & preprocessing:
  - `scikit-learn`
  - `imbalanced-learn`
  - `xgboost`
- Visualization:
  - `matplotlib`
  - `seaborn`
  - `plotly`
- Deployment:
  - `streamlit`
  - `joblib`
- Notebook:
  - `jupyter`

You can install all dependencies at once with:

```bash
pip install -r requirements.txt
```

## 🛠️ Setup Instructions

### 🔬 Analysis Environment

```bash
# Clone repository
git clone https://github.com/ML_Stroke_Prediction_Project

cd Stroke_Prediction_ML_Dev

# (Optional) Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🧪 Running the Analysis

Start Jupyter Notebook:
```bash
jupyter notebook Stroke_Prediction_ML_Dev.ipynb
```
This will open the notebook in your default web browser
Run all cells sequentially to reproduce the complete analysis

To run the analysis successfully, make sure the **stroke dataset CSV** file is placed in the root project directory with the following filename in Jupyter Notebook:
```bash
import pandas as pd

# Load dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Basic overview
print(f"Dataset shape: {df.shape}")
df.head()
```

The analysis includes comprehensive visualizations created with functions from `stroke_functions.py`:

Example usage:
```bash
import stroke_functions as viz

# Plot feature importance
viz.plot_feature_importance(model, feature_names, model_type="linear")
```

File Structure Note: Ensure stroke_functions.py is in the same directory as the Jupyter Notebook for successful analysis execution.

## 🧱 Deployment Preparation: `Stroke_Prediction_ML_Deployment.ipynb`

This intermediate notebook prepares the trained model for deployment.  
It serves as a **bridge between model development and the live app**.

### 📋 What this notebook does:
- Loads the final trained Logistic Regression model (`best_model`)
- Loads and applies the fitted preprocessing pipeline
- Defines the `predict_stroke_risk()` function used in the app
- Tests two realistic patient cases (low- and high-risk profiles)
- Saves all deployment-ready artifacts to the `final_model_artifacts/` folder

### 📁 Outputs:
- `final_model.pkl` – Trained model
- `preprocessing_pipeline.pkl` – Data transformer
- `feature_names.json` – Feature structure used for inference
- `final_evaluation_report.csv` – Final metrics on test set

### ▶️ How to use it:
1. Open the notebook:
   ```bash
   jupyter notebook Stroke_Prediction_ML_Deployment.ipynb
   ```
2. Run all cells after completing your analysis.
3. Confirm the two local patient tests return valid risk predictions.
4. Proceed to launching the app using Streamlit


## 🌐 Deploying the Web Application (Local)

Ensure model `final_model_artifacts/` folder has been generated by running the notebook (run the analysis notebook first)
Then Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```
The application will open automatically at http://localhost:8501
Input patient data and receive instant stroke risk predictions


## 📚 Glossary

- **SMOTE**: Synthetic Minority Oversampling Technique — generates synthetic samples of the minority class to address data imbalance.

- **CHA₂DS₂-VASc**: A widely used clinical stroke risk score, particularly for patients with atrial fibrillation. It helps estimate stroke risk based on comorbidities.

- **Streamlit**: Lightweight Python framework for building ML web apps with minimal effort and no frontend expertise required.

- **REST API**: A standard way for applications to send input data and receive predictions over the web using HTTP requests.

- **WCAG**: Web Content Accessibility Guidelines — a set of standards to ensure websites and apps are usable by people with disabilities (e.g., screen readers, keyboard navigation).

## ⚠️ Disclaimer
This tool is designed for **educational and research purposes**. In clinical settings, it should supplement, not replace, professional medical judgment and established diagnostic protocols.
