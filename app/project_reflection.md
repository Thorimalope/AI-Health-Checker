# Project Reflection — AI Health Predictor (Month 3)

## Project Overview

This project involved building an end-to-end machine learning application that predicts diabetes risk using a public health dataset. The workflow covered data preparation, model development (classical ML + deep learning), model evaluation, and deployment of an interactive user interface using Streamlit.

The final application allows users to enter 8 health parameters and instantly receive:

- a diabetes risk probability
- a risk level (Low / Medium / High)

---

## Dataset Used

**Dataset:** Diabetes dataset (768 rows, 9 columns)  
**Target variable:** `Outcome` (0 = low/no diabetes risk, 1 = higher diabetes risk)

### Key data notes

Some clinical features contained impossible values (e.g., glucose = 0), which were treated as missing values and corrected during preprocessing.

---

## Data Preparation Summary

### Cleaning & preprocessing steps

- Loaded dataset using Pandas
- Identified “hidden missing values” represented as 0 in:
  - Glucose, BloodPressure, SkinThickness, Insulin, BMI
- Replaced invalid 0 values with `NaN`
- Filled missing values using **median imputation**
- Split data into:
  - 70% training
  - 15% validation
  - 15% test (stratified splits)
- Applied **StandardScaler** fitted on training data only
- Saved the scaler for deployment (`scaler.pkl`)

---

## Model Comparisons

### Models Built

1. **Baseline (scikit-learn): Logistic Regression**
   - Standard scaling applied
   - Two versions tested:
     - Normal Logistic Regression
     - Logistic Regression with `class_weight="balanced"`

2. **Deep Learning Model (TensorFlow/Keras)**
   - 3 hidden layers with ReLU activations
   - Sigmoid output layer
   - EarlyStopping on validation loss

---

### Validation Results

| Model                          | Accuracy | Precision | Recall     | F1         | ROC-AUC |
| ------------------------------ | -------- | --------- | ---------- | ---------- | ------- |
| Logistic Regression            | 0.7130   | 0.5946    | 0.5500     | 0.5714     | 0.8097  |
| Logistic Regression (Balanced) | 0.7304   | 0.5918    | **0.7250** | **0.6517** | 0.8097  |
| Deep Learning (Keras)          | 0.7043   | 0.5938    | 0.4750     | 0.5278     | 0.7983  |

✅ **Chosen baseline for deployment:** Logistic Regression (Balanced)  
Reason: Much stronger recall and F1-score, which is critical for health risk prediction.

---

### Test Results (Final Evaluation)

| Model                          | Accuracy   | Precision  | Recall     | F1         | ROC-AUC    |
| ------------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression (Balanced) | **0.7931** | 0.7179     | **0.6829** | **0.7000** | **0.8608** |
| Deep Learning (Keras)          | 0.7586     | **0.7407** | 0.4878     | 0.5882     | 0.8501     |

✅ Final decision confirmed: Logistic Regression (Balanced) performed best overall, especially recall.

---

## Challenges Encountered

### 1) Hidden missing values

The dataset did not contain obvious missing values (`NaN`) but had invalid zeros in clinical measurements. This required additional investigation and preprocessing steps.

### 2) TensorFlow installation issues

TensorFlow failed to import due to DLL dependency issues and compatibility challenges with Python versions.  
Solution:

- Created a new virtual environment using Python 3.11
- Installed required Windows runtime dependencies (Visual C++ Redistributable)
- Installed TensorFlow successfully in the correct venv

### 3) Kernel / notebook execution issues

Jupyter kernels sometimes stalled and variables disappeared after restarts.  
Solution:

- Improved workflow by saving processed data (`diabetes_clean.csv`) and scaler (`scaler.pkl`)
- Reused saved assets instead of rerunning all preprocessing

### 4) Git tracking virtual environment files

Virtual environment folders caused thousands of tracked changes.  
Solution:

- Updated `.gitignore` to exclude venv folders

---

## Deployment Steps

Deployment was completed using **Streamlit**.

### Steps followed:

1. Built Streamlit UI (`app/app.py`)
2. Loaded saved model + scaler:
   - `baseline_model.pkl`
   - `scaler.pkl`
3. Generated `requirements.txt` using:
   ```bash
   pip freeze > requirements.txt
   ```
4. Uploaded project to GitHub

5. Deployed using Streamlit Cloud by selecting:

6. Main file path: app/app.py

7. Verified application works online and produces consistent predictions

## Key Learnings

- End-to-end ML applications require more than training a model:
  - proper preprocessing
  - consistent scaling
  - saving artifacts
  - deployment readiness

- Recall matters more than accuracy in health prediction use cases
- Deep learning does not automatically beat classical ML on smaller datasets
- Good project structure and saving preprocessing artifacts (CSV + scaler) saves significant time
- Deployment requires careful handling of:
  - file paths
  - package versions
  - dependency management
