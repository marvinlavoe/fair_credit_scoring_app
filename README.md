# SHAP-Enhanced Fair Credit Scoring System

## Project Overview

This project is a fair and explainable credit scoring application built for an academic final-year project demonstration. It combines a machine learning backend with a Streamlit frontend so a user can enter applicant information, receive a creditworthiness prediction, inspect explanation outputs, and review fairness indicators.

The application is titled:

```text
Fair Credit Scoring with Explainability
```

Subtitle:

```text
TabNet Prediction | SHAP Explanation | Fairness Evaluation
```

The system is designed for research and demonstration only. It must not be used for real financial decision-making.

## Main Objectives

- Predict whether an applicant is creditworthy or not creditworthy.
- Train a Logistic Regression baseline model for comparison.
- Train a TabNet model for tabular credit risk prediction.
- Apply fairness-aware sample reweighting to reduce group imbalance during TabNet training.
- Evaluate fairness using demographic parity difference, equalized odds difference, disparate impact ratio, and group-level rates.
- Generate SHAP-based explanation artifacts for model interpretability.
- Provide a clean Streamlit frontend for project demonstration.

## Important Methodology Note

The project name references adversarial debiasing because the overall research theme is fairness-aware debiasing. The implemented backend currently uses fairness-aware sample reweighting based on target class and sensitive group. This is not a full neural adversarial debiasing network. The code and saved metadata label it correctly as:

```text
fairness-aware reweighting
```

This keeps the implementation academically honest while still supporting a fairness-improvement workflow.

## Technology Stack

- Python for backend development.
- Pandas and NumPy for data handling.
- Scikit-learn for preprocessing, Logistic Regression, metrics, and train/test splitting.
- PyTorch TabNet through `pytorch-tabnet` for the main neural tabular model.
- Fairlearn for fairness metrics.
- SHAP for explainability.
- Matplotlib for saved SHAP visualizations.
- Plotly for frontend charts.
- Streamlit for the user interface.
- Joblib and JSON for saved model artifacts and metadata.

## Final Project Structure

```text
fair_credit_scoring_app/
|-- app.py
|-- README.md
|-- requirements.txt
|-- .streamlit/
|   `-- config.toml
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- german_clean.csv
|   |-- heloc_dataset_v1 (1).csv
|   `-- processed/
|       |-- german_X_train.csv
|       |-- german_X_test.csv
|       |-- german_y_train.csv
|       |-- german_y_test.csv
|       |-- german_protected_train.csv
|       |-- german_protected_test.csv
|       |-- heloc_X_train.csv
|       |-- heloc_X_test.csv
|       |-- heloc_y_train.csv
|       |-- heloc_y_test.csv
|       `-- heloc_special_value_summary.csv
|-- frontend/
|   |-- __init__.py
|   |-- components.py
|   |-- pages.py
|   `-- styles.py
|-- models/
|   |-- logistic_regression.pkl
|   |-- tabnet_baseline.zip
|   |-- tabnet_debiased.zip
|   |-- preprocessor.pkl
|   |-- feature_names.pkl
|   |-- raw_feature_columns.pkl
|   |-- metadata.json
|   |-- german_preprocessor.joblib
|   `-- heloc_preprocessor.joblib
|-- notebooks/
|   `-- preprocess.ipynb
|-- outputs/
|   |-- metrics.json
|   |-- fairness_results.json
|   |-- model_comparison.csv
|   |-- shap_summary.png
|   `-- shap_waterfall.png
|-- src/
|   |-- clean_headers.py
|   |-- config.py
|   |-- data_loader.py
|   |-- evaluate.py
|   |-- explainability.py
|   |-- fairness.py
|   |-- generate_shap_artifacts.py
|   |-- inference.py
|   |-- preprocess.py
|   |-- preprocessing.py
|   |-- train_all.py
|   |-- train_baseline.py
|   |-- train_debiased_tabnet.py
|   |-- train_tabnet.py
|   `-- utils.py
`-- utils/
    |-- __init__.py
    |-- mock_data.py
    `-- mock_prediction.py
```

## Data Used

### German Credit Dataset

The German Credit dataset is the primary dataset. In this project, the cleaned file is:

```text
data/german_clean.csv
```

The raw file name expected by the cleaner is:

```text
data/german_credit_data.csv
```

If the raw file is missing, `src/clean_headers.py` safely uses `data/german_clean.csv` and enhances it with sensitive attributes.

### HELOC Dataset

The HELOC dataset is included as an optional extension:

```text
data/heloc_dataset_v1 (1).csv
```

The notebook preprocessing workflow handles HELOC special negative values such as `-9`, `-8`, and `-7` by creating special-code indicators and imputing missing numeric values.

## Dataset Cleaning

Cleaning is handled by:

```text
src/clean_headers.py
```

The cleaner performs these steps:

- Loads the raw German Credit file if present.
- Renames German column headers to readable English names.
- Preserves binary target labels where already encoded as `1` and `0`.
- Converts raw target labels where needed using `1 -> 1` and `2 -> 0`.
- Extracts `sex` from `personal_status_sex`.
- Creates `age_group` using `below_25` and `25_and_above`.
- Saves the final cleaned dataset to `data/german_clean.csv`.

For this project's current data, `personal_status_sex` is numeric-coded. The implementation maps code `2` to `female` and the other observed codes to `male`.

Run cleaning with:

```powershell
.\.conda\python.exe src\clean_headers.py
```

## Preprocessing Pipeline

The backend preprocessing module is:

```text
src/preprocess.py
```

It performs:

- Loading of `data/german_clean.csv`.
- Validation of the target column.
- Validation or creation of sensitive attributes.
- Separation of features, target, and sensitive attributes.
- Exclusion of `sex`, `age_group`, and `personal_status_sex` from model training features.
- Numeric imputation and scaling.
- Categorical imputation and one-hot encoding.
- Stratified train/test splitting.
- Saving of fitted preprocessing artifacts.

The preprocessing dictionary contains:

```python
{
    "X_train": X_train_processed,
    "X_test": X_test_processed,
    "y_train": y_train,
    "y_test": y_test,
    "A_train": A_train,
    "A_test": A_test,
    "preprocessor": preprocessor,
    "feature_names": feature_names
}
```

Run preprocessing with:

```powershell
.\.conda\python.exe src\preprocess.py
```

## Notebook Preprocessing

The notebook:

```text
notebooks/preprocess.ipynb
```

was built to preprocess both German Credit and HELOC datasets. It saves processed train/test CSV files under:

```text
data/processed/
```

German Credit notebook outputs include:

- `german_X_train.csv`
- `german_X_test.csv`
- `german_y_train.csv`
- `german_y_test.csv`
- `german_protected_train.csv`
- `german_protected_test.csv`

HELOC notebook outputs include:

- `heloc_X_train.csv`
- `heloc_X_test.csv`
- `heloc_y_train.csv`
- `heloc_y_test.csv`
- `heloc_special_value_summary.csv`

## Model Training

### Logistic Regression Baseline

File:

```text
src/train_baseline.py
```

Model:

```python
LogisticRegression(max_iter=1000, class_weight="balanced")
```

Metrics:

- Accuracy
- Weighted F1 score
- AUC-ROC

Saved artifact:

```text
models/logistic_regression.pkl
```

Run with:

```powershell
.\.conda\python.exe src\train_baseline.py
```

### TabNet Baseline

File:

```text
src/train_tabnet.py
```

The TabNet baseline uses:

```python
TabNetClassifier(
    n_d=16,
    n_a=16,
    n_steps=4,
    gamma=1.5,
    lambda_sparse=1e-4,
    seed=42,
    verbose=1
)
```

Default training settings:

```text
max_epochs=100
patience=15
batch_size=256
virtual_batch_size=64
eval_metric=["auc"]
```

Saved artifact:

```text
models/tabnet_baseline.zip
```

Run with:

```powershell
.\.conda\python.exe src\train_tabnet.py
```

### Fairness-Aware Reweighted TabNet

File:

```text
src/train_debiased_tabnet.py
```

This model uses sample weights computed from:

- Target class.
- Sensitive group, currently `sex`.

The sample weighting method is implemented in:

```text
src/fairness.py
```

Function:

```python
compute_group_reweighting(y_train, sensitive_features)
```

Saved artifact:

```text
models/tabnet_debiased.zip
```

Run with:

```powershell
.\.conda\python.exe src\train_debiased_tabnet.py
```

## Full Training Pipeline

The full backend pipeline is:

```text
src/train_all.py
```

It runs:

1. Header cleaning and sensitive attribute creation.
2. Preprocessing.
3. Logistic Regression baseline training.
4. TabNet baseline training.
5. Fairness-aware reweighted TabNet training.
6. Fairness evaluation.
7. Model comparison generation.
8. SHAP artifact generation.
9. Metadata saving.

Run the full pipeline:

```powershell
.\.conda\python.exe src\train_all.py
```

For a quick smoke test, reduce TabNet epochs:

```powershell
$env:FAIR_CREDIT_TABNET_EPOCHS='5'
.\.conda\python.exe src\train_all.py
```

For full training, remove the override or set:

```powershell
$env:FAIR_CREDIT_TABNET_EPOCHS='100'
.\.conda\python.exe src\train_all.py
```

## Fairness Evaluation

Fairness logic is implemented in:

```text
src/fairness.py
```

The system evaluates fairness for:

- `sex`
- `age_group`

Metrics include:

- Demographic parity difference.
- Equalized odds difference.
- Disparate impact ratio.
- Selection rate by group.
- True positive rate by group.
- False positive rate by group.

Disparate impact interpretation:

```text
>= 0.8 = acceptable
< 0.8 = needs review
```

Saved result:

```text
outputs/fairness_results.json
```

Run evaluation with:

```powershell
.\.conda\python.exe src\evaluate.py
```

## SHAP Explainability

SHAP functions are implemented in:

```text
src/explainability.py
```

Supported functions include:

- `generate_shap_values`
- `generate_shap_waterfall_plot`
- `generate_shap_summary_plot`
- `get_top_feature_contributions`

Because TabNet is not a tree model, the project uses:

```python
shap.KernelExplainer
```

To reduce computation time, SHAP uses a small background sample.

Generate SHAP artifacts with:

```powershell
.\.conda\python.exe src\generate_shap_artifacts.py
```

Saved outputs:

```text
outputs/shap_summary.png
outputs/shap_waterfall.png
```

## Backend Inference API

Frontend-ready functions are exposed in:

```text
src/inference.py
```

Main functions:

```python
load_artifacts()
predict_credit(applicant_data: dict)
explain_prediction(applicant_data: dict)
get_fairness_metrics()
get_model_comparison()
```

`predict_credit()` returns:

```python
{
    "prediction": "Creditworthy",
    "prediction_label": 1,
    "confidence": 0.87,
    "risk_level": "Low Risk"
}
```

Risk level rules:

- `confidence >= 0.75` and `prediction_label == 1`: Low Risk.
- `0.55 <= confidence < 0.75`: Medium Risk.
- `prediction_label == 0`: High Risk.

Test backend inference:

```powershell
.\.conda\python.exe -c "import sys; sys.path.insert(0,'src'); from inference import predict_credit; print(predict_credit({'age':34,'loan_amount':3500,'loan_duration':24}))"
```

## Streamlit Frontend

The main frontend entry point is:

```text
app.py
```

The frontend is modularized into:

```text
frontend/components.py
frontend/pages.py
frontend/styles.py
```

Mock frontend utilities are kept in:

```text
utils/mock_prediction.py
utils/mock_data.py
```

The frontend includes:

- Sidebar navigation.
- Dashboard overview.
- Applicant input form.
- Prediction result panel.
- SHAP explanation panel.
- Fairness metrics panel.
- Model comparison panel.
- About and methodology page.

The current UI theme is configured in:

```text
.streamlit/config.toml
frontend/styles.py
```

The design uses a high-contrast academic navy-and-ivory theme to keep text, cards, forms, tables, and charts visible during demonstrations.

Run the frontend:

```powershell
.\.conda\python.exe -m streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

## How to Test the Complete Software

### 1. Verify cleaning

```powershell
.\.conda\python.exe src\clean_headers.py
```

Expected output:

```text
data/german_clean.csv
```

### 2. Verify preprocessing

```powershell
.\.conda\python.exe src\preprocess.py
```

Expected output:

```text
models/preprocessor.pkl
models/feature_names.pkl
models/raw_feature_columns.pkl
```

### 3. Train baseline model

```powershell
.\.conda\python.exe src\train_baseline.py
```

Expected output:

```text
models/logistic_regression.pkl
outputs/metrics.json
```

### 4. Train TabNet models

Quick smoke test:

```powershell
$env:FAIR_CREDIT_TABNET_EPOCHS='5'
.\.conda\python.exe src\train_tabnet.py
.\.conda\python.exe src\train_debiased_tabnet.py
```

Expected outputs:

```text
models/tabnet_baseline.zip
models/tabnet_debiased.zip
outputs/model_comparison.csv
```

### 5. Run full pipeline

```powershell
$env:FAIR_CREDIT_TABNET_EPOCHS='5'
.\.conda\python.exe src\train_all.py
```

Expected outputs:

```text
models/metadata.json
outputs/metrics.json
outputs/fairness_results.json
outputs/model_comparison.csv
outputs/shap_summary.png
outputs/shap_waterfall.png
```

### 6. Run frontend

```powershell
.\.conda\python.exe -m streamlit run app.py
```

Then open:

```text
http://127.0.0.1:8501
```

Test these pages:

- Dashboard.
- Predict Applicant.
- SHAP Explanation.
- Fairness Metrics.
- Model Comparison.
- About Project.

## Running With Conda

This project was tested using a local Conda environment:

```text
.conda/
```

Use:

```powershell
.\.conda\python.exe -m pip install -r requirements.txt
```

Then run commands using:

```powershell
.\.conda\python.exe
```

## Requirements

The core dependencies are listed in:

```text
requirements.txt
```

Main packages:

- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- streamlit
- plotly
- shap
- fairlearn
- torch
- pytorch-tabnet
- ucimlrepo

## Generated Output Files

### Model artifacts

```text
models/logistic_regression.pkl
models/tabnet_baseline.zip
models/tabnet_debiased.zip
models/preprocessor.pkl
models/feature_names.pkl
models/raw_feature_columns.pkl
models/metadata.json
```

### Evaluation outputs

```text
outputs/metrics.json
outputs/fairness_results.json
outputs/model_comparison.csv
```

### Explainability outputs

```text
outputs/shap_summary.png
outputs/shap_waterfall.png
```

## Current Smoke-Test Results

The backend was smoke-tested with:

```powershell
$env:FAIR_CREDIT_TABNET_EPOCHS='5'
.\.conda\python.exe src\train_all.py
```

This verified:

- Cleaning completed successfully.
- Logistic Regression trained successfully.
- TabNet baseline trained successfully.
- Fairness-aware reweighted TabNet trained successfully.
- Fairness results were generated.
- Model comparison was generated.
- SHAP summary and waterfall images were generated.
- `src/inference.py` returned frontend-ready predictions.

Short smoke-test metrics are not final model results because TabNet was trained for only 5 epochs. For report-quality results, run the full 100-epoch training pipeline.

## Known Limitations

- The current fairness-aware method is sample reweighting, not a full adversarial neural debiasing model.
- TabNet training on CPU can be slow.
- SHAP KernelExplainer is computationally expensive, so the implementation uses a small sample size for practical execution.
- The Streamlit frontend uses the trained backend for predictions and Kernel SHAP explanations when model artifacts are available; mock values remain only as a fallback when artifacts cannot be loaded.
- If `data/german_credit_data.csv` is absent, the cleaner relies on the existing `data/german_clean.csv`.

## Future Improvements

- Connect the Streamlit frontend directly to `src/inference.py` for live trained-model predictions.
- Add a true adversarial debiasing neural network branch.
- Add model selection between Logistic Regression, TabNet baseline, and reweighted TabNet.
- Add dataset switching between German Credit and HELOC.
- Add downloadable reports for prediction explanations and fairness metrics.
- Add unit tests for preprocessing, fairness, inference, and artifact loading.

## Disclaimer

This system is for academic research and demonstration only. It should not be used for real credit approval, lending decisions, or financial risk assessment in production.
