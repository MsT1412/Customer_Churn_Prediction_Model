# 🔍 Customer Churn Prediction — Hybrid Ensemble ML Model
________________________________________________________________________________________________________________________________________________________________
## 📌 Project Overview
An end-to-end machine learning project that predicts customer churn for a telecommunications company using a hybrid ensemble model combining Random Forest and XGBoost, with SHAP-based Explainable AI to identify and interpret key churn drivers.

This project demonstrates the full data science pipeline — from raw data ingestion and exploratory analysis through to model deployment and business insight generation — directly applicable to customer retention strategy in financial services and telecommunications.
_________________________________________________________________________________________________________________________________________________________________

## 🎯 Business Problem
Approximately 1 in 4 customers (26.6%) were churning, representing significant revenue loss. The goal was to build a model that:
- Accurately predicts which customers are likely to churn
- Explains **why** they are likely to churn (not just that they will)
- Generates actionable business recommendations to improve retention
__________________________________________________________________________________________________________________________________________________________________

## 🛠️ Tech Stack
| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation and cleaning |
| Matplotlib & Seaborn | Exploratory data visualisation |
| Scikit-learn | Model building and evaluation |
| XGBoost | Gradient boosting classifier |
| SHAP | Explainable AI — feature impact analysis |
| Joblib | Model serialisation and saving |
| Jupyter Notebook | Development environment |
__________________________________________________________________________________________________________________________________________________________________

## 📁 Project Structure
```
Customer_Churn_Prediction_Model/
│
├── data/
│   └── raw/                        ← Original IBM Telco dataset
│
├── notebooks/
│   └── customer_churn_prediction.ipynb   ← Full analysis notebook
│
├── visuals/                        ← All charts and SHAP plots
│   ├── 01_churn_distribution.png
│   ├── 02_churn_patterns.png
│   ├── 03_charges_service_analysis.png
│   ├── 04_confusion_matrix.png
│   ├── 05_shap_feature_importance.png
│   └── 06_shap_impact_plot.png
│
├── models/                         ← Saved trained models
│   ├── hybrid_churn_model.pkl
│   └── scaler.pkl
│
├── screenshots/                    ← Step-by-step process documentation
│   ├── 01_importinglibraries.png
│   ├── 02_data_loading.png
│   ├── 03_DataExploration1.png
│   ├── 04_DataExploration2.png
│   ├── 05_data_cleaning.png
│   ├── 06_eda_churn_distribution.png
│   ├── 07_eda_churn_patterns.png
│   ├── 08_eda_charges_service.png
│   ├── 09_preprocessing.png
│   ├── 10_train_test_split.png
│   ├── 11_model_training.png
│   ├── 12_model_evaluation1.png
│   ├── 13_ModelEvaluation2.png
│   ├── 14_SHAP_Implementation.png
│   ├── 15_Shap_Plots1.png
│   └── 16_Shap_Plots2.png
│
└── README.md
```
_______________________________________________________________________________________________________________________________________________________________

## 📊 Dataset
- **Source:** IBM Telco Customer Churn Dataset (via Kaggle)
- **Size:** 7,043 customers, 21 features
- **Target:** Churn (Yes/No)
- **Class Distribution:** 73.4% No Churn / 26.6% Churned
__________________________________________________________________________________________________________________________________________________________________
## 🔬 Methodology

### 1. Data Cleaning & Preprocessing
- Fixed TotalCharges column incorrectly typed as string — converted to float64
- Removed 11 rows with hidden blank values
- Encoded all binary Yes/No columns to 1/0
- Applied one-hot encoding to multi-class categorical columns
- Scaled numeric features using StandardScaler

### 2. Exploratory Data Analysis
Key findings before modelling:
- Month-to-month contract customers churn at nearly 3x the rate of annual contract customers
- Customers in their first 10 months are the highest churn risk
- Fiber optic internet customers show disproportionately high churn rates
- Higher monthly charges correlate with increased churn likelihood

### 3. Hybrid Ensemble Model
Combined two complementary models using Soft Voting:
- **Random Forest** — robust to overfitting, handles mixed feature types well
- **XGBoost** — high performance gradient boosting, strong on structured tabular data
- **Soft Voting** — averages predicted probabilities from both models for a more confident final prediction
- Class imbalance handled via `class_weight='balanced'` and `scale_pos_weight`

### 4. Explainable AI — SHAP Analysis
Applied SHAP (SHapley Additive exPlanations) to interpret model predictions:
- Identifies which features most influence churn predictions
- Shows the **direction** of each feature's impact (increases or decreases churn probability)
- Provides both global feature importance and individual prediction explanations
___________________________________________________________________________________________________________________________________________________________________

## 📈 Results

| Metric | Score |
|---|---|
| Overall Accuracy | 77% |
| ROC-AUC Score | 0.82 |
| Precision (Churned) | 56% |
| Recall (Churned) | 62% |

A ROC-AUC of 0.82 indicates strong discriminative ability — the model is significantly better than random chance at identifying customers who will churn.

__________________________________________________________________________________________________________________________________________________________________

## 💡 Key Business Insights from SHAP

1. **Contract type is the strongest churn signal** — Month-to-month customers are dramatically more likely to churn. Migrating customers to annual contracts is the single highest-impact retention strategy.
2. **Tenure is critical** — The first 10 months are the highest-risk window. Early engagement programmes targeting new customers would significantly reduce churn.
3. **Fiber optic customers are underserved** — High churn among fiber optic users suggests a pricing or service quality issue that needs investigation.
4. **Electronic check payment correlates with churn** — These customers may benefit from incentives to switch to automatic payment methods, which correlate with higher retention.
5. **Online security absence increases churn** — Bundling security features into standard packages could improve retention.

___________________________________________________________________________________________________________________________________________________________________

## 🔗 Relevance to Honours Research
This project directly parallels the methodology used in my honours research on **Ransomware Detection in Digital Banking Systems**, which applies the same hybrid ensemble approach (Random Forest + XGBoost) with SHAP-based Explainable AI — demonstrating the transferability of these techniques across both business analytics and cybersecurity domains.

__________________________________________________________________________________________________________________________________________________________________

## 👩‍💻 Author
**Tiaksha Kowlesar**
Honours Student — Information and Communications Technology
