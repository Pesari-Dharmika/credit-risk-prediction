# 💳 Credit Risk Prediction

A machine learning project to predict whether a customer is likely to default on a loan, using a synthetic dataset of 10,000 customers and 20 features.

---

## 📌 Problem Statement

Credit risk prediction is a critical task in the banking and finance industry. The goal is to classify customers as **high risk (1)** or **low risk (0)** of defaulting on a loan, based on their financial and demographic information.

---

## 📂 Project Structure

```
credit-risk-prediction/
├── credit_risk_prediction.ipynb   # Main Jupyter Notebook
├── synthetic_dataset_10000x20.csv # Dataset
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | Synthetic dataset |
| Rows | 10,000 |
| Features | 20 |
| Target | `target_default_risk` (0 = No Default, 1 = Default) |
| Class Balance | Slightly imbalanced (51.3% vs 48.7%) |

**Key Features:**
- `age`, `income`, `savings`, `monthly_expenses`
- `credit_score`, `loan_amount`, `loan_term_months`
- `num_dependents`, `education`, `marital_status`
- `home_ownership`, `region`, `signup_date`

---

## 🔄 Project Workflow

### 1. 🔍 Exploratory Data Analysis (EDA)
- Dataset shape, data types, missing values, unique values
- Histograms and boxplots for numeric features
- Bar charts for categorical features
- Correlation heatmap
- Pairplot (on 500 sample rows)
- Target class distribution analysis

### 2. 🛠️ Data Preprocessing
- **Missing Values:** Filled with median (Income, Savings, Monthly Expenses, Credit Score)
- **Spelling Fix:** Corrected `Bachlors` → `Bachelors` in Education column
- **Encoding:**
  - One-Hot Encoding → `marital_status`, `home_ownership`, `region`
  - Ordinal Encoding → `education` (HS=1, Bachelors=2, Masters=3, PhD=4, Other=5)
- **Date Feature:** Converted `signup_date` → `days_since_signup`
- **Dropped Columns:** `customer_id`, original categorical columns post-encoding
- **Feature Engineering:** Created `loan_to_income_ratio`
- **Scaling:** StandardScaler on numeric features
- **Outlier Treatment:** Winsorization using IQR method

### 3. 🤖 Model Building

| Model | Accuracy |
|---|---|
| Logistic Regression | 93.95% |
| Decision Tree | 91.30% |
| SVM | 92.25% |
| Random Forest | 93.10% |
| **XGBoost** | **95.00%** ✅ |

### 4. ⚖️ Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) on training data

### 5. 🎯 Hyperparameter Tuning
- **Random Forest** → GridSearchCV
  - Best Params: `max_depth=20`, `max_features='sqrt'`, `n_estimators=500`
  - Accuracy after tuning: **93.3%**
- **XGBoost** → RandomizedSearchCV
  - Best Params: `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`
  - Accuracy after tuning: **95.0%**

### 6. 📉 Confusion Matrix
- Visualized confusion matrices for all 5 models

---

## 🏆 Best Model — XGBoost

```
              precision    recall  f1-score   support

           0       0.95      0.95      0.95       968
           1       0.95      0.95      0.95      1032

    accuracy                           0.95      2000
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook credit_risk_prediction.ipynb
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Programming Language |
| Pandas & NumPy | Data Manipulation |
| Matplotlib & Seaborn | Data Visualization |
| Scikit-learn | ML Models & Preprocessing |
| XGBoost | Gradient Boosting Model |
| Imbalanced-learn | SMOTE for class imbalance |

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
jupyter
```

---

## 📈 Results Summary

- Best Model: **XGBoost** with **95% accuracy**
- XGBoost showed consistent 95% precision and recall for both classes
- SMOTE improved model fairness across classes
- Hyperparameter tuning confirmed XGBoost's robustness

---

## 🙋 Author

**PESARI DHARMIKA**
- GitHub: Pesari-Dharmika (https://github.com/Pesari-Dharmika)
- LinkedIn:Dharmika pesari(www.linkedin.com/in/dharmika-pesari-2809913a2)
