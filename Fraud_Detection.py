#!/usr/bin/env python3
"""
build_fraud_notebook.py

Creates a complete Jupyter notebook for the Credit Card Fraud Detection pipeline,
saves an ROC image, and packages both into a zip file.

Usage:
    python build_fraud_notebook.py
"""

import os
import sys
import nbformat as nbf
from datetime import datetime
import zipfile

# Check requirements
missing = []
try:
    import pandas as pd
except Exception:
    missing.append("pandas")
try:
    import sklearn
except Exception:
    missing.append("scikit-learn")
try:
    import matplotlib
except Exception:
    missing.append("matplotlib")

if missing:
    print("WARNING: The following packages are required but not all are installed:", ", ".join(missing))
    print("Install them with: pip install pandas scikit-learn matplotlib xgboost (optional)")
    # We continue because notebook will run on user's machine where they can install packages.

# Filenames
CSV_NAME = "creditcard.csv"
NOTEBOOK_NAME = "Completed_Credit_Card_Fraud_Notebook.ipynb"
ROC_NAME = "roc_curves.png"
ZIP_NAME = "Credit_Card_Fraud_Full_Package.zip"

# Notebook creation
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
nb = nbf.v4.new_notebook()
nb['metadata'] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"}
}

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(f"# Credit Card Fraud Detection — Completed Notebook\n\n**Generated:** {now}\n\nThis notebook runs EDA, preprocessing, training, and evaluation (Logistic Regression, RandomForest, optional XGBoost) on the `creditcard.csv` dataset. It includes code, explanations, and plots.\n\n---"))

# 1. Imports & load
cells.append(nbf.v4.new_markdown_cell("## 1. Imports and load dataset\nLoad the CSV file (place `creditcard.csv` in the same folder)."))
cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils import resample

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception:
    xgb_available = False

df = pd.read_csv('creditcard.csv')
print('Dataset shape:', df.shape)
print('Target distribution:')
print(df['Class'].value_counts())
df.head()"""
))

# 2. EDA
cells.append(nbf.v4.new_markdown_cell("## 2. Quick EDA"))
cells.append(nbf.v4.new_code_cell(
"""# Basic stats
print(df[['Time','Amount']].describe())

# Correlation with target
corrs = df.corr()['Class'].abs().sort_values(ascending=False)
print('\\nTop correlations with Class:')
print(corrs.head(15))"""
))

# 3. Preprocessing
cells.append(nbf.v4.new_markdown_cell("## 3. Preprocessing\n- Drop `Time`\n- Scale `Amount`\n- Train/test split (25% test, stratified)"))
cells.append(nbf.v4.new_code_cell(
"""# Preprocessing
X = df.drop(columns=['Class','Time'])
scaler = StandardScaler()
X['Amount_scaled'] = scaler.fit_transform(X['Amount'].values.reshape(-1,1))
X = X.drop(columns=['Amount'])
y = df['Class'].copy()

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)
print('Train class distribution:\\n', y_train.value_counts())"""
))

# 4. Handle imbalance + models
cells.append(nbf.v4.new_markdown_cell("## 4. Handle class imbalance and train models\nWe will: (A) train LogisticRegression with class_weight='balanced' on full data; (B) undersample majority class for RandomForest; (C) optionally train XGBoost with scale_pos_weight."))
cells.append(nbf.v4.new_code_cell(
"""# Create undersampled balanced training set (minority * 4)
train_df = pd.concat([X_train, y_train], axis=1)
fraud = train_df[train_df['Class'] == 1]
not_fraud = train_df[train_df['Class'] == 0]
n_down = max(len(fraud)*4, 1)
not_fraud_down = resample(not_fraud, replace=False, n_samples=n_down, random_state=42)
balanced = pd.concat([fraud, not_fraud_down]).sample(frac=1, random_state=42)
X_train_bal = balanced.drop(columns=['Class'])
y_train_bal = balanced['Class']

print('Balanced train shape:', X_train_bal.shape)
print('Balanced class counts:\\n', y_train_bal.value_counts())

# Logistic Regression - trained on original (class_weight)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=42)
lr.fit(X_train, y_train)

# RandomForest - trained on undersampled balanced set
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)

# XGBoost (optional)
xgb_model = None
if xgb_available:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42, n_estimators=200)
    xgb_model.fit(X_train, y_train)
    print('Trained XGBoost.')
else:
    print('XGBoost not available in this environment; skip.')"""
))

# 5. Evaluation
cells.append(nbf.v4.new_markdown_cell("## 5. Evaluation\nEvaluate on the test set: AUC, classification report, confusion matrix, and ROC curves."))
cells.append(nbf.v4.new_code_cell(
"""import matplotlib.pyplot as plt
models = {'LogisticRegression': lr, 'RandomForest': rf}
if xgb_model is not None:
    models['XGBoost'] = xgb_model

results = {}
plt.figure(figsize=(8,6))
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        try:
            y_proba = model.decision_function(X_test)
            y_proba = 1/(1+np.exp(-y_proba))
        except Exception:
            y_proba = np.zeros(len(y_test))
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    print('\\n---', name, '---')
    print('AUC: {:.4f}'.format(auc))
    print(classification_report(y_test, y_pred, digits=4))
    print('Confusion matrix:\\n', confusion_matrix(y_test, y_pred))
    results[name] = {'auc': auc, 'y_proba': y_proba}
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f\"{name} (AUC={auc:.4f})\")

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.show()

# Save ROC plot to file
plt.savefig('roc_curves.png')
print('\\nSaved ROC plot to roc_curves.png')"""
))

# 6. Feature importance
cells.append(nbf.v4.new_markdown_cell("## 6. Feature importance (RandomForest)"))
cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print('Top 20 features by importance:') 
print(feat_imp.head(20))"""
))

# 7. Save a small inference snippet
cells.append(nbf.v4.new_markdown_cell("## 7. Quick inference example (useful for deployment)"))
cells.append(nbf.v4.new_code_cell(
"""# Example: prepare a small sample and predict probability of fraud
sample = X_test.sample(5, random_state=42)
probs = rf.predict_proba(sample)[:,1]
print('Sample prediction probabilities (RandomForest):')
print(probs)"""
))

# 8. Wrap up notes
cells.append(nbf.v4.new_markdown_cell(
"""---\n\n**Notes & next steps (suggested):**\n\n- Consider SMOTE or ADASYN instead of undersampling.\n- Do threshold tuning to maximize recall for fraud detection.\n- Use cross-validation and hyperparameter search (RandomizedSearchCV) for better models.\n- Persist trained models with `joblib.dump()` for deployment.\n\nGood luck — if you want, I can also provide a version that saves `joblib` model files and an inference script."""
))

nb['cells'] = cells

# Write notebook file
try:
    with open(NOTEBOOK_NAME, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Wrote notebook: {NOTEBOOK_NAME}")
except Exception as e:
    print("ERROR writing notebook:", e)
    print("You can still open the printed notebook by copying the notebook JSON content manually (ask me for the JSON).")
    # continue to create placeholder ROC and zip

# Create a simple ROC placeholder if matplotlib not available or saving failed in notebook runtime
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curves (placeholder)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROC_NAME)
    plt.close()
    print(f"Wrote ROC placeholder image: {ROC_NAME}")
except Exception as e:
    print("Could not create ROC image locally:", e)

# Package into zip
try:
    with zipfile.ZipFile(ZIP_NAME, 'w') as zf:
        if os.path.exists(NOTEBOOK_NAME):
            zf.write(NOTEBOOK_NAME, arcname=NOTEBOOK_NAME)
        if os.path.exists(ROC_NAME):
            zf.write(ROC_NAME, arcname=ROC_NAME)
    print(f"Created zip package: {ZIP_NAME}")
except Exception as e:
    print("ERROR creating zip package:", e)

print("\\nAll done. To open the notebook, run: jupyter notebook", NOTEBOOK_NAME)
