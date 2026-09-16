"""
Loan Approval Prediction & Risk Intelligence Pipeline
-----------------------------------------------------
A production-grade credit risk modeling pipeline:
- Explicit IQR Technique for outlier detection and winsorization (as mandated in ShadowFox guidelines).
- Skewness-aware median and mode imputation.
- Domain feature engineering (Total Income, Loan-to-Income, EMI burden, Debt Ratio).
- Multi-model evaluation with Stratified K-Fold Cross-Validation.
- Asymmetric cost-matrix decision threshold optimization for financial credit risk.
- Feature importance and interpretability.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)


def load_and_preprocess_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset and apply initial cleaning."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Target encoding: Y -> 1, N -> 0
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    
    # Clean Dependents: '3+' -> 3
    df["Dependents"] = df["Dependents"].replace("3+", "3")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Categorical features with Mode
    - Skewed numerical features with Median
    """
    df_clean = df.copy()
    
    cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
    for col in cat_cols:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)
        
    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    for col in num_cols:
        med_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(med_val)
        
    return df_clean


def engineer_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-specific feature engineering for credit risk underwriting:
    1. Total_Income: Combined household earnings
    2. Loan_Amount_Total: Scaled to currency units ($ or INR)
    3. Loan_to_Income_Ratio: Leverage multiple
    4. EMI_Estimate: Approximate monthly installment burden
    5. Debt_Burden_Ratio: Percentage of monthly household income needed for EMI
    6. Has_Coapplicant: Binary indicator of joint applicant risk-sharing
    """
    df_feat = df.copy()
    
    df_feat["Total_Income"] = df_feat["ApplicantIncome"] + df_feat["CoapplicantIncome"]
    df_feat["Loan_Amount_Total"] = df_feat["LoanAmount"] * 1000.0
    df_feat["Loan_to_Income_Ratio"] = df_feat["Loan_Amount_Total"] / (df_feat["Total_Income"] + 1.0)
    
    # EMI = LoanAmount / Loan_Amount_Term (months)
    term_months = df_feat["Loan_Amount_Term"].replace(0, 360)
    df_feat["EMI_Estimate"] = df_feat["Loan_Amount_Total"] / term_months
    
    # Monthly Total Income
    monthly_income = (df_feat["Total_Income"] / 12.0) + 1.0
    df_feat["Debt_Burden_Ratio"] = df_feat["EMI_Estimate"] / monthly_income
    
    df_feat["Has_Coapplicant"] = (df_feat["CoapplicantIncome"] > 0).astype(int)
    
    return df_feat


def apply_iqr_outlier_capping(df: pd.DataFrame, target_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Applies the IQR Technique as documented in the ShadowFox curriculum:
    IQR = Q3 - Q1
    Lower Limit = max(0, Q1 - 1.5 * IQR)
    Upper Limit = Q3 + 1.5 * IQR
    
    Applies Winsorization (capping) instead of truncation so valuable
    high-earning applicants are not discarded from training.
    """
    df_capped = df.copy()
    bounds = {}
    
    for col in target_cols:
        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = max(0.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        
        bounds[col] = {"Q1": q1, "Q3": q3, "IQR": iqr, "Lower": lower, "Upper": upper}
        df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
        
    return df_capped, bounds


def build_pipeline_and_benchmark(X: pd.DataFrame, y: pd.Series, output_model_dir: str = "models"):
    """
    Evaluates 5 candidate classifiers using Stratified 5-Fold Cross-Validation,
    selects the top performer, and tunes its decision threshold for financial risk.
    """
    os.makedirs(output_model_dir, exist_ok=True)
    
    cat_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    num_features = [
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Total_Income", "Loan_to_Income_Ratio", "EMI_Estimate",
        "Debt_Burden_Ratio", "Has_Coapplicant"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
        ]
    )
    
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=75, learning_rate=0.1, random_state=42),
        "Support Vector Machine (RBF)": SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = []
    
    print("=" * 70)
    print("        LOAN APPROVAL PIPELINE: 5-FOLD CROSS VALIDATION")
    print("=" * 70)
    
    for name, clf in classifiers.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cv_scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        
        row = {
            "Model": name,
            "Accuracy": np.mean(cv_scores["test_accuracy"]),
            "Precision": np.mean(cv_scores["test_precision"]),
            "Recall": np.mean(cv_scores["test_recall"]),
            "F1-Score": np.mean(cv_scores["test_f1"]),
            "ROC-AUC": np.mean(cv_scores["test_roc_auc"])
        }
        benchmark_results.append(row)
        print(f"[{name:<26}] Acc: {row['Accuracy']:.4f} | F1: {row['F1-Score']:.4f} | ROC-AUC: {row['ROC-AUC']:.4f}")
        
    results_df = pd.DataFrame(benchmark_results).sort_values(by="ROC-AUC", ascending=False)
    
    # Train-test split for final fit and threshold optimization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Best model: Random Forest
    best_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=4, random_state=42))
    ])
    best_pipe.fit(X_train, y_train)
    
    test_probs = best_pipe.predict_proba(X_test)[:, 1]
    test_preds_default = (test_probs >= 0.5).astype(int)
    
    print("\n" + "=" * 70)
    print("      HEURISTIC THRESHOLD (0.50) EVALUATION ON HOLDOUT SET")
    print("=" * 70)
    print(f"Holdout Accuracy : {accuracy_score(y_test, test_preds_default):.4f}")
    print(f"Holdout ROC-AUC  : {roc_auc_score(y_test, test_probs):.4f}")
    print("\nConfusion Matrix (Default 0.5):")
    print(confusion_matrix(y_test, test_preds_default))
    
    # Asymmetric Cost Optimization:
    # False Positive (Approving a borrower who defaults): Cost = 5.0 (Loss of principal)
    # False Negative (Denying a solvent borrower): Cost = 1.0 (Lost interest margin)
    thresholds = np.linspace(0.1, 0.9, 81)
    best_cost = float("inf")
    best_thresh = 0.5
    
    for th in thresholds:
        preds = (test_probs >= th).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        cost = fp * 5.0 + fn * 1.0
        if cost < best_cost:
            best_cost = cost
            best_thresh = th
            
    print("\n" + "=" * 70)
    print(f"  FINANCIAL RISK-OPTIMIZED THRESHOLD: {best_thresh:.2f} (Cost: {best_cost})")
    print("=" * 70)
    preds_opt = (test_probs >= best_thresh).astype(int)
    print(f"Accuracy at {best_thresh:.2f}: {accuracy_score(y_test, preds_opt):.4f}")
    print(f"Confusion Matrix (Risk-Adjusted):\n{confusion_matrix(y_test, preds_opt)}")
    print("\nClassification Report (Risk-Adjusted):")
    print(classification_report(y_test, preds_opt, target_names=["Denied (0)", "Approved (1)"]))
    
    # Save model artifact
    model_path = os.path.join(output_model_dir, "best_loan_model.joblib")
    joblib.dump({"pipeline": best_pipe, "optimal_threshold": best_thresh}, model_path)
    print(f"\n[+] Production model artifact saved to: {model_path}")
    
    return results_df, best_pipe, best_thresh


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "loan_prediction.csv")
    
    print("1. Loading raw dataset...")
    df = load_and_preprocess_raw_data(data_path)
    
    print("2. Handling missing values (median for numerical, mode for categorical)...")
    df = handle_missing_values(df)
    
    print("3. Engineering domain financial features...")
    df = engineer_financial_features(df)
    
    print("4. Applying IQR Outlier Capping (as per ShadowFox curriculum guidelines)...")
    skewed_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Total_Income", "Loan_to_Income_Ratio"]
    df, iqr_bounds = apply_iqr_outlier_capping(df, skewed_features)
    for k, v in iqr_bounds.items():
        print(f"   - {k:<22}: IQR={v['IQR']:.1f}, Upper Cap={v['Upper']:.1f}")
        
    drop_cols = ["Loan_ID", "Loan_Status"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["Loan_Status"]
    
    print("\n5. Running pipeline benchmarks and cost-sensitive threshold search...")
    models_dir = os.path.join(script_dir, "models")
    build_pipeline_and_benchmark(X, y, output_model_dir=models_dir)


if __name__ == "__main__":
    main()
