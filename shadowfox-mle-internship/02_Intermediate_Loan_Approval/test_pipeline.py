"""
Inference Smoke Tests for Loan Approval Pipeline
------------------------------------------------
Tests model loading and inference on synthetic credit profiles:
1. Low-risk applicant (High income, flawless credit history) -> Should approve
2. High-risk applicant (No credit history, very high debt ratio) -> Should deny
"""

import os
import unittest
import joblib
import pandas as pd


class TestLoanPipelineInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(os.path.dirname(__file__), "models", "best_loan_model.joblib")
        cls.artifact = joblib.load(model_path)
        cls.pipeline = cls.artifact["pipeline"]
        cls.threshold = cls.artifact["optimal_threshold"]

    def test_prime_borrower_approval(self):
        """Borrower with solid credit history and low debt ratio."""
        applicant = pd.DataFrame([{
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 7500.0,
            "CoapplicantIncome": 2500.0,
            "LoanAmount": 120.0,
            "Loan_Amount_Term": 360.0,
            "Credit_History": 1.0,
            "Property_Area": "Urban",
            "Total_Income": 10000.0,
            "Loan_to_Income_Ratio": 12.0,
            "EMI_Estimate": 333.33,
            "Debt_Burden_Ratio": 0.40,
            "Has_Coapplicant": 1
        }])
        prob = self.pipeline.predict_proba(applicant)[0, 1]
        decision = int(prob >= self.threshold)
        print(f"Prime Borrower Approval Prob: {prob:.4f} (Threshold: {self.threshold:.2f}) -> {'Approved' if decision else 'Denied'}")
        self.assertEqual(decision, 1, "Prime applicant should be approved")

    def test_subprime_borrower_rejection(self):
        """Borrower with failed credit history (0.0) and high debt burden."""
        applicant = pd.DataFrame([{
            "Gender": "Male",
            "Married": "No",
            "Dependents": "2",
            "Education": "Not Graduate",
            "Self_Employed": "Yes",
            "ApplicantIncome": 1800.0,
            "CoapplicantIncome": 0.0,
            "LoanAmount": 250.0,
            "Loan_Amount_Term": 180.0,
            "Credit_History": 0.0,
            "Property_Area": "Rural",
            "Total_Income": 1800.0,
            "Loan_to_Income_Ratio": 138.8,
            "EMI_Estimate": 1388.8,
            "Debt_Burden_Ratio": 9.2,
            "Has_Coapplicant": 0
        }])
        prob = self.pipeline.predict_proba(applicant)[0, 1]
        decision = int(prob >= self.threshold)
        print(f"Subprime Borrower Approval Prob: {prob:.4f} (Threshold: {self.threshold:.2f}) -> {'Approved' if decision else 'Denied'}")
        self.assertEqual(decision, 0, "Subprime applicant with poor credit history should be rejected")


if __name__ == "__main__":
    unittest.main()
