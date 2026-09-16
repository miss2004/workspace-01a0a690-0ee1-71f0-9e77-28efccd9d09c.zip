"""
Empirical Evaluation of DeBERTa-v3 on Disentangled Syntactic Robustness,
Adversarial Typo Invariance, and Confidence Calibration (ECE)
----------------------------------------------------------------------
Evaluates cross-encoder/nli-deberta-v3-small across three rigorous Research Questions:
- RQ1: Semantic Invariance under Syntactic Inversion & Negation
- RQ2: Typographical Perturbation & Subword Fragmentation Dynamics
- RQ3: Epistemic Uncertainty, Confidence Calibration & Temperature Scaling
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple


def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> Tuple[float, List[Dict]]:
    """
    Computes Expected Calibration Error (ECE):
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []
    
    n_samples = len(labels)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin.astype(float))
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            bin_details.append({
                "bin": f"({bin_lower:.2f}, {bin_upper:.2f}]",
                "count": int(np.sum(in_bin)),
                "avg_confidence": float(avg_confidence_in_bin),
                "accuracy": float(accuracy_in_bin),
                "calibration_gap": float(abs_diff)
            })
            
    return float(ece), bin_details


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    """
    Brier Score: Mean squared difference between predicted probabilities
    and one-hot true outcome: BS = (1/N) * sum_i sum_k (p_ik - y_ik)^2
    """
    one_hot = np.zeros_like(probs)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


class DeBERTaEvaluator:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.label_names = ["Contradiction", "Entailment", "Neutral"]

    def predict_pair(self, premise: str, hypothesis: str) -> Tuple[np.ndarray, int, float, List[str]]:
        """Inference for single premise-hypothesis pair."""
        tokens = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        token_strings = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            logits = self.model(**tokens).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])
        return probs, pred_label, confidence, token_strings

    def evaluate_rq1_syntactic_invariance(self, syntactic_pairs: List[Dict]) -> Dict:
        """
        RQ1: Semantic Invariance under Syntactic Inversion & Negation
        Measures:
        - Clean vs Perturbed label consistency rate
        - Mean probability variation for true class
        - Invariance preservation score
        """
        results = []
        consistent_count = 0
        prob_shifts = []

        print("\n" + "=" * 70)
        print(" [RQ1] SYNTACTIC DISENTANGLEMENT & INVARIANCE EXPERIMENT")
        print("=" * 70)

        for item in syntactic_pairs:
            true_label = item["label"]
            # Clean
            c_probs, c_pred, c_conf, _ = self.predict_pair(item["premise_clean"], item["hypo_clean"])
            # Perturbed (syntactically rearranged)
            p_probs, p_pred, p_conf, _ = self.predict_pair(item["premise_perturbed"], item["hypo_perturbed"])

            is_consistent = (c_pred == p_pred == true_label)
            if is_consistent:
                consistent_count += 1

            shift = abs(c_probs[true_label] - p_probs[true_label])
            prob_shifts.append(shift)

            results.append({
                "id": item["id"],
                "category": item["category"],
                "true_label": self.label_names[true_label],
                "clean_pred": self.label_names[c_pred],
                "clean_conf": float(c_conf),
                "perturbed_pred": self.label_names[p_pred],
                "perturbed_conf": float(p_conf),
                "shift": float(shift),
                "label_preserved": bool(c_pred == p_pred)
            })
            print(f"[{item['id']}] {item['category']:<28} | Clean: {self.label_names[c_pred]} ({c_conf:.3f}) -> Perturbed: {self.label_names[p_pred]} ({p_conf:.3f}) | Shift: {shift:.3f}")

        invariance_rate = consistent_count / len(syntactic_pairs)
        mean_shift = float(np.mean(prob_shifts))
        print(f"\n>> RQ1 Synthesis: Invariance Consistency Rate = {invariance_rate * 100:.1f}%, Mean Prob Shift = {mean_shift:.4f}")
        return {
            "invariance_rate": invariance_rate,
            "mean_prob_shift": mean_shift,
            "details": results
        }

    def evaluate_rq2_typo_fragmentation(self, typo_pairs: List[Dict]) -> Dict:
        """
        RQ2: Typographical Perturbation & Subword Fragmentation Dynamics
        Measures:
        - Subword token expansion factor across severity levels (clean, level 1, level 2, level 3)
        - Accuracy drop across typo severity
        - Confidence degradation curve
        """
        print("\n" + "=" * 70)
        print(" [RQ2] ADVERSARIAL TYPO & SUBWORD FRAGMENTATION EXPERIMENT")
        print("=" * 70)

        severity_levels = ["clean_hypo", "typo_hypo_level_1", "typo_hypo_level_2", "typo_hypo_level_3"]
        summary = {level: {"correct": 0, "confidences": [], "token_counts": []} for level in severity_levels}

        for item in typo_pairs:
            true_label = item["label"]
            premise = item["clean_premise"]

            for level in severity_levels:
                hypo = item[level]
                probs, pred, conf, tokens = self.predict_pair(premise, hypo)
                is_correct = (pred == true_label)

                summary[level]["correct"] += int(is_correct)
                summary[level]["confidences"].append(conf)
                summary[level]["token_counts"].append(len(tokens))

        n_samples = len(typo_pairs)
        table = []
        for level in severity_levels:
            acc = summary[level]["correct"] / n_samples
            avg_conf = float(np.mean(summary[level]["confidences"]))
            avg_tokens = float(np.mean(summary[level]["token_counts"]))
            base_tokens = np.mean(summary["clean_hypo"]["token_counts"])
            expansion = avg_tokens / base_tokens
            row = {
                "level": level,
                "accuracy": acc,
                "avg_confidence": avg_conf,
                "avg_tokens": avg_tokens,
                "token_expansion_ratio": round(expansion, 2)
            }
            table.append(row)
            print(f"Severity [{level:<18}] -> Accuracy: {acc*100:>5.1f}% | Avg Conf: {avg_conf:.3f} | Tokens: {avg_tokens:.1f} (x{expansion:.2f})")

        return {"table": table}

    def evaluate_rq3_calibration_and_uncertainty(self, all_pairs: List[Tuple[str, str, int]]) -> Dict:
        """
        RQ3: Confidence Calibration & Epistemic Uncertainty
        Computes ECE, Brier Score, and tests Temperature Scaling.
        """
        print("\n" + "=" * 70)
        print(" [RQ3] CONFIDENCE CALIBRATION & TEMPERATURE SCALING EXPERIMENT")
        print("=" * 70)

        all_logits = []
        all_labels = []

        for p, h, y in all_pairs:
            tokens = self.tokenizer(p, h, return_tensors="pt", truncation=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                logits = self.model(**tokens).logits.cpu().numpy()[0]
            all_logits.append(logits)
            all_labels.append(y)

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)

        # Baseline probabilities (T=1.0)
        exp_l = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
        probs_raw = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

        ece_raw, raw_bins = calculate_ece(probs_raw, all_labels)
        brier_raw = compute_brier_score(probs_raw, all_labels)

        print(f"Raw Model (T = 1.00) : ECE = {ece_raw:.4f} | Brier Score = {brier_raw:.4f}")

        # Post-hoc calibration via Temperature Scaling
        # Grid search optimal temperature T in [0.5, 3.0]
        best_ece = ece_raw
        best_t = 1.0
        for t_cand in np.linspace(0.5, 3.0, 51):
            exp_t = np.exp(all_logits / t_cand - np.max(all_logits / t_cand, axis=-1, keepdims=True))
            probs_t = exp_t / np.sum(exp_t, axis=-1, keepdims=True)
            ece_t, _ = calculate_ece(probs_t, all_labels)
            if ece_t < best_ece:
                best_ece = ece_t
                best_t = t_cand

        exp_best = np.exp(all_logits / best_t - np.max(all_logits / best_t, axis=-1, keepdims=True))
        probs_calibrated = exp_best / np.sum(exp_best, axis=-1, keepdims=True)
        brier_calibrated = compute_brier_score(probs_calibrated, all_labels)

        print(f"Calibrated (T = {best_t:.2f}) : ECE = {best_ece:.4f} (Reduced by {(ece_raw - best_ece)/ece_raw*100:.1f}%) | Brier Score = {brier_calibrated:.4f}")

        return {
            "raw_ece": ece_raw,
            "raw_brier": brier_raw,
            "optimal_temperature": float(best_t),
            "calibrated_ece": float(best_ece),
            "calibrated_brier": float(brier_calibrated),
            "raw_bins": raw_bins
        }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_file = os.path.join(script_dir, "data", "evaluation_benchmarks.json")

    with open(bench_file, "r") as f:
        bench_data = json.load(f)

    evaluator = DeBERTaEvaluator()

    # RQ1
    rq1_res = evaluator.evaluate_rq1_syntactic_invariance(bench_data["syntactic_pairs"])

    # RQ2
    rq2_res = evaluator.evaluate_rq2_typo_fragmentation(bench_data["adversarial_typo_pairs"])

    # RQ3 Dataset preparation: combine syntactic clean, perturbed, typo variants, and OOD
    rq3_triplets = []
    for item in bench_data["syntactic_pairs"]:
        rq3_triplets.append((item["premise_clean"], item["hypo_clean"], item["label"]))
        rq3_triplets.append((item["premise_perturbed"], item["hypo_perturbed"], item["label"]))
    for item in bench_data["adversarial_typo_pairs"]:
        rq3_triplets.append((item["clean_premise"], item["clean_hypo"], item["label"]))
        rq3_triplets.append((item["clean_premise"], item["typo_hypo_level_1"], item["label"]))
        rq3_triplets.append((item["clean_premise"], item["typo_hypo_level_2"], item["label"]))
    for item in bench_data["ood_paradox_pairs"]:
        rq3_triplets.append((item["premise"], item["hypo"], item["label"]))

    rq3_res = evaluator.evaluate_rq3_calibration_and_uncertainty(rq3_triplets)

    # Save full empirical findings
    results_path = os.path.join(script_dir, "results", "empirical_evaluation_report.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "model": "cross-encoder/nli-deberta-v3-small",
            "RQ1_syntactic_invariance": rq1_res,
            "RQ2_typo_fragmentation": rq2_res,
            "RQ3_calibration": rq3_res
        }, f, indent=2)
    print(f"\n[+] Comprehensive evaluation findings written to: {results_path}")


if __name__ == "__main__":
    main()
