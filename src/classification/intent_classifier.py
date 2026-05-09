"""
Intent Classifier using Naive Bayes + TF-IDF
Agent 06 (Service Developer): Implements Naive Bayes classification for
query intent detection with full evaluation metrics.

Satisfies Syllabus Module 3: Text classification, sentiment analysis
- Naive Bayes Classifier, Confusion Matrix, TF-IDF
"""

import os
import sys
import json
import math
import random
import pickle
from collections import Counter, defaultdict

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.preprocessor import tokenize, remove_stopwords, lemmatize_tokens
from classification.intent_dataset import INTENT_DATASET, TFIDFVectorizer, INTENT_LABELS

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Naive Bayes Classifier (Multinomial, from scratch)
# ============================================================
class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes Classifier implemented from scratch.
    
    P(class|features) ∝ P(class) * ∏ P(feature_i|class)
    
    With Laplace smoothing:
    P(feature_i|class) = (count(feature_i, class) + alpha) / (total_features_in_class + alpha * |V|)
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.classes = []
        self.n_features = 0
    
    def fit(self, X: list, y: list):
        """
        Train the classifier.
        X: list of feature vectors (list of lists)
        y: list of class labels
        """
        self.classes = sorted(set(y))
        self.n_features = len(X[0]) if X else 0
        n_samples = len(y)
        
        # Compute class priors P(class)
        class_counts = Counter(y)
        self.class_priors = {c: math.log(count / n_samples) for c, count in class_counts.items()}
        
        # Compute feature likelihoods P(feature|class)
        self.feature_likelihoods = {}
        
        for cls in self.classes:
            # Get all samples of this class
            class_vectors = [X[i] for i in range(len(y)) if y[i] == cls]
            
            # Sum features across all documents of this class
            feature_sums = [0.0] * self.n_features
            total_sum = 0.0
            
            for vec in class_vectors:
                for j in range(self.n_features):
                    # Use absolute values for TF-IDF (can be negative after normalization)
                    val = max(vec[j], 0)
                    feature_sums[j] += val
                    total_sum += val
            
            # Compute log-likelihoods with Laplace smoothing
            denom = total_sum + self.alpha * self.n_features
            self.feature_likelihoods[cls] = [
                math.log((feature_sums[j] + self.alpha) / denom)
                for j in range(self.n_features)
            ]
        
        return self
    
    def predict_one(self, x: list) -> tuple:
        """Predict class for a single sample. Returns (class, log_probability)."""
        best_class = None
        best_score = float('-inf')
        scores = {}
        
        for cls in self.classes:
            score = self.class_priors[cls]
            for j in range(self.n_features):
                if x[j] > 0:
                    score += self.feature_likelihoods[cls][j] * x[j]
            scores[cls] = score
            if score > best_score:
                best_score = score
                best_class = cls
        
        # Convert to probabilities using log-sum-exp
        max_score = max(scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in scores.items()}
        total = sum(exp_scores.values())
        probabilities = {c: exp_s / total for c, exp_s in exp_scores.items()}
        
        return best_class, probabilities
    
    def predict(self, X: list) -> list:
        """Predict classes for multiple samples."""
        return [self.predict_one(x)[0] for x in X]
    
    def predict_proba(self, X: list) -> list:
        """Predict class probabilities for multiple samples."""
        return [self.predict_one(x)[1] for x in X]


# ============================================================
# Evaluation Metrics
# ============================================================
def confusion_matrix(y_true: list, y_pred: list, labels: list = None) -> dict:
    """
    Compute confusion matrix.
    Returns dict with matrix and per-class metrics.
    """
    if labels is None:
        labels = sorted(set(y_true + y_pred))
    
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * n for _ in range(n)]
    
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            matrix[label_to_idx[true]][label_to_idx[pred]] += 1
    
    # Per-class metrics
    metrics = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n)) - tp
        fn = sum(matrix[i][j] for j in range(n)) - tp
        tn = sum(sum(row) for row in matrix) - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }
    
    # Overall accuracy
    correct = sum(matrix[i][i] for i in range(n))
    total = sum(sum(row) for row in matrix)
    accuracy = correct / total if total > 0 else 0.0
    
    # Macro averages
    macro_precision = sum(m["precision"] for m in metrics.values()) / n
    macro_recall = sum(m["recall"] for m in metrics.values()) / n
    macro_f1 = sum(m["f1"] for m in metrics.values()) / n
    
    return {
        "matrix": matrix,
        "labels": labels,
        "per_class": metrics,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
    }


def print_confusion_matrix(cm_result: dict):
    """Pretty print confusion matrix."""
    labels = cm_result["labels"]
    matrix = cm_result["matrix"]
    
    # Shorten labels for display
    short = {l: l[:12] for l in labels}
    
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    
    # Header
    actual_pred = 'Actual\\Pred'
    header = f"{actual_pred:<15}" + "".join(f"{short[l]:>13}" for l in labels)
    print(header)
    print("-" * len(header))
    
    for i, label in enumerate(labels):
        row = f"{short[label]:<15}" + "".join(f"{matrix[i][j]:>13}" for j in range(len(labels)))
        print(row)
    
    print("\n" + "=" * 80)
    print("PER-CLASS METRICS")
    print("=" * 80)
    print(f"{'Class':<18} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)
    
    for label in labels:
        m = cm_result["per_class"][label]
        print(f"{label:<18} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
    
    print("-" * 60)
    print(f"{'MACRO AVG':<18} {cm_result['macro_precision']:>10.4f} {cm_result['macro_recall']:>10.4f} {cm_result['macro_f1']:>10.4f}")
    print(f"\nOverall Accuracy: {cm_result['accuracy']:.4f}")


# ============================================================
# Training Pipeline
# ============================================================
class IntentClassificationPipeline:
    """
    Complete intent classification pipeline:
    Preprocessing → TF-IDF → Naive Bayes → Evaluation
    """
    
    def __init__(self):
        self.vectorizer = TFIDFVectorizer(max_features=500)
        self.classifier = NaiveBayesClassifier(alpha=1.0)
        self.is_trained = False
    
    def _preprocess_query(self, query: str) -> list:
        """Preprocess a single query into tokens."""
        tokens = tokenize(query)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize_tokens(tokens)
        return tokens
    
    def train(self, test_split=0.2, random_state=42):
        """
        Train the pipeline on the intent dataset.
        Returns evaluation results on test set.
        """
        random.seed(random_state)
        
        # Prepare data
        data = list(INTENT_DATASET)
        random.shuffle(data)
        
        split_idx = int(len(data) * (1 - test_split))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        # Tokenize
        train_tokens = [self._preprocess_query(q) for q, _, _, _ in train_data]
        test_tokens = [self._preprocess_query(q) for q, _, _, _ in test_data]
        
        train_labels = [label for _, label, _, _ in train_data]
        test_labels = [label for _, label, _, _ in test_data]
        
        # TF-IDF
        X_train = self.vectorizer.fit_transform(train_tokens)
        X_test = self.vectorizer.transform(test_tokens)
        
        # Train Naive Bayes
        self.classifier.fit(X_train, train_labels)
        self.is_trained = True
        
        # Evaluate
        train_preds = self.classifier.predict(X_train)
        test_preds = self.classifier.predict(X_test)
        
        train_cm = confusion_matrix(train_labels, train_preds, INTENT_LABELS)
        test_cm = confusion_matrix(test_labels, test_preds, INTENT_LABELS)
        
        # Error analysis
        errors = []
        for i, (true, pred) in enumerate(zip(test_labels, test_preds)):
            if true != pred:
                query = test_data[i][0]
                probs = self.classifier.predict_one(X_test[i])[1]
                errors.append({
                    "query": query,
                    "true_label": true,
                    "predicted": pred,
                    "confidence": round(probs.get(pred, 0), 4),
                    "true_prob": round(probs.get(true, 0), 4),
                })
        
        results = {
            "train_accuracy": train_cm["accuracy"],
            "test_accuracy": test_cm["accuracy"],
            "test_confusion_matrix": test_cm,
            "train_confusion_matrix": train_cm,
            "errors": errors,
            "n_train": len(train_data),
            "n_test": len(test_data),
        }
        
        return results
    
    def predict(self, query: str) -> dict:
        """Predict intent for a single query."""
        if not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")
        
        tokens = self._preprocess_query(query)
        vec = self.vectorizer.transform([tokens])
        label, probs = self.classifier.predict_one(vec[0])
        
        # Sort probabilities
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "intent": label,
            "confidence": round(probs[label], 4),
            "all_probabilities": {k: round(v, 4) for k, v in sorted_probs},
            "tokens": tokens,
        }
    
    def save(self, path=None):
        """Save trained model."""
        if path is None:
            path = os.path.join(MODEL_DIR, "intent_model.pkl")
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path=None):
        """Load trained model."""
        if path is None:
            path = os.path.join(MODEL_DIR, "intent_model.pkl")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
            self.is_trained = True
        print(f"Model loaded from {path}")


# ============================================================
# Main - Train and Evaluate
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("INTENT CLASSIFICATION PIPELINE - Training & Evaluation")
    print("=" * 80)
    
    pipeline = IntentClassificationPipeline()
    results = pipeline.train(test_split=0.2)
    
    print(f"\nTraining samples: {results['n_train']}")
    print(f"Test samples: {results['n_test']}")
    print(f"\nTrain Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy:  {results['test_accuracy']:.4f}")
    
    print_confusion_matrix(results['test_confusion_matrix'])
    
    if results['errors']:
        print(f"\n{'='*80}")
        print(f"ERROR ANALYSIS ({len(results['errors'])} misclassifications)")
        print(f"{'='*80}")
        for err in results['errors'][:10]:
            print(f"  Query: \"{err['query']}\"")
            print(f"  True: {err['true_label']} | Pred: {err['predicted']} (conf: {err['confidence']})")
            print()
    
    # Save model
    pipeline.save()
    
    # Demo predictions
    print("\n" + "=" * 80)
    print("DEMO PREDICTIONS")
    print("=" * 80)
    test_queries = [
        "show me all customers",
        "how many orders were placed last month",
        "top 5 products by sales",
        "revenue by category",
        "show customer name with order details",
        "orders where total is more than 10000",
    ]
    for q in test_queries:
        result = pipeline.predict(q)
        print(f"  \"{q}\"")
        print(f"    → Intent: {result['intent']} (confidence: {result['confidence']:.2%})")
        print()
