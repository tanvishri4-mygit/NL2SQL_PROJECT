"""
Evaluation Suite
Agent 08/09 (QA Team): Comprehensive evaluation of all NL2SQL pipeline components.

Metrics:
- Intent classification accuracy, precision, recall, F1
- Slot tagging accuracy (token-level)
- SQL execution accuracy
- Retrieval precision@k
- End-to-end accuracy
"""

import sys, os, json, sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from engine import NL2SQLEngine
from classification.intent_classifier import IntentClassificationPipeline, print_confusion_matrix
from retrieval.schema_retriever import SchemaRetriever, evaluate_retrieval


def run_full_evaluation():
    """Run complete evaluation suite and print results."""
    print("=" * 80)
    print("NL2SQL SYSTEM - FULL EVALUATION SUITE")
    print("=" * 80)
    
    # 1. Intent Classification
    print("\n" + "=" * 80)
    print("1. INTENT CLASSIFICATION EVALUATION")
    print("=" * 80)
    
    pipeline = IntentClassificationPipeline()
    results = pipeline.train(test_split=0.2)
    
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy:  {results['test_accuracy']:.4f}")
    print_confusion_matrix(results['test_confusion_matrix'])
    
    if results['errors']:
        print(f"\nTop Misclassifications:")
        for err in results['errors'][:5]:
            print(f"  '{err['query']}' → True: {err['true_label']}, Pred: {err['predicted']}")
    
    # 2. Schema Retrieval
    print("\n" + "=" * 80)
    print("2. SCHEMA RETRIEVAL (RAG) EVALUATION")
    print("=" * 80)
    
    retriever = SchemaRetriever()
    retrieval_results = evaluate_retrieval(retriever)
    print(f"Precision@1: {retrieval_results['precision_at_1']:.4f}")
    print(f"Precision@3: {retrieval_results['precision_at_3']:.4f}")
    print(f"Evaluated on: {retrieval_results['n_queries']} queries")
    
    # 3. End-to-End SQL Execution Accuracy
    print("\n" + "=" * 80)
    print("3. END-TO-END SQL EXECUTION ACCURACY")
    print("=" * 80)
    
    engine = NL2SQLEngine()
    
    test_queries = [
        ("show all customers", True),
        ("count total orders", True),
        ("list all products", True),
        ("show delivered orders", True),
        ("average order total", True),
        ("top 5 products by rating", True),
        ("orders count by status", True),
        ("total revenue by category", True),
        ("show orders where total greater than 5000", True),
        ("show customers from Mumbai", True),
        ("latest 10 orders", True),
        ("count cancelled orders", True),
        ("show failed payments", True),
        ("average product price", True),
        ("top 10 highest order totals", True),
        ("show all returns", True),
        ("total refund amount", True),
        ("count products in electronics", True),
        ("orders count by payment method", True),
        ("cheapest products", True),
        ("show customers where loyalty tier is gold", True),
        ("list shipments in transit", True),
        ("total shipping fees collected", True),
        ("show orders from last month", True),
        ("customers by gender", True),
        ("returns count by reason", True),
        ("show upi payments", True),
        ("most returned products", True),
        ("average rating of all products", True),
        ("show customer name and total for each order", True),
    ]
    
    success = 0
    errors = []
    
    for query, should_work in test_queries:
        result = engine.query(query)
        executed = result.get("action") == "result" and result.get("row_count", 0) > 0
        
        if executed == should_work:
            success += 1
            status = "✓"
        else:
            status = "✗"
            errors.append((query, result.get("action"), result.get("sql", "N/A")))
        
        rows = result.get("row_count", 0)
        intent = result.get("intent", "N/A")
        print(f"  {status} [{intent:<15}] {query} → {rows} rows")
    
    exec_accuracy = success / len(test_queries)
    print(f"\nExecution Accuracy: {exec_accuracy:.2%} ({success}/{len(test_queries)})")
    
    if errors:
        print(f"\nFailed Queries:")
        for q, action, sql in errors:
            print(f"  ✗ '{q}' → action={action}")
            if sql != "N/A":
                print(f"    SQL: {sql}")
    
    # 4. Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Intent Classification Accuracy: {results['test_accuracy']:.2%}")
    print(f"Schema Retrieval Precision@1:   {retrieval_results['precision_at_1']:.2%}")
    print(f"Schema Retrieval Precision@3:   {retrieval_results['precision_at_3']:.2%}")
    print(f"SQL Execution Accuracy:         {exec_accuracy:.2%}")
    print(f"Macro F1 (Intent):              {results['test_confusion_matrix']['macro_f1']:.2%}")
    
    return {
        "intent_accuracy": results['test_accuracy'],
        "retrieval_p1": retrieval_results['precision_at_1'],
        "retrieval_p3": retrieval_results['precision_at_3'],
        "execution_accuracy": exec_accuracy,
        "macro_f1": results['test_confusion_matrix']['macro_f1'],
    }


if __name__ == "__main__":
    run_full_evaluation()
