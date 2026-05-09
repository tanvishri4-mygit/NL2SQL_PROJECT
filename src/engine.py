"""
NL2SQL Engine - Main Orchestrator
Agent 04 (Dev Tech Lead): Integrates all modules into a cohesive pipeline.

Pipeline:
  User Query → Preprocessing → Intent Classification → Schema Retrieval (RAG)
  → Slot Tagging → CFG SQL Generation → Agent Planning → Execution → Response
"""

import os, sys

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from preprocessing.preprocessor import NLPreprocessor
from classification.intent_classifier import IntentClassificationPipeline
from classification.intent_dataset import save_dataset
from retrieval.schema_retriever import SchemaRetriever
from slot_tagging.slot_tagger import SlotTagger
from sql_generation.sql_generator import SQLGenerator
from agent.planner import AgentPlanner
from conversation.state_manager import ConversationState


class NL2SQLEngine:
    """
    Main engine orchestrating the entire NL-to-SQL pipeline.
    Provides a clean interface for the UI layer.
    """
    
    def __init__(self):
        print("Initializing NL2SQL Engine...")
        
        # Initialize components
        self.preprocessor = NLPreprocessor()
        self.retriever = SchemaRetriever()
        self.tagger = SlotTagger(self.retriever)
        self.generator = SQLGenerator(self.retriever)
        
        # Train intent classifier
        self.intent_pipeline = IntentClassificationPipeline()
        train_results = self.intent_pipeline.train()
        print(f"  Intent classifier trained: accuracy={train_results['test_accuracy']:.2%}")
        
        # Initialize agent
        self.agent = AgentPlanner(
            self.intent_pipeline, self.tagger, self.generator, self.retriever
        )
        
        # Conversation state
        self.conversation = ConversationState()
        
        print("NL2SQL Engine ready!")
    
    def query(self, user_input: str) -> dict:
        """
        Process a natural language query and return results.
        
        Returns dict with:
        - action: "result" | "clarification" | "error"
        - sql: generated SQL query
        - data: list of result dicts
        - columns: column names
        - row_count: number of rows
        - explanation: natural language explanation
        - confidence: confidence score
        - intent: detected intent
        - preprocessing: preprocessing details
        - debug: full debug info
        """
        user_input = user_input.strip()
        if not user_input:
            return {"action": "error", "message": "Please enter a query."}
        
        # Check for follow-up
        is_followup = self.conversation.is_followup(user_input)
        
        # Preprocessing (for display)
        preprocessing = self.preprocessor.preprocess(user_input)
        
        if is_followup and self.conversation.last_sql:
            result = self._handle_followup(user_input, preprocessing)
        else:
            result = self.agent.process_query(user_input)
        
        # Update conversation state on success
        if result.get("action") == "result":
            self.conversation.update(
                user_input,
                result.get("sql", ""),
                result.get("intent", ""),
                result.get("debug", {}).get("slots", {}).get("tables", []),
                result.get("row_count", 0)
            )
        
        # Add preprocessing info
        result["preprocessing"] = {
            "tokens": preprocessing["tokens"],
            "tokens_clean": preprocessing["tokens_clean"],
            "pos_tags": [(t, tag) for t, tag in preprocessing["pos_tags"]],
            "lemmas": preprocessing["lemmas"],
            "bigrams": [" ".join(bg) for bg in preprocessing["bigrams"]],
        }
        
        result["is_followup"] = is_followup
        result["turn"] = self.conversation.turn_count
        
        return result
    
    def _handle_followup(self, query: str, preprocessing: dict) -> dict:
        """Handle a follow-up query by patching the previous SQL."""
        # Simple follow-up handling: detect what to add/change
        tokens = preprocessing["tokens_clean"]
        
        new_conditions = {}
        
        # Check for city filter: "only Mumbai"
        cities = {"mumbai","pune","delhi","bangalore","hyderabad","chennai",
                  "kolkata","ahmedabad","jaipur","lucknow","noida","gurgaon","surat","indore","nagpur"}
        for t in tokens:
            if t in cities:
                new_conditions.setdefault("add_where", []).append(
                    f"addresses.city = '{t.title()}'"
                )
        
        # Check for status filter
        statuses = {"delivered","cancelled","placed","packed","shipped","returned"}
        for t in tokens:
            if t in statuses:
                new_conditions.setdefault("add_where", []).append(
                    f"orders.order_status = '{t.capitalize()}'"
                )
        
        # Check for GROUP BY
        if "group" in tokens or "by" in tokens:
            group_col_map = {
                "category": "categories.category_name",
                "status": "orders.order_status",
                "city": "addresses.city",
                "month": "strftime('%Y-%m', orders.order_date)",
                "year": "strftime('%Y', orders.order_date)",
                "brand": "products.brand",
                "method": "payments.payment_method",
                "courier": "shipments.courier_name",
                "reason": "returns.return_reason",
                "tier": "customers.loyalty_tier",
                "gender": "customers.gender",
                "department": "categories.category_name",
                "product": "products.product_name",
                "customer": "customers.full_name",
            }
            for t in tokens:
                if t in group_col_map:
                    new_conditions["set_group_by"] = group_col_map[t]
                    break
        
        # Check for ORDER BY
        if "sort" in tokens or "order" in tokens:
            for t in tokens:
                if t in {"ascending", "asc"}:
                    new_conditions["set_order_by"] = (
                        self.conversation.last_order_by.replace("DESC", "ASC")
                        if self.conversation.last_order_by else "1 ASC"
                    )
                elif t in {"descending", "desc"}:
                    new_conditions["set_order_by"] = (
                        self.conversation.last_order_by.replace("ASC", "DESC")
                        if self.conversation.last_order_by else "1 DESC"
                    )
        
        # Check for LIMIT
        for t in tokens:
            if t.isdigit():
                n = int(t)
                if 1 <= n <= 1000:
                    new_conditions["set_limit"] = n
        
        if new_conditions:
            patched_sql = self.conversation.patch_sql(query, new_conditions)
            if patched_sql:
                # Execute patched SQL
                data, columns, error = self.agent._execute_sql(patched_sql)
                if not error:
                    return {
                        "action": "result",
                        "sql": patched_sql,
                        "data": data,
                        "columns": columns,
                        "row_count": len(data),
                        "explanation": f"Updated previous query with your modifications. Found {len(data)} results.",
                        "confidence": 0.7,
                        "intent": self.conversation.last_intent,
                        "debug": {"steps": [f"Follow-up: patched SQL with {new_conditions}"]},
                    }
        
        # Fallback: treat as new query
        return self.agent.process_query(query)
    
    def reset_conversation(self):
        """Reset conversation state."""
        self.conversation.reset()
        return {"action": "info", "message": "Conversation reset. Start a new query!"}
    
    def get_suggestions(self) -> list:
        """Get query suggestions for the user."""
        return [
            "Show all customers from Mumbai",
            "Count total orders in 2025",
            "Top 5 products by rating",
            "Total revenue by category",
            "Average order value by city",
            "Show cancelled orders with refund above 5000",
            "List payments by method",
            "Most returned products",
            "Show customer name and order total for delivered orders",
            "Latest 10 orders",
        ]
    
    def get_system_info(self) -> dict:
        """Get system information for display."""
        return {
            "name": "NL2SQL Engine v1.0",
            "title": "Schema-Aware Conversational Natural Language Interface for Relational Databases",
            "components": [
                "Classical NLP Preprocessing (Tokenization, POS Tagging, Lemmatization, N-grams)",
                "TF-IDF Feature Extraction & Count Vectorization",
                "Naive Bayes Intent Classifier with Confusion Matrix Evaluation",
                "Sequence Slot Tagger (SQL semantic role tagging)",
                "Context-Free Grammar based SQL Generation with Syntax Trees",
                "Retrieval-Augmented Schema Linking (RAG)",
                "Agentic Planning with Error Repair Loop",
                "Conversational State Management (Multi-turn)",
            ],
            "database": "E-Commerce (10 tables, 28K+ rows)",
            "supported_sql": [
                "SELECT with WHERE",
                "Aggregations (COUNT, SUM, AVG, MIN, MAX)",
                "GROUP BY",
                "ORDER BY + LIMIT (Top-K)",
                "JOINs (up to 2 tables)",
            ],
        }


# Singleton for Streamlit caching
_engine_instance = None

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = NL2SQLEngine()
    return _engine_instance


if __name__ == "__main__":
    engine = NL2SQLEngine()
    
    print("\n" + "="*70)
    print("NL2SQL Interactive Console")
    print("="*70)
    print("Type your query (or 'quit' to exit, 'reset' to clear history)")
    print("Suggestions:", engine.get_suggestions()[:5])
    
    while True:
        query = input("\n> ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if query.lower() == 'reset':
            engine.reset_conversation()
            print("Conversation reset.")
            continue
        
        result = engine.query(query)
        
        if result["action"] == "result":
            print(f"\nSQL: {result['sql']}")
            print(f"Intent: {result.get('intent', 'N/A')} | Confidence: {result.get('confidence', 0):.0%}")
            print(f"Results: {result['row_count']} rows")
            if result['data']:
                # Print table header
                cols = result['columns']
                print("\n" + " | ".join(f"{c:<20}" for c in cols[:6]))
                print("-" * (22 * min(len(cols), 6)))
                for row in result['data'][:5]:
                    print(" | ".join(f"{str(row.get(c, '')):<20}" for c in cols[:6]))
                if result['row_count'] > 5:
                    print(f"  ... and {result['row_count'] - 5} more rows")
            print(f"\n{result.get('explanation', '')}")
        elif result["action"] == "clarification":
            print(f"\n{result['message']}")
        else:
            print(f"\nError: {result.get('message', 'Unknown error')}")
