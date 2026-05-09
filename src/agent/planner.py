"""
Agentic Planning Module
Agent 06: Implements lightweight agentic reasoning loop.
- Decides: generate SQL / ask clarification / repair / retrieve more schema
- Handles ambiguity, errors, and iterative refinement
"""

import os, sys, sqlite3, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DB_PATH = os.path.join(DATA_DIR, "ecommerce.db")


class AgentPlanner:
    """
    Lightweight agentic planner that orchestrates the NL2SQL pipeline.
    
    Decision loop:
    1. Classify intent → if low confidence → ask clarification
    2. Retrieve schema context
    3. Tag slots
    4. Generate SQL
    5. Execute SQL → if error → repair and retry
    6. Return result
    """
    
    def __init__(self, intent_pipeline, slot_tagger, sql_generator, schema_retriever):
        self.intent_pipeline = intent_pipeline
        self.slot_tagger = slot_tagger
        self.sql_generator = sql_generator
        self.retriever = schema_retriever
        self.db_path = DB_PATH
        
        # Thresholds
        self.CONFIDENCE_THRESHOLD = 0.20
        self.MAX_REPAIR_ATTEMPTS = 3
    
    def process_query(self, query: str, conversation_state: dict = None) -> dict:
        """
        Main entry point: process a natural language query.
        
        Returns dict with:
        - action: "result" | "clarification" | "error"
        - sql: generated SQL
        - data: query results (list of dicts)
        - columns: column names
        - explanation: human-readable explanation
        - confidence: confidence score
        - debug: debugging info (preprocessing, slots, etc.)
        """
        debug = {"steps": []}
        
        # Step 1: Intent Classification
        intent_result = self.intent_pipeline.predict(query)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        debug["steps"].append(f"Intent: {intent} (confidence: {confidence:.2%})")
        debug["intent"] = intent_result
        
        # Decision: Low confidence → ask clarification
        if confidence < self.CONFIDENCE_THRESHOLD:
            top_2 = list(intent_result["all_probabilities"].items())[:2]
            return {
                "action": "clarification",
                "message": f"I'm not sure what you're asking. Did you mean:\n"
                           f"1) {self._intent_description(top_2[0][0])}\n"
                           f"2) {self._intent_description(top_2[1][0])}\n"
                           f"Could you rephrase your question?",
                "confidence": confidence,
                "debug": debug,
            }
        
        # Step 2: Retrieve schema context (RAG)
        context = self.retriever.get_retrieval_context(query, top_k=5)
        debug["steps"].append(f"Retrieved tables: {[t['table'] for t in context['retrieved_tables'][:3]]}")
        debug["retrieved_context"] = {
            "tables": [{"table": t["table"], "score": t["score"]} for t in context["retrieved_tables"][:5]]
        }
        
        # Step 3: Slot tagging
        tagged = self.slot_tagger.tag(query, context)
        slots = self.slot_tagger.get_slots(tagged)
        debug["steps"].append(f"Slots: tables={slots['tables']}, agg={slots['agg_functions']}, values={len(slots['values'])}")
        debug["tagged_sequence"] = [(t, tag, meta) for t, tag, meta in tagged]
        debug["slots"] = slots
        
        # Step 4: Generate SQL
        result = self.sql_generator.generate(intent, slots, context)
        sql = result["sql"]
        debug["steps"].append(f"Generated SQL: {sql}")
        debug["parse_tree"] = result.get("parse_tree", {})
        
        # Step 5: Execute with repair loop
        data, columns, exec_error = self._execute_sql(sql)
        
        if exec_error:
            debug["steps"].append(f"Execution error: {exec_error}")
            # Attempt repair
            for attempt in range(self.MAX_REPAIR_ATTEMPTS):
                repaired_sql = self._repair_sql(sql, exec_error, context, slots)
                if repaired_sql == sql:
                    break
                debug["steps"].append(f"Repair attempt {attempt+1}: {repaired_sql}")
                sql = repaired_sql
                data, columns, exec_error = self._execute_sql(sql)
                if not exec_error:
                    debug["steps"].append("Repair successful!")
                    break
        
        if exec_error:
            return {
                "action": "error",
                "message": f"I generated a query but encountered an error: {exec_error}\n"
                           f"Generated SQL: {sql}\n"
                           f"Could you try rephrasing your question?",
                "sql": sql,
                "confidence": result["confidence"],
                "debug": debug,
            }
        
        # Step 6: Return result
        explanation = self._build_explanation(query, intent, result, len(data))
        
        return {
            "action": "result",
            "sql": sql,
            "data": data,
            "columns": columns,
            "row_count": len(data),
            "explanation": explanation,
            "confidence": result["confidence"],
            "intent": intent,
            "debug": debug,
        }
    
    def _execute_sql(self, sql: str) -> tuple:
        """Execute SQL and return (data, columns, error)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            
            if rows:
                columns = list(rows[0].keys())
                data = [dict(row) for row in rows]
            else:
                columns = []
                data = []
            
            conn.close()
            return data, columns, None
        except Exception as e:
            return [], [], str(e)
    
    def _repair_sql(self, sql: str, error: str, context: dict, slots: dict) -> str:
        """Attempt to repair a broken SQL query."""
        error_lower = error.lower()
        
        # Fix: no such table
        table_match = re.search(r'no such table: (\w+)', error_lower)
        if table_match:
            bad_table = table_match.group(1)
            # Try to find the correct table
            for tinfo in context.get("retrieved_tables", []):
                if tinfo["table"] != bad_table:
                    sql = sql.replace(bad_table, tinfo["table"])
                    return sql
        
        # Fix: no such column
        col_match = re.search(r'no such column: (\S+)', error_lower)
        if col_match:
            bad_col = col_match.group(1)
            # Remove the problematic column reference
            if "." in bad_col:
                table, col = bad_col.split(".", 1)
                # Check if column exists in table
                schema = self.retriever.get_table_schema(table)
                if schema:
                    valid_cols = [c["name"] for c in schema["columns"]]
                    # Find closest match
                    for vc in valid_cols:
                        if col in vc or vc in col:
                            sql = sql.replace(bad_col, f"{table}.{vc}")
                            return sql
            
            # Fallback: replace with *
            sql = re.sub(r'SELECT\s+.+?\s+FROM', 'SELECT * FROM', sql, count=1, flags=re.IGNORECASE)
            return sql
        
        # Fix: ambiguous column name
        if "ambiguous column" in error_lower:
            # Add table prefix to all columns
            for tinfo in context.get("retrieved_tables", []):
                tname = tinfo["table"]
                schema = context.get("table_schemas", {}).get(tname, {})
                for col in schema.get("columns", []):
                    cname = col["name"]
                    # Only replace bare column names (not already qualified)
                    pattern = rf'\b(?<!\.)({cname})\b'
                    sql = re.sub(pattern, f"{tname}.{cname}", sql)
            return sql
        
        # Fix: near syntax errors
        if "near" in error_lower:
            # Remove trailing commas before FROM
            sql = re.sub(r',\s*FROM', ' FROM', sql)
            # Fix double spaces
            sql = re.sub(r'\s+', ' ', sql).strip()
            return sql
        
        return sql
    
    def _build_explanation(self, query, intent, result, row_count):
        """Build human-readable explanation of the query."""
        parts = [f"I interpreted your question as a {self._intent_description(intent).lower()}."]
        
        parse_tree = result.get("parse_tree", {})
        for child in parse_tree.get("children", []):
            ctype = child.get("type", "")
            if ctype == "WHERE_CLAUSE":
                parts.append(f"Filtered by: {child['value']}")
            elif ctype == "GROUP_BY_CLAUSE":
                parts.append(f"Grouped by: {child['value']}")
            elif ctype == "ORDER_BY_CLAUSE":
                parts.append(f"Sorted by: {child['value']}")
            elif ctype == "JOIN_CLAUSE":
                parts.append(f"Joined with: {child['table']}")
        
        parts.append(f"Found {row_count} result{'s' if row_count != 1 else ''}.")
        return " ".join(parts)
    
    def _intent_description(self, intent):
        """Human-readable intent description."""
        descriptions = {
            "SELECT_SIMPLE": "Simple data retrieval",
            "FILTER_WHERE": "Filtered search with conditions",
            "AGGREGATE": "Aggregate calculation (count/sum/average)",
            "GROUP_BY": "Grouped summary",
            "ORDER_LIMIT": "Ranked/sorted results",
            "JOIN_QUERY": "Cross-table lookup",
        }
        return descriptions.get(intent, intent)


if __name__ == "__main__":
    from classification.intent_classifier import IntentClassificationPipeline
    from slot_tagging.slot_tagger import SlotTagger
    from sql_generation.sql_generator import SQLGenerator
    from retrieval.schema_retriever import SchemaRetriever
    
    retriever = SchemaRetriever()
    tagger = SlotTagger(retriever)
    generator = SQLGenerator(retriever)
    
    pipeline = IntentClassificationPipeline()
    pipeline.train()
    
    agent = AgentPlanner(pipeline, tagger, generator, retriever)
    
    queries = [
        "show all customers from Mumbai",
        "count total orders",
        "top 5 products by rating",
        "total revenue by category",
        "show orders where total greater than 10000",
    ]
    
    for q in queries:
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        result = agent.process_query(q)
        print(f"Action: {result['action']}")
        if result['action'] == 'result':
            print(f"SQL: {result['sql']}")
            print(f"Rows: {result['row_count']}")
            if result['data']:
                print(f"Sample: {result['data'][0]}")
            print(f"Explanation: {result['explanation']}")
        else:
            print(f"Message: {result.get('message', '')}")
