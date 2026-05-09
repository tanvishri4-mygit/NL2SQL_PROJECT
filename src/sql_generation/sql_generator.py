"""
CFG-Based SQL Generator
Agent 06: Generates SQL queries from tagged slots using Context-Free Grammar rules.

Satisfies Syllabus Module 4: Context-Free Grammars, Syntax Trees, Chunking

Grammar Rules (simplified SQL CFG):
  QUERY       → SELECT_CLAUSE FROM_CLAUSE [WHERE_CLAUSE] [GROUP_CLAUSE] [ORDER_CLAUSE] [LIMIT_CLAUSE]
  SELECT_CLAUSE → "SELECT" COLUMN_LIST | "SELECT" AGG_EXPR
  FROM_CLAUSE → "FROM" TABLE_NAME [JOIN_CLAUSE]*
  WHERE_CLAUSE → "WHERE" CONDITION [LOGIC_OP CONDITION]*
  GROUP_CLAUSE → "GROUP BY" COLUMN_LIST
  ORDER_CLAUSE → "ORDER BY" COLUMN_REF SORT_DIR
  LIMIT_CLAUSE → "LIMIT" NUMBER
  AGG_EXPR    → AGG_FUNC "(" COLUMN_REF ")"
  CONDITION   → COLUMN_REF OP VALUE
"""

import os, sys, re
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from retrieval.schema_retriever import SchemaRetriever


class SQLGenerator:
    """
    Context-Free Grammar based SQL Generator.
    Constructs SQL from structured slots extracted by the SlotTagger.
    """
    
    def __init__(self, schema_retriever=None):
        self.retriever = schema_retriever or SchemaRetriever()
        
        # Column resolution cache
        self._column_cache = {}
        self._build_column_map()
    
    def _build_column_map(self):
        """Build a map of column names to their fully qualified references."""
        self.column_map = {}  # col_name -> [(table_name, col_name)]
        self.table_columns = {}  # table_name -> [col_names]
        
        for table in self.retriever.tables:
            tname = table["name"]
            cols = [c["name"] for c in table["columns"]]
            self.table_columns[tname] = cols
            for col in table["columns"]:
                cname = col["name"]
                if cname not in self.column_map:
                    self.column_map[cname] = []
                self.column_map[cname].append((tname, cname))
                for syn in col.get("synonyms", []):
                    s = syn.lower()
                    if s not in self.column_map:
                        self.column_map[s] = []
                    self.column_map[s].append((tname, cname))
    
    def _resolve_column(self, col_info: dict, context_tables: list) -> str:
        """Resolve a column reference to table.column format."""
        table = col_info.get("table")
        column = col_info.get("column")
        
        if table and column:
            return f"{table}.{column}"
        
        if column:
            # Find which table this column belongs to
            candidates = self.column_map.get(column.lower(), [])
            for tname, cname in candidates:
                if tname in context_tables:
                    return f"{tname}.{cname}"
            if candidates:
                return f"{candidates[0][0]}.{candidates[0][1]}"
        
        return column or "*"
    
    def _resolve_table_for_value(self, value_meta: dict, context_tables: list) -> tuple:
        """Find which table/column a value belongs to."""
        value = value_meta.get("value", "")
        vtype = value_meta.get("value_type", "")
        
        if vtype == "city":
            return ("addresses", "city")
        elif vtype == "brand":
            return ("products", "brand")
        elif vtype == "year":
            # Could be order_date, date_joined, etc.
            for t in context_tables:
                if t == "orders":
                    return ("orders", "order_date")
                elif t == "customers":
                    return ("customers", "date_joined")
                elif t == "payments":
                    return ("payments", "payment_date")
                elif t == "returns":
                    return ("returns", "return_date")
                elif t == "shipments":
                    return ("shipments", "shipped_date")
            return ("orders", "order_date")
        elif vtype == "month":
            for t in context_tables:
                if t == "orders":
                    return ("orders", "order_date")
            return ("orders", "order_date")
        elif vtype == "enum":
            val_str = str(value).lower()
            # Status matching
            order_statuses = {"delivered","cancelled","placed","packed","shipped","returned"}
            payment_statuses = {"success","failed","pending","refunded"}
            payment_methods = {"upi","card","cod","netbanking","wallet"}
            shipping_statuses = {"shipped","intransit","delivered","lost","returnedtoseller"}
            return_reasons = {"damaged","wrongitem","notneeded","delayed","other"}
            return_statuses = {"requested","approved","pickedup","refunded","rejected"}
            loyalty_tiers = {"bronze","silver","gold","platinum"}
            genders = {"male","female","other"}
            
            if val_str in order_statuses:
                return ("orders", "order_status")
            elif val_str in payment_statuses:
                return ("payments", "payment_status")
            elif val_str in payment_methods:
                return ("payments", "payment_method")
            elif val_str in shipping_statuses:
                return ("shipments", "shipping_status")
            elif val_str in return_reasons:
                return ("returns", "return_reason")
            elif val_str in return_statuses:
                return ("returns", "return_status")
            elif val_str in loyalty_tiers:
                return ("customers", "loyalty_tier")
            elif val_str in genders:
                return ("customers", "gender")
        
        return (None, None)
    
    def generate(self, intent: str, slots: dict, retrieved_context: dict) -> dict:
        """
        Generate SQL from intent + slots + retrieved context.
        
        Returns dict with:
        - sql: the SQL string
        - explanation: human-readable explanation
        - confidence: confidence score
        - parse_tree: syntax tree representation
        """
        # Determine tables needed
        tables = list(slots.get("tables", []))
        
        # Add tables from retrieved context
        for tinfo in retrieved_context.get("retrieved_tables", []):
            if tinfo["score"] > 0.1 and tinfo["table"] not in tables:
                tables.append(tinfo["table"])
        
        # Add tables from column references
        for col in slots.get("columns", []) + slots.get("group_by", []) + slots.get("order_by", []):
            t = col.get("table")
            if t and t not in tables:
                tables.append(t)
        
        # Add tables from values
        for val in slots.get("values", []):
            t, c = self._resolve_table_for_value(val, tables)
            if t and t not in tables:
                tables.append(t)
        
        # Ensure at least one table
        if not tables:
            tables = [retrieved_context["retrieved_tables"][0]["table"]] if retrieved_context.get("retrieved_tables") else ["orders"]
        
        primary_table = tables[0]
        
        # Route to specific generator based on intent
        if intent == "SELECT_SIMPLE":
            return self._gen_select_simple(slots, tables, primary_table, retrieved_context)
        elif intent == "FILTER_WHERE":
            return self._gen_filter_where(slots, tables, primary_table, retrieved_context)
        elif intent == "AGGREGATE":
            return self._gen_aggregate(slots, tables, primary_table, retrieved_context)
        elif intent == "GROUP_BY":
            return self._gen_group_by(slots, tables, primary_table, retrieved_context)
        elif intent == "ORDER_LIMIT":
            return self._gen_order_limit(slots, tables, primary_table, retrieved_context)
        elif intent == "JOIN_QUERY":
            return self._gen_join_query(slots, tables, primary_table, retrieved_context)
        else:
            return self._gen_select_simple(slots, tables, primary_table, retrieved_context)
    
    def _build_where_clause(self, slots, tables):
        """Build WHERE clause from values and operators."""
        conditions = []
        values = slots.get("values", [])
        operators = slots.get("operators", [])
        
        for i, val_meta in enumerate(values):
            tbl, col = self._resolve_table_for_value(val_meta, tables)
            if not tbl or not col:
                continue
            
            op = operators[i] if i < len(operators) else "="
            value = val_meta.get("value", "")
            vtype = val_meta.get("value_type", "")
            
            if vtype == "year":
                # Use strftime for SQLite date comparison
                conditions.append(f"strftime('%Y', {tbl}.{col}) = '{value}'")
            elif vtype == "month":
                conditions.append(f"strftime('%m', {tbl}.{col}) = '{value}'")
            elif vtype == "number":
                conditions.append(f"{tbl}.{col} {op} {value}")
            elif vtype in ("enum", "city", "brand"):
                conditions.append(f"{tbl}.{col} = '{value}'")
            else:
                conditions.append(f"{tbl}.{col} = '{value}'")
        
        return " AND ".join(conditions) if conditions else ""
    
    def _build_join_clause(self, tables):
        """Build JOIN clauses for multiple tables."""
        if len(tables) <= 1:
            return "", tables[0] if tables else "orders"
        
        join_conditions = self.retriever.find_join_path(tables)
        
        if not join_conditions:
            return "", tables[0]
        
        primary = tables[0]
        joined_tables = {primary}
        join_parts = []
        
        for jc in join_conditions:
            # Parse "table1.col = table2.col"
            parts = jc.split("=")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                left_table = left.split(".")[0]
                right_table = right.split(".")[0]
                
                if left_table not in joined_tables:
                    join_parts.append(f"JOIN {left_table} ON {jc.strip()}")
                    joined_tables.add(left_table)
                elif right_table not in joined_tables:
                    join_parts.append(f"JOIN {right_table} ON {jc.strip()}")
                    joined_tables.add(right_table)
        
        join_str = " ".join(join_parts)
        return join_str, primary
    
    def _gen_select_simple(self, slots, tables, primary, ctx):
        """Generate simple SELECT query."""
        where = self._build_where_clause(slots, tables)
        join_str, from_table = self._build_join_clause(tables)
        
        cols = slots.get("columns", [])
        if cols:
            select_cols = ", ".join(self._resolve_column(c, tables) for c in cols)
        else:
            select_cols = f"{primary}.*"
        
        sql = f"SELECT {select_cols} FROM {from_table}"
        if join_str:
            sql += f" {join_str}"
        if where:
            sql += f" WHERE {where}"
        sql += " LIMIT 50"
        
        return self._result(sql, slots, "Simple SELECT query", 0.8)
    
    def _gen_filter_where(self, slots, tables, primary, ctx):
        """Generate SELECT with WHERE filters."""
        where = self._build_where_clause(slots, tables)
        join_str, from_table = self._build_join_clause(tables)
        
        cols = slots.get("columns", [])
        if cols:
            select_cols = ", ".join(self._resolve_column(c, tables) for c in cols)
        else:
            select_cols = f"{primary}.*"
        
        sql = f"SELECT {select_cols} FROM {from_table}"
        if join_str:
            sql += f" {join_str}"
        if where:
            sql += f" WHERE {where}"
        else:
            sql += " WHERE 1=1"
        sql += " LIMIT 50"
        
        return self._result(sql, slots, "Filtered SELECT query", 0.75)
    
    def _gen_aggregate(self, slots, tables, primary, ctx):
        """Generate aggregate query (COUNT/SUM/AVG/MIN/MAX)."""
        agg_funcs = slots.get("agg_functions", ["COUNT"])
        agg = agg_funcs[0] if agg_funcs else "COUNT"
        
        where = self._build_where_clause(slots, tables)
        join_str, from_table = self._build_join_clause(tables)
        
        # Determine which column to aggregate
        cols = slots.get("columns", [])
        if cols:
            agg_col = self._resolve_column(cols[0], tables)
        elif agg == "COUNT":
            agg_col = "*"
        else:
            # Default to a numeric column
            if primary == "orders":
                agg_col = "orders.order_total"
            elif primary == "products":
                agg_col = "products.unit_price"
            elif primary == "payments":
                agg_col = "payments.amount_paid"
            elif primary == "returns":
                agg_col = "returns.refund_amount"
            elif primary == "shipments":
                agg_col = "shipments.shipping_fee"
            elif primary == "order_items":
                agg_col = "order_items.line_total"
            else:
                agg_col = "*"
        
        select_expr = f"{agg}({agg_col}) AS result"
        sql = f"SELECT {select_expr} FROM {from_table}"
        if join_str:
            sql += f" {join_str}"
        if where:
            sql += f" WHERE {where}"
        
        return self._result(sql, slots, f"Aggregate query: {agg}", 0.85)
    
    def _gen_group_by(self, slots, tables, primary, ctx):
        """Generate GROUP BY query."""
        agg_funcs = slots.get("agg_functions", ["COUNT"])
        agg = agg_funcs[0] if agg_funcs else "COUNT"
        
        where = self._build_where_clause(slots, tables)
        join_str, from_table = self._build_join_clause(tables)
        
        # Determine GROUP BY column
        group_cols = slots.get("group_by", [])
        if not group_cols:
            # Use last COLUMN tag as group by
            group_cols = slots.get("columns", [])[-1:] if slots.get("columns") else []
        
        if group_cols:
            group_col = self._resolve_column(group_cols[0], tables)
        else:
            group_col = f"{primary}.{self.table_columns.get(primary, ['*'])[0]}"
        
        # Aggregate column
        agg_cols = slots.get("columns", [])
        if len(agg_cols) > 0 and agg_cols[0] != group_cols[0] if group_cols else True:
            agg_col = self._resolve_column(agg_cols[0], tables)
        else:
            if agg == "COUNT":
                agg_col = "*"
            elif primary == "orders":
                agg_col = "orders.order_total"
            elif primary == "payments":
                agg_col = "payments.amount_paid"
            elif primary == "order_items":
                agg_col = "order_items.line_total"
            else:
                agg_col = "*"
        
        select_expr = f"{group_col}, {agg}({agg_col}) AS result"
        sql = f"SELECT {select_expr} FROM {from_table}"
        if join_str:
            sql += f" {join_str}"
        if where:
            sql += f" WHERE {where}"
        sql += f" GROUP BY {group_col}"
        sql += f" ORDER BY result DESC"
        
        return self._result(sql, slots, f"Group by query: {agg} grouped by {group_col}", 0.75)
    
    def _gen_order_limit(self, slots, tables, primary, ctx):
        """Generate ORDER BY + LIMIT query."""
        agg_funcs = slots.get("agg_functions", [])
        agg = agg_funcs[0] if agg_funcs else None
        
        where = self._build_where_clause(slots, tables)
        join_str, from_table = self._build_join_clause(tables)
        limit = slots.get("limit", 10)
        sort_dir = slots.get("sort_direction", "DESC")
        
        cols = slots.get("columns", [])
        order_cols = slots.get("order_by", [])
        
        if agg and order_cols:
            # e.g., "top 5 customers by total spending"
            group_col = self._resolve_column(cols[0], tables) if cols else f"{primary}.*"
            order_col = self._resolve_column(order_cols[0], tables) if order_cols else group_col
            select_expr = f"{group_col}, {agg}({order_col}) AS result"
            sql = f"SELECT {select_expr} FROM {from_table}"
            if join_str:
                sql += f" {join_str}"
            if where:
                sql += f" WHERE {where}"
            sql += f" GROUP BY {group_col}"
            sql += f" ORDER BY result {sort_dir}"
        elif order_cols:
            order_col = self._resolve_column(order_cols[0], tables)
            select_cols = f"{primary}.*"
            sql = f"SELECT {select_cols} FROM {from_table}"
            if join_str:
                sql += f" {join_str}"
            if where:
                sql += f" WHERE {where}"
            sql += f" ORDER BY {order_col} {sort_dir}"
        else:
            # Default ordering
            select_cols = f"{primary}.*"
            order_col = f"{primary}.{self.table_columns.get(primary, ['rowid'])[0]}"
            sql = f"SELECT {select_cols} FROM {from_table}"
            if join_str:
                sql += f" {join_str}"
            if where:
                sql += f" WHERE {where}"
            sql += f" ORDER BY {order_col} {sort_dir}"
        
        sql += f" LIMIT {limit}"
        
        return self._result(sql, slots, f"Ordered query: top {limit}", 0.7)
    
    def _gen_join_query(self, slots, tables, primary, ctx):
        """Generate JOIN query."""
        where = self._build_where_clause(slots, tables)
        join_str, from_table = self._build_join_clause(tables)
        
        cols = slots.get("columns", [])
        if cols:
            select_cols = ", ".join(self._resolve_column(c, tables) for c in cols)
        else:
            # Select key columns from each table
            select_parts = []
            for t in tables[:3]:
                t_cols = self.table_columns.get(t, [])
                for c in t_cols[:3]:
                    select_parts.append(f"{t}.{c}")
            select_cols = ", ".join(select_parts) if select_parts else "*"
        
        sql = f"SELECT {select_cols} FROM {from_table}"
        if join_str:
            sql += f" {join_str}"
        if where:
            sql += f" WHERE {where}"
        sql += " LIMIT 50"
        
        return self._result(sql, slots, "Join query across tables", 0.65)
    
    def _result(self, sql, slots, explanation, confidence):
        """Build result dict."""
        # Build parse tree representation
        parse_tree = self._build_parse_tree(sql)
        
        return {
            "sql": sql,
            "explanation": explanation,
            "confidence": confidence,
            "parse_tree": parse_tree,
            "slots_used": slots,
        }
    
    def _build_parse_tree(self, sql: str) -> dict:
        """
        Build a simplified syntax tree from the SQL string.
        Satisfies syllabus: Syntax Trees requirement.
        """
        tree = {"type": "QUERY", "children": []}
        
        # Parse SELECT
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE)
        if select_match:
            tree["children"].append({
                "type": "SELECT_CLAUSE",
                "value": select_match.group(1).strip()
            })
        
        # Parse FROM
        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if from_match:
            tree["children"].append({
                "type": "FROM_CLAUSE",
                "value": from_match.group(1).strip()
            })
        
        # Parse JOINs
        join_matches = re.finditer(r'JOIN\s+(\w+)\s+ON\s+(.+?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT|$))', sql, re.IGNORECASE)
        for m in join_matches:
            tree["children"].append({
                "type": "JOIN_CLAUSE",
                "table": m.group(1),
                "condition": m.group(2).strip()
            })
        
        # Parse WHERE
        where_match = re.search(r'WHERE\s+(.+?)(?=\s+(?:GROUP|ORDER|LIMIT|$))', sql, re.IGNORECASE)
        if where_match:
            tree["children"].append({
                "type": "WHERE_CLAUSE",
                "value": where_match.group(1).strip()
            })
        
        # Parse GROUP BY
        group_match = re.search(r'GROUP\s+BY\s+(.+?)(?=\s+(?:HAVING|ORDER|LIMIT|$))', sql, re.IGNORECASE)
        if group_match:
            tree["children"].append({
                "type": "GROUP_BY_CLAUSE",
                "value": group_match.group(1).strip()
            })
        
        # Parse ORDER BY
        order_match = re.search(r'ORDER\s+BY\s+(.+?)(?=\s+LIMIT|$)', sql, re.IGNORECASE)
        if order_match:
            tree["children"].append({
                "type": "ORDER_BY_CLAUSE",
                "value": order_match.group(1).strip()
            })
        
        # Parse LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
        if limit_match:
            tree["children"].append({
                "type": "LIMIT_CLAUSE",
                "value": int(limit_match.group(1))
            })
        
        return tree


if __name__ == "__main__":
    from slot_tagging.slot_tagger import SlotTagger
    from classification.intent_classifier import IntentClassificationPipeline
    
    retriever = SchemaRetriever()
    tagger = SlotTagger(retriever)
    generator = SQLGenerator(retriever)
    
    pipeline = IntentClassificationPipeline()
    pipeline.train()
    
    queries = [
        "show all customers",
        "count total orders in 2025",
        "top 5 products by rating",
        "average order total by city",
        "show orders where total greater than 5000",
        "total revenue by category",
    ]
    
    for q in queries:
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        
        intent_result = pipeline.predict(q)
        intent = intent_result["intent"]
        
        context = retriever.get_retrieval_context(q)
        tagged = tagger.tag(q, context)
        slots = tagger.get_slots(tagged)
        
        result = generator.generate(intent, slots, context)
        print(f"Intent: {intent} ({intent_result['confidence']:.0%})")
        print(f"SQL: {result['sql']}")
        print(f"Explanation: {result['explanation']}")
