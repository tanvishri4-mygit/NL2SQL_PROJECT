"""
Conversational State Manager
Agent 06: Manages multi-turn dialogue state for follow-up queries.
Maintains context of last SQL, tables, filters, and enables SQL patching.

Example flow:
  User: "show sales in 2025"
  User: "only Mumbai"          → patches WHERE with city = Mumbai
  User: "group by category"    → adds GROUP BY
"""

import re
import copy


class ConversationState:
    """Tracks dialogue state across conversation turns."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset conversation state."""
        self.history = []  # List of (query, sql, result) tuples
        self.last_sql = None
        self.last_tables = []
        self.last_where = []
        self.last_group_by = None
        self.last_order_by = None
        self.last_limit = None
        self.last_intent = None
        self.last_columns = []
        self.turn_count = 0
    
    def update(self, query: str, sql: str, intent: str, tables: list, result_count: int):
        """Update state after a successful query."""
        self.turn_count += 1
        self.history.append({
            "turn": self.turn_count,
            "query": query,
            "sql": sql,
            "intent": intent,
            "result_count": result_count,
        })
        
        self.last_sql = sql
        self.last_intent = intent
        self.last_tables = tables
        
        # Parse SQL components
        self._parse_sql(sql)
    
    def _parse_sql(self, sql: str):
        """Parse SQL to extract reusable components."""
        # Extract WHERE conditions
        where_match = re.search(r'WHERE\s+(.+?)(?=\s+(?:GROUP|ORDER|LIMIT|$))', sql, re.IGNORECASE)
        if where_match:
            self.last_where = [where_match.group(1).strip()]
        else:
            self.last_where = []
        
        # Extract GROUP BY
        group_match = re.search(r'GROUP\s+BY\s+(.+?)(?=\s+(?:HAVING|ORDER|LIMIT|$))', sql, re.IGNORECASE)
        self.last_group_by = group_match.group(1).strip() if group_match else None
        
        # Extract ORDER BY
        order_match = re.search(r'ORDER\s+BY\s+(.+?)(?=\s+LIMIT|$)', sql, re.IGNORECASE)
        self.last_order_by = order_match.group(1).strip() if order_match else None
        
        # Extract LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
        self.last_limit = int(limit_match.group(1)) if limit_match else None
        
        # Extract SELECT columns
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE)
        self.last_columns = select_match.group(1).strip() if select_match else "*"
    
    def is_followup(self, query: str) -> bool:
        """Detect if query is a follow-up to previous turn."""
        if self.turn_count == 0 or self.last_sql is None:
            return False
        
        query_lower = query.lower().strip()
        
        # Short queries are likely follow-ups
        if len(query_lower.split()) <= 4:
            return True
        
        # Starts with modifiers
        followup_starters = [
            "only", "just", "also", "and", "but", "except",
            "group by", "sort by", "order by", "limit",
            "add", "remove", "change", "modify", "update",
            "what about", "how about", "instead",
        ]
        for starter in followup_starters:
            if query_lower.startswith(starter):
                return True
        
        return False
    
    def patch_sql(self, query: str, new_conditions: dict) -> str:
        """
        Patch the last SQL with new conditions from a follow-up query.
        
        new_conditions dict can contain:
        - add_where: list of new WHERE conditions
        - set_group_by: new GROUP BY column
        - set_order_by: new ORDER BY column
        - set_limit: new LIMIT value
        """
        if not self.last_sql:
            return None
        
        sql = self.last_sql
        
        # Add WHERE conditions
        if new_conditions.get("add_where"):
            for cond in new_conditions["add_where"]:
                if "WHERE" in sql.upper():
                    # Insert before GROUP/ORDER/LIMIT
                    sql = re.sub(
                        r'(WHERE\s+.+?)(\s+(?:GROUP|ORDER|LIMIT))',
                        rf'\1 AND {cond}\2',
                        sql, count=1, flags=re.IGNORECASE
                    )
                    if cond not in sql:
                        # Fallback: append before LIMIT
                        sql = re.sub(r'\s+LIMIT', f' AND {cond} LIMIT', sql, flags=re.IGNORECASE)
                else:
                    # Insert WHERE before GROUP/ORDER/LIMIT
                    for keyword in ['GROUP', 'ORDER', 'LIMIT']:
                        if keyword in sql.upper():
                            sql = re.sub(
                                rf'(\s+{keyword})',
                                f' WHERE {cond}\\1',
                                sql, count=1, flags=re.IGNORECASE
                            )
                            break
                    else:
                        sql += f" WHERE {cond}"
        
        # Set GROUP BY
        if new_conditions.get("set_group_by"):
            gb = new_conditions["set_group_by"]
            if "GROUP BY" in sql.upper():
                sql = re.sub(r'GROUP\s+BY\s+\S+', f'GROUP BY {gb}', sql, flags=re.IGNORECASE)
            else:
                # Insert before ORDER/LIMIT
                for keyword in ['ORDER', 'LIMIT']:
                    if keyword in sql.upper():
                        sql = re.sub(
                            rf'(\s+{keyword})',
                            f' GROUP BY {gb}\\1',
                            sql, count=1, flags=re.IGNORECASE
                        )
                        break
                else:
                    sql += f" GROUP BY {gb}"
        
        # Set ORDER BY
        if new_conditions.get("set_order_by"):
            ob = new_conditions["set_order_by"]
            if "ORDER BY" in sql.upper():
                sql = re.sub(r'ORDER\s+BY\s+.+?(?=\s+LIMIT|$)', f'ORDER BY {ob}', sql, flags=re.IGNORECASE)
            else:
                if "LIMIT" in sql.upper():
                    sql = re.sub(r'\s+LIMIT', f' ORDER BY {ob} LIMIT', sql, flags=re.IGNORECASE)
                else:
                    sql += f" ORDER BY {ob}"
        
        # Set LIMIT
        if new_conditions.get("set_limit") is not None:
            lim = new_conditions["set_limit"]
            if "LIMIT" in sql.upper():
                sql = re.sub(r'LIMIT\s+\d+', f'LIMIT {lim}', sql, flags=re.IGNORECASE)
            else:
                sql += f" LIMIT {lim}"
        
        return sql
    
    def get_context_summary(self) -> str:
        """Get a summary of current conversation context."""
        if self.turn_count == 0:
            return "No conversation history."
        
        last = self.history[-1]
        return (f"Turn {self.turn_count}: Last query was '{last['query']}' "
                f"({last['intent']}) returning {last['result_count']} rows.")


if __name__ == "__main__":
    state = ConversationState()
    
    # Simulate conversation
    state.update(
        "show orders in 2025",
        "SELECT orders.* FROM orders WHERE strftime('%Y', orders.order_date) = '2025' LIMIT 50",
        "SELECT_SIMPLE", ["orders"], 150
    )
    
    print(f"Context: {state.get_context_summary()}")
    print(f"Is 'only Mumbai' a followup? {state.is_followup('only Mumbai')}")
    
    patched = state.patch_sql("only Mumbai", {
        "add_where": ["addresses.city = 'Mumbai'"]
    })
    print(f"Patched SQL: {patched}")
    
    patched2 = state.patch_sql("group by status", {
        "set_group_by": "orders.order_status"
    })
    print(f"Patched SQL 2: {patched2}")
