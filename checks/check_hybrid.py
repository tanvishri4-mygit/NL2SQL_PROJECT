# save as checks/check_hybrid.py
import sys
sys.path.insert(0, '.')
from rules.sql_rules import RuleBasedSQL
import sqlite3

rb = RuleBasedSQL()
db = "data/spider/database/chinook_1/chinook_1.sqlite"

queries = [
    "how many artists are there",
    "show all customers from usa",
    "what is the total revenue",
    "show top 5 invoices by total",
    "count customers by country",
    "show tracks with their genre names",
    "find top 5 customers by spending",
]

print("="*60)
for q in queries:
    result = rb.try_generate(q)
    print(f"\nQ   : {q}")
    if result["matched"]:
        print(f"Rule: {result['rule']}")
        print(f"SQL : {result['sql']}")
        # execute
        try:
            conn = sqlite3.connect(db)
            cursor = conn.cursor()
            cursor.execute(result["sql"])
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            conn.close()
            print(f"Cols: {cols}")
            print(f"Rows: {len(rows)}  Sample: {rows[:2]}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Rule: NO MATCH → neural model")