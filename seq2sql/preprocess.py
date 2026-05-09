"""
preprocess.py
=============
Merges and cleans:
    1. WikiSQL  (56K pairs — general SQL grammar)
    2. Spider   (10K pairs — complex SQL across 166 databases)
       - includes store_1 (112 pairs) and chinook_1 (84 pairs)
         which are the SAME schema — gives 196 schema-specific pairs
    
All data is real human-annotated. Zero synthetic data.

Run from project root:
    python seq2sql/preprocess.py
"""

import pandas as pd
import json
import ast
import re
import sys
from pathlib import Path

# ensure project root in path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from config import (
    WIKISQL_DIR, SPIDER_DIR, PROCESSED_DIR,
    CHINOOK_SCHEMA_CONTEXT, STORE1_SCHEMA_CONTEXT
)

# ═══════════════════════════════════════════════════════
# Normalisation helpers
# ═══════════════════════════════════════════════════════

SQL_KEYWORDS = [
    "select", "from", "where", "group by", "order by", "having",
    "limit", "join", "inner join", "left join", "right join",
    "outer join", "on", "as", "count", "sum", "avg", "min", "max",
    "distinct", "and", "or", "not", "in", "like", "between",
    "is", "null", "asc", "desc", "union", "intersect", "except",
    "exists", "all", "case", "when", "then", "else", "end"
]

def normalise_sql(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        return ""
    sql = sql.strip().rstrip(";")
    for kw in sorted(SQL_KEYWORDS, key=len, reverse=True):
        sql = re.sub(r'\b' + re.escape(kw) + r'\b',
                     kw.upper(), sql, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', sql).strip()

def normalise_question(q: str) -> str:
    if not isinstance(q, str):
        return ""
    return re.sub(r'\s+', ' ', q.lower().strip()).strip('?.')

# ═══════════════════════════════════════════════════════
# WikiSQL
# ═══════════════════════════════════════════════════════

def parse_wikisql_sql(sql_str: str) -> str:
    try:
        d = ast.literal_eval(sql_str)
        return d.get("human_readable", "")
    except Exception:
        m = re.search(r"'human_readable':\s*'([^']+)'", str(sql_str))
        return m.group(1) if m else ""

def load_wikisql(split: str) -> pd.DataFrame:
    path = WIKISQL_DIR / f"{split}.csv"
    print(f"  WikiSQL {split} ... ", end="")
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        q   = normalise_question(str(row.get("question", "")))
        sql = normalise_sql(parse_wikisql_sql(str(row.get("sql", ""))))
        if not q or not sql or len(sql) < 5:
            continue
        records.append({
            "question"       : q,
            "sql"            : sql,
            "schema_context" : "",
            "source"         : "wikisql",
            "split"          : split,
            "db_id"          : "wikisql"
        })
    out = pd.DataFrame(records)
    print(f"{len(out)} rows")
    return out

# ═══════════════════════════════════════════════════════
# Spider
# ═══════════════════════════════════════════════════════

def load_spider_schemas() -> dict:
    path = SPIDER_DIR / "tables.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    schemas = {}
    for db in data:
        db_id       = db["db_id"]
        table_names = db.get("table_names_original", [])
        col_names   = [c[1] for c in db.get("column_names_original", [])
                       if c[0] >= 0]
        schemas[db_id] = (
            "tables : " + " | ".join(table_names) +
            " ; columns : " + " | ".join(col_names)
        )
    print(f"  Loaded schemas for {len(schemas)} Spider DBs")
    return schemas

def load_spider_file(filename: str, schemas: dict,
                     split: str) -> pd.DataFrame:
    path = SPIDER_DIR / filename
    print(f"  Spider {filename} ... ", end="")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for entry in data:
        q     = normalise_question(entry.get("question", ""))
        sql   = normalise_sql(entry.get("query", ""))
        db_id = entry.get("db_id", "")

        # use our detailed schema context for chinook + store_1
        if db_id == "chinook_1":
            schema = CHINOOK_SCHEMA_CONTEXT
        elif db_id == "store_1":
            schema = STORE1_SCHEMA_CONTEXT
        else:
            schema = schemas.get(db_id, "")

        if not q or not sql or len(sql) < 5:
            continue
        records.append({
            "question"       : q,
            "sql"            : sql,
            "schema_context" : schema,
            "source"         : "spider",
            "split"          : split,
            "db_id"          : db_id
        })
    out = pd.DataFrame(records)
    print(f"{len(out)} rows")
    return out

# ═══════════════════════════════════════════════════════
# Complexity filter
# ═══════════════════════════════════════════════════════

def filter_complexity(df: pd.DataFrame,
                      max_tokens: int = 80) -> pd.DataFrame:
    def ok(sql):
        if len(sql.split()) > max_tokens:
            return False
        if sql.count("SELECT") > 2:
            return False
        return True
    before = len(df)
    df = df[df["sql"].apply(ok)].reset_index(drop=True)
    print(f"  Complexity filter: {before} → {len(df)}")
    return df

# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "="*55)
    print("  NL2SQL Data Preprocessing")
    print("="*55)

    # ── WikiSQL ───────────────────────────────────────
    print("\n[1] WikiSQL")
    wik_train = load_wikisql("train")
    wik_val   = load_wikisql("validation")
    wik_test  = load_wikisql("test")

    # ── Spider ────────────────────────────────────────
    print("\n[2] Spider")
    schemas   = load_spider_schemas()
    sp_train1 = load_spider_file("train_spider.json", schemas, "train")
    sp_train2 = load_spider_file("train_others.json", schemas, "train")
    sp_dev    = load_spider_file("dev.json",           schemas, "validation")

    # ── Show chinook + store_1 counts ─────────────────
    all_spider = pd.concat([sp_train1, sp_train2, sp_dev])
    chinook_count = (all_spider["db_id"] == "chinook_1").sum()
    store1_count  = (all_spider["db_id"] == "store_1").sum()
    print(f"\n  chinook_1 pairs : {chinook_count}")
    print(f"  store_1   pairs : {store1_count}")
    print(f"  Combined (same schema): {chinook_count + store1_count}")

    # ── Merge ─────────────────────────────────────────
    print("\n[3] Merging")
    all_df = pd.concat([
        wik_train, wik_val, wik_test,
        sp_train1, sp_train2, sp_dev
    ], ignore_index=True)
    print(f"  Total before filter: {len(all_df)}")

    # ── Filter ────────────────────────────────────────
    print("\n[4] Filtering")
    all_df = all_df.dropna(subset=["question", "sql"])
    before = len(all_df)
    all_df = all_df.drop_duplicates(
        subset=["question", "sql"]).reset_index(drop=True)
    print(f"  Dedup: {before} → {len(all_df)}")
    all_df = filter_complexity(all_df)

    # ── Split ─────────────────────────────────────────
    print("\n[5] Splitting")
    train_df = all_df[all_df["split"] == "train"].reset_index(drop=True)
    val_df   = all_df[all_df["split"] == "validation"].reset_index(drop=True)
    test_df  = all_df[all_df["split"] == "test"].reset_index(drop=True)

    print(f"  Train : {len(train_df)}")
    print(f"  Val   : {len(val_df)}")
    print(f"  Test  : {len(test_df)}")

    # ── Source distribution ───────────────────────────
    print("\n[6] Source distribution in train:")
    print(train_df["source"].value_counts().to_string())

    # ── Save ─────────────────────────────────────────
    print("\n[7] Saving")
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR   / "val.csv",   index=False)
    test_df.to_csv(PROCESSED_DIR  / "test.csv",  index=False)
    all_df.to_csv(PROCESSED_DIR   / "all.csv",   index=False)
    print(f"  Saved to {PROCESSED_DIR}")

    print("\n✅ Preprocessing complete!")
    print("Next: python seq2sql/vocabulary.py")

if __name__ == "__main__":
    main()
