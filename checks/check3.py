# save as check3.py
import sqlite3, json

# check store_1 schema
conn = sqlite3.connect('data/spider/database/store_1/store_1.sqlite')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in c.fetchall()]
print("store_1 TABLES:", tables)
for t in tables:
    c.execute(f'PRAGMA table_info({t})')
    cols = [col[1] for col in c.fetchall()]
    print(f'  {t}: {cols}')
conn.close()

print()

# check sample store_1 pairs
with open('data/spider/train_spider.json') as f:
    train = json.load(f)
with open('data/spider/dev.json') as f:
    dev = json.load(f)

store_pairs = [x for x in train+dev if x['db_id'] == 'store_1']
print(f"store_1 pairs: {len(store_pairs)}")
print("Samples:")
for x in store_pairs[:5]:
    print(f"  Q: {x['question']}")
    print(f"  SQL: {x['query']}")
    print()