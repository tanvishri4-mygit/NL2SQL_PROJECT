import pandas as pd
import sqlite3
import json

# WikiSQL check
df = pd.read_csv('data/wikisql/train.csv')
print("WikiSQL columns:", df.columns.tolist())
print("WikiSQL rows:", len(df))
print()

# Chinook schema
conn = sqlite3.connect('data/spider/database/chinook_1/chinook_1.sqlite')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in c.fetchall()]
print("TABLES:", tables)
for t in tables:
    c.execute(f'PRAGMA table_info({t})')
    cols = [col[1] for col in c.fetchall()]
    print(f'{t}: {cols}')
conn.close()
print()

# Chinook pairs in Spider
with open('data/spider/train_spider.json') as f:
    train = json.load(f)
with open('data/spider/dev.json') as f:
    dev = json.load(f)

chinook_train = [x for x in train if x['db_id'] == 'chinook_1']
chinook_dev = [x for x in dev if x['db_id'] == 'chinook_1']
print(f'Chinook train pairs: {len(chinook_train)}')
print(f'Chinook dev pairs: {len(chinook_dev)}')
print(f'Total: {len(chinook_train) + len(chinook_dev)}')
print()
print('Sample pairs:')
for x in chinook_train[:3]:
    print(f'Q: {x["question"]}')
    print(f'SQL: {x["query"]}')
    print()