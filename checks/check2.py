# save as check2.py
import json

with open('data/spider/train_spider.json') as f:
    train = json.load(f)
with open('data/spider/dev.json') as f:
    dev = json.load(f)

from collections import Counter
train_counts = Counter(x['db_id'] for x in train)
dev_counts = Counter(x['db_id'] for x in dev)

# combine
all_counts = {}
for db_id in set(list(train_counts.keys()) + list(dev_counts.keys())):
    all_counts[db_id] = train_counts.get(db_id, 0) + dev_counts.get(db_id, 0)

# sort by count
sorted_dbs = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)

print("Top 30 databases by NL-SQL pair count:")
for db_id, count in sorted_dbs[:30]:
    print(f"  {db_id}: {count}")