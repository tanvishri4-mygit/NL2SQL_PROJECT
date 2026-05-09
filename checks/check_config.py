# add to checks/check_config.py
import sys
sys.path.insert(0, '.')
from config import *

print(f"BASE_DIR       : {BASE_DIR}")
print(f"DATA_DIR       : {DATA_DIR}")
print(f"WIKISQL_DIR    : {WIKISQL_DIR}")
print(f"SPIDER_DIR     : {SPIDER_DIR}")
print(f"CHINOOK_DB     : {CHINOOK_DB_PATH}")
print(f"WIKISQL exists : {WIKISQL_DIR.exists()}")
print(f"SPIDER exists  : {SPIDER_DIR.exists()}")
print(f"CHINOOK exists : {CHINOOK_DB_PATH.exists()}")
print(f"ENCODER_TYPE   : {ENCODER_TYPE}")