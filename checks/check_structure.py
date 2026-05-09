# save as check_structure.py in nl2sql_project folder
import os

def show_tree(path, prefix="", ignore=None):
    if ignore is None:
        ignore = {'.git', '__pycache__', '.idea', 
                  'node_modules', '.venv', 'venv',
                  'anaconda3', '.ipynb_checkpoints'}
    
    entries = sorted(os.listdir(path))
    entries = [e for e in entries if e not in ignore]
    
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries)-1 else "├── "
        full_path = os.path.join(path, entry)
        print(prefix + connector + entry)
        if os.path.isdir(full_path):
            extension = "    " if i == len(entries)-1 else "│   "
            show_tree(full_path, prefix + extension, ignore)

print("nl2sql_project/")
show_tree(".")