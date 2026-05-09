# save as check_env.py
import sys
import os

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")

packages = ["torch", "transformers", "pandas", 
            "nltk", "sklearn", "numpy", "streamlit"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'installed')
        print(f"  {pkg}: {ver}")
    except ImportError:
        print(f"  {pkg}: NOT INSTALLED")