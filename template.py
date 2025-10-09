import os
from pathlib import Path

# Define your package name here
package_name = "vectorDBpipe"

# Define the full list of files & structure for your AI/LLM pipeline package
list_of_files = [
    f"{package_name}/__init__.py",
    f"{package_name}/pipeline/__init__.py",
    f"{package_name}/pipeline/text_pipeline.py",
    f"{package_name}/config/__init__.py",
    f"{package_name}/config/config.yaml",
    f"{package_name}/utils/__init__.py",
    f"{package_name}/utils/common.py",
    f"{package_name}/data/__init__.py",
    f"{package_name}/data/loader.py",
    f"{package_name}/embeddings/__init__.py",
    f"{package_name}/embeddings/embedder.py",
    f"{package_name}/vectordb/__init__.py",
    f"{package_name}/vectordb/store.py",
    f"{package_name}/logger/__init__.py",
    f"{package_name}/logger/logging.py",
    f"{package_name}/tests/__init__.py",
    f"{package_name}/tests/test_pipeline.py",
    "setup.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
]

# Create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # create an empty file

print(f"✅ Project structure for '{package_name}' created successfully!")
