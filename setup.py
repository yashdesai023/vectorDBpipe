import os
from setuptools import setup, find_packages

setup(
    name="vectordbpipe",  # Corrected to lowercase
    version="0.1.6",      # Updated to a new version
    author="Yash Desai",
    author_email="desaisyash1000@gmail.com",
    description="A modular text embedding and vector database pipeline for local and cloud vector stores.",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "PyYAML>=6.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "transformers>=4.28.1",
        "torch>=2.2.0",  # Allow any version of torch from 2.2.0 onwards
        "torchvision",
        "chromadb>=0.4.22",
        "pinecone-client>=3.0.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "docx2txt>=0.8",
        "beautifulsoup4>=4.12.3",
        "PyMuPDF>=1.23.26"
    ],
    python_requires=">=3.8",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)