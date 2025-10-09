from setuptools import setup, find_packages

setup(
    name="vectorDBpipe",
    version="0.1.0",
    author="Yash Desai",
    author_email="yash.desai@gmail.com",
    description="A modular text embedding and vector database pipeline for local and cloud vector stores.",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "PyYAML>=6.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
        "pinecone-client>=3.0.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0"
    ],
    python_requires=">=3.8",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
