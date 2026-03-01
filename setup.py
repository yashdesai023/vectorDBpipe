import os
from setuptools import setup, find_packages

setup(
    name="vectordbpipe",
    version="0.2.3",
    author="Yash Desai",
    author_email="desaisyash1000@gmail.com",

    # ─── PyPI short description (appears in search results) ───────────────────
    description=(
        "vectorDBpipe v0.2.3 — Enterprise Omni-RAG SDK. "
        "Tri-Processing Ingestion + 4 AI Engines (Vector RAG, Vectorless RAG, "
        "GraphRAG, Structured JSON Extract) + 15+ data connectors. "
        "Hotfix: includes LLM provider subpackage (sarvam, openai, groq, anthropic). "
        "One pipeline. One API. Zero glue code."
    ),

    # ─── Long description (PyPI page body) ────────────────────────────────────
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",

    # ─── Project URLs (shown in PyPI sidebar) ─────────────────────────────────
    project_urls={
        "Homepage":      "https://github.com/yashdesai023/vectorDBpipe",
        "Source":        "https://github.com/yashdesai023/vectorDBpipe",
        "Bug Tracker":   "https://github.com/yashdesai023/vectorDBpipe/issues",
        "Changelog":     "https://github.com/yashdesai023/vectorDBpipe/releases",
        "Documentation": "https://github.com/yashdesai023/vectorDBpipe#readme",
    },

    # ─── Packages ─────────────────────────────────────────────────────────────
    packages=find_packages(exclude=[
        "vectorDBpipe-tui", "vectorDBpipe-tui.*",
        "frontend-vectordbpipe", "frontend-vectordbpipe.*",
        "tests*",
    ]),
    include_package_data=True,

    # ─── Dependencies ─────────────────────────────────────────────────────────
    install_requires=[
        "PyYAML>=6.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "transformers>=4.28.1",
        "torch>=2.2.0",
        "torchvision",
        "chromadb>=0.4.22",
        "pinecone-client>=3.0.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "docx2txt>=0.8",
        "beautifulsoup4>=4.12.3",
        "PyMuPDF>=1.23.26",
        "networkx>=3.1",
        "langchain>=0.1.13",
        "langchain-core>=0.1.33",
        "pydantic>=2.0.0",
        "boto3>=1.26.0",
        "markdown>=3.4.0",
        "requests>=2.32.3",
    ],

    python_requires=">=3.8",
    license="MIT",

    # ─── PyPI Classifiers ─────────────────────────────────────────────────────
    classifiers=[
        # Maturity
        "Development Status :: 4 - Beta",

        # Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Database",
        "Topic :: Text Processing :: Indexing",

        # License
        "License :: OSI Approved :: MIT License",

        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",

        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],

    # ─── Search Keywords ──────────────────────────────────────────────────────
    keywords=[
        "rag", "retrieval-augmented-generation", "vector-database",
        "llm", "embeddings", "langchain", "faiss", "chromadb", "pinecone",
        "graphrag", "knowledge-graph", "nlp", "ai", "genai", "etl",
        "semantic-search", "document-qa", "openai", "groq", "sentence-transformers",
        "vectordbpipe", "omni-rag", "tri-processing",
    ],
)