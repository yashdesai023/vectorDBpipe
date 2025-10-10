

# **vectorDBpipe**

**Version:** 0.1.2
**Author:** Yash Desai
**Email:** [desaisyash1000@gmail.com](mailto:desaisyash1000@gmail.com)

---

### Overview

`vectorDBpipe` is a modular Python framework designed to simplify the creation of **text embedding and vector database pipelines**.
It enables developers and researchers to efficiently process, embed, and retrieve large text datasets using modern vector databases such as **FAISS**, **Chroma**, or **Pinecone**.

The framework follows a **layered, plug-and-play architecture**, allowing easy customization of data loaders, embedding models, and storage backends.

---

## Key Features

* Structured **data ingestion, cleaning, and chunking**
* Embedding generation via **Sentence Transformers**
* Pluggable vector storage engines: **FAISS**, **Chroma**, **Pinecone**
* Unified CRUD API for inserting, searching, updating, and deleting embeddings
* YAML-based configuration for quick workflow adjustments
* Integrated logging and exception handling
* End-to-end orchestration through a single pipeline interface

---

## Installation

Install from PyPI:

```bash
pip install vectordbpipe
```

Or for local development:

```bash
git clone https://github.com/yashdesai023/vectorDBpipe.git
cd vectorDBpipe
pip install -e .
```

---

## Configuration

The system reads settings from a YAML configuration file (`config.yaml`), which defines parameters for:

* **Data sources** (paths, formats)
* **Embedding model** (e.g., `all-MiniLM-L6-v2`)
* **Vector database backend** (FAISS, Chroma, or Pinecone)
* **Index parameters and persistence options**

### Pinecone Setup (Optional)

If you choose `pinecone` as your vector database, provide your API key as an environment variable.

**macOS/Linux:**

```bash
export PINECONE_API_KEY="your_api_key"
```

**Windows PowerShell:**

```powershell
$env:PINECONE_API_KEY="your_api_key"
```

---

## Quick Start

### 1. Data Loading and Embedding

```python
from vectorDBpipe.data.loader import DataLoader
from vectorDBpipe.embeddings.embedder import Embedder

loader = DataLoader("data/")
documents = loader.load_all_files()
texts = [d["content"] for d in documents]

embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.encode(texts)

print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}.")
```

---

### 2. Text Cleaning and Chunking

```python
from vectorDBpipe.utils.common import clean_text, chunk_text
from vectorDBpipe.logger.logging import setup_logger

logger = setup_logger("Preprocess")

sample_text = "AI   is transforming   industries worldwide."
cleaned = clean_text(sample_text)
chunks = chunk_text(cleaned, chunk_size=50)

logger.info(f"Cleaned Text: {cleaned}")
logger.info(f"Generated {len(chunks)} chunks.")
```

---

### 3. Vector Storage and Retrieval

```python
from vectorDBpipe.vectordb.store import VectorStore

store = VectorStore(backend="faiss", dim=384)
store.insert_vectors(texts, embeddings)

query = "Applications of Artificial Intelligence"
results = store.search_vectors(query, top_k=3)

print("Top Similar Results:")
for r in results:
    print("-", r)
```

---

### 4. Full Pipeline Execution

```python
from vectorDBpipe.pipeline.text_pipeline import TextPipeline
from vectorDBpipe.config.config_manager import ConfigManager

config = ConfigManager().get_config()
pipeline = TextPipeline(config)

results = pipeline.run(["Machine learning enables predictive analytics."],
                       query="What is machine learning?")
print(results)
```

---

## Project Structure

```
vectorDBpipe/
│
├── vectorDBpipe/
│   ├── config/                # Configuration management
│   ├── data/                  # Data loading and preprocessing
│   ├── embeddings/            # Embedding generation
│   ├── vectordb/              # Vector database abstraction layer
│   ├── pipeline/              # End-to-end workflow orchestration
│   ├── utils/                 # Common utilities (cleaning, chunking)
│   └── logger/                # Logging utilities
│
├── tests/                     # Unit tests
├── demo/                      # Example Jupyter notebooks
├── setup.py
└── README.md
```

---

## Logging and Error Handling

Every module integrates with a centralized logging system to track operations and debug efficiently.

```python
from vectorDBpipe.logger.logging import setup_logger
logger = setup_logger("VectorDBPipe")
logger.info("Pipeline started successfully.")
```

---

## Testing

Run the test suite to verify installation and functionality:

```bash
pytest -v --cov=vectorDBpipe
```

Coverage reports can be generated to ensure code reliability.

---

## Example Notebook

A demonstration notebook `vector_pipeline_demo.ipynb` is included, showcasing:

* Document embedding and visualization
* Vector similarity retrieval
* PCA-based embedding visualization

You can also run it directly in Google Colab:

```markdown
[Open in Colab](https://colab.research.google.com/github/yashdesai023/vectorDBpipe/blob/main/vector_pipeline_demo.ipynb)
```

---

## Contributing

Contributions are welcome.
Please ensure all pull requests include:

* Clear, modular code
* Type hints and docstrings
* Unit tests covering new functionality

**Development Workflow**

```bash
git checkout -b feature/my-feature
# Add your changes
pytest -v
git commit -m "Add new feature"
git push origin feature/my-feature
```

Then submit a pull request.

---

## License

Distributed under the **MIT License**.
See the [LICENSE](LICENSE) file for full terms.

---

## Author & Contact

**Yash Desai**
Computer Science & Engineering (AI)
Email: [desaisyash1000@gmail.com](mailto:desaisyash1000@gmail.com)
GitHub: [yashdesai023](https://github.com/yashdesai023)

---


