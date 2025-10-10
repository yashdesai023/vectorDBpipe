# vectorDBpipe

**Version:** 0.1.0  
**Author:** Yash Desai  
**Email:** desaisyash1000@gmail.com  

---

A modular **text embedding and vector database pipeline** for local and cloud vector stores.  
Designed to streamline text preprocessing, embedding generation, and semantic search with multiple backends such as **FAISS, Chroma, and Pinecone**.

---

## 🚀 Features

- Load text from files or directories
- Clean and preprocess text efficiently
- Chunk text for large documents
- Generate embeddings with **Sentence Transformers**
- Store and retrieve embeddings using local (**FAISS**, **Chroma**) or cloud (**Pinecone**) vector databases
- Integrated logging for pipeline operations
- Fully modular and extendable design

---

## 💻 Installation

Install `vectorDBpipe` directly from PyPI:

```bash
pip install vectorDBpipe
```

---

## ⚙️ Configuration

`vectorDBpipe` uses a `config.yaml` file for configuration. You can customize paths, models, and vector database settings.

### Pinecone API Key

If you use the `pinecone` vector database, you must provide your API key via an environment variable. The library will automatically load it.

**Linux/macOS:**
```bash
export PINECONE_API_KEY="YOUR_API_KEY"
```

**Windows:**
```powershell
$env:PINECONE_API_KEY="YOUR_API_KEY"
```

---

## ⚙️ Basic Usage

### 1️⃣ Load Data and Generate Embeddings

```python
from vectorDBpipe.data.loader import DataLoader
from vectorDBpipe.embeddings.embedder import Embedder

# Load all text files from a directory
loader = DataLoader("data/")
data = loader.load_all_files()

# Extract text contents
texts = [d["content"] for d in data]

# Create embeddings
embedder = Embedder()
vectors = embedder.encode(texts)

print("Vectors shape:", vectors.shape)
```

---

### 2️⃣ Text Cleaning and Chunking

```python
from vectorDBpipe.logger.logging import setup_logger
from vectorDBpipe.utils.common import clean_text, chunk_text

logger = setup_logger("TextPipeline")

text = "AI   is transforming   the world!"
cleaned = clean_text(text)
chunks = chunk_text(cleaned, chunk_size=50)

logger.info(f"Cleaned text: {cleaned}")
logger.info(f"Generated {len(chunks)} chunks.")
```

**Output Example:**

```
INFO:TextPipeline: Cleaned text: ai is transforming the world!
INFO:TextPipeline: Generated 1 chunks.
```

---

### 3️⃣ Modular Vector Storage & Retrieval

```python
from vectorDBpipe.vectorstore.faiss_store import FAISSVectorStore

# Initialize vector store
vector_store = FAISSVectorStore(dim=384)

# Add embeddings and metadata
metadata = [{"text": t} for t in texts]
vector_store.add(vectors, metadata)

# Search similar text
query = "Artificial Intelligence"
results = vector_store.search(query, top_k=3)
print("Search results:", results)
```

---

## 📝 Project Structure

```
vectorDBpipe/
├── data/                      # Example dataset
├── vectorDBpipe/
│   ├── data/loader.py         # Data loading module
│   ├── embeddings/embedder.py # Embedding generation
│   ├── vectorstore/           # Vector DB modules (FAISS, Chroma, Pinecone)
│   ├── logger/                # Logging setup
│   └── utils/                 # Helper functions (cleaning, chunking, etc.)
├── tests/                     # Unit tests
├── demo/                      # Demo Jupyter notebooks
├── setup.py
└── README.md
```

---

## 📒 Logging & Debugging

- Use `setup_logger()` to create named loggers for your pipeline.
- Logs capture preprocessing, embedding, and vector store operations for easier debugging.

```python
logger = setup_logger("TextPipeline")
logger.info("Pipeline started...")
```

---

## ✅ Contribution Guide

1. **Fork** the repository
2. **Create a branch:**  
   `git checkout -b feature/my-feature`
3. **Add or modify code** with proper docstrings and type hints
4. **Add tests** under `tests/`
5. **Submit a Pull Request** with a detailed description

---

## 📖 Demo Notebooks

- `demo/TextPipeline_demo.ipynb`: Step-by-step demonstration of data loading, preprocessing, embedding, storage, and search.
- Visualize similarity search results using `pandas` or `matplotlib`.

---

## 📜 License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for more details.

---

## 🔗 Contact

**Author:** Yash Desai  
**Email:** [desaisyash1000@gmail.com](mailto:desaisyash1000@gmail.com)  
**GitHub:** [https://github.com/yashdesai023/vectorDBpipe](https://github.com/yashdesai023/vectorDBpipe)

---

*Ready for contributions and feedback! If you need a polished demo notebook, let me know!*

