# pipeline/text_pipeline.py

import logging
from vectorDBpipe.config.config_manager import ConfigManager
from vectorDBpipe.data.loader import DataLoader
from vectorDBpipe.embeddings.embedder import Embedder
from vectorDBpipe.vectordb.store import get_vector_store
from vectorDBpipe.utils.common import clean_text, chunk_text
from vectorDBpipe.logger.logging import setup_logger

class TextPipeline:
    """
    Main pipeline for processing text documents and storing embeddings
    into a selected vector database (Chroma or Pinecone).
    """
    def __init__(self, config_path="vectorDBpipe/config/config.yaml"):
        self.config = ConfigManager(config_path)
        
        # Setup logger
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            name="TextPipeline",
            log_dir=self.config.get('paths.logs_dir', 'logs/'),
            level=log_config.get('level', 'INFO')
        )
        
        # Load components from config
        self.data_path = self.config.get("paths.data_dir")
        self.embed_model = self.config.get("model.name")
        # try both keys commonly used: vectordb or vector_db
        self.vector_db_cfg = self.config.get("vectordb") or self.config.get("vector_db") or {}

        self.logger.info("Initializing TextPipeline with configuration")
        self.loader = DataLoader(self.data_path)
        self.embedder = Embedder(model_name=self.embed_model)
        self.vector_store = get_vector_store(
            store_type=self.vector_db_cfg.get("type", "chroma"),
            config=self.vector_db_cfg
        )

    def process(self):
        """
        End-to-end document processing pipeline:
        1. Load -> 2. Clean & Chunk -> 3. Embed -> 4. Store
        """
        try:
            self.logger.info("Starting document processing pipeline...")
            
            self.logger.info(f"Loading data from: {self.data_path}")
            documents = self.loader.load_data()
            if not documents:
                self.logger.warning("No documents found in the specified data path. Exiting process.")
                return

            self.logger.info(f"Found {len(documents)} documents to process.")
            
            all_chunks = []
            metadata_list = []
            for doc in documents:
                content = doc.get("content")
                source = doc.get("source")
                if not content:
                    continue
                
                cleaned = clean_text(content)
                chunks = chunk_text(cleaned, chunk_size=512)
                all_chunks.extend(chunks)
                metadata_list.extend([{"source": source, "text": chunk} for chunk in chunks])

            if not all_chunks:
                self.logger.warning("No text chunks were generated after cleaning. Exiting process.")
                return

            self.logger.info(f"Generated {len(all_chunks)} chunks for embedding.")
            
            self.logger.info("Generating embeddings...")
            embeddings = self.embedder.encode(all_chunks)
            
            self.logger.info("Storing embeddings in the vector database...")
            # ensure embeddings are lists for storage backends
            emb_as_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
            self.vector_store.insert_vectors(emb_as_list, metadata=metadata_list)
            
            self.logger.info("Pipeline completed successfully!")

        except Exception as e:
            self.logger.exception(f"Pipeline failed due to: {str(e)}")
            raise

    def search(self, query, top_k=5):
        """
        Query the vector store to find semantically similar content.
        """
        try:
            self.logger.info(f"Searching for top {top_k} results for query: '{query[:100]}...'")
            query_embedding = self.embedder.encode(query)
            
            # Convert to python list
            if hasattr(query_embedding, "tolist"):
                qvec = query_embedding.tolist()[0] if isinstance(query_embedding.tolist(), list) and isinstance(query_embedding.tolist()[0], list) else query_embedding.tolist()
            else:
                qvec = query_embedding

            results = self.vector_store.search_vectors(qvec, top_k=top_k)
            self.logger.info("Search completed.")
            return results
            
        except Exception as e:
            self.logger.exception(f"Search failed: {e}")
            return []

    def run(self):
        """Executes the full text processing pipeline."""
        self.logger.info("Running TextPipeline: load → embed → store")

        # Step 1: Load data
        data = self.loader.load_data()
        if not data:
            self.logger.warning("No data found to process.")
            return None

        # Step 2: Embed texts
        if isinstance(data[0], dict) and "content" in data[0]:
            texts = [d["content"] for d in data]
        else:
            texts = data

        embeddings = self.embedder.embed_texts(texts)

        # Step 3: Store vectors
        self.vector_store.insert_vectors(embeddings, texts)
        self.logger.info("Pipeline completed successfully.")

        return True
