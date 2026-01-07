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

    def process(self, batch_size=100):
        """
        End-to-end document processing pipeline with batch processing:
        1. Load -> 2. Clean & Chunk -> 3. Embed -> 4. Store
        
        :param batch_size: Number of chunks to process in a single batch to manage memory.
        """
        try:
            self.logger.info("Starting document processing pipeline...")
            
            self.logger.info(f"Loading data from: {self.data_path}")
            documents = self.loader.load_data()
            if not documents:
                self.logger.warning("No documents found in the specified data path. Exiting process.")
                return

            self.logger.info(f"Found {len(documents)} documents to process.")
            
            # Temporary storage for batch processing
            chunk_batch = []
            metadata_batch = []
            
            total_chunks = 0
            
            for doc in documents:
                content = doc.get("content")
                source = doc.get("source")
                if not content:
                    continue
                
                cleaned = clean_text(content)
                chunks = chunk_text(cleaned, chunk_size=512)
                
                # Add chunks to current batch
                chunk_batch.extend(chunks)
                metadata_batch.extend([{"source": source, "text": chunk} for chunk in chunks])
                
                # If batch size exceeded, process and flush
                if len(chunk_batch) >= batch_size:
                    self._embed_and_store(chunk_batch, metadata_batch)
                    total_chunks += len(chunk_batch)
                    chunk_batch = []
                    metadata_batch = []
            
            # Process remaining chunks
            if chunk_batch:
                self._embed_and_store(chunk_batch, metadata_batch)
                total_chunks += len(chunk_batch)

            if total_chunks == 0:
                self.logger.warning("No text chunks were generated after cleaning. Exiting process.")
            else:
                self.logger.info(f"Pipeline completed successfully! Processed {total_chunks} total chunks.")

        except Exception as e:
            self.logger.exception(f"Pipeline failed due to: {str(e)}")
            raise

    def _embed_and_store(self, chunks, metadata):
        """Helper to embed and store a batch of chunks."""
        self.logger.info(f"Processing batch of {len(chunks)} chunks...")
        embeddings = self.embedder.encode(chunks)
        
        # ensure embeddings are lists for storage backends
        emb_as_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        self.vector_store.insert_vectors(emb_as_list, metadata=metadata)
        self.logger.info("Batch stored.")

    def run(self):
        """
        Deprecated: Use process() instead.
        Executes the full text processing pipeline.
        """
        self.logger.warning("'run()' is deprecated and will be removed. Please use 'process()' instead. Calling process() now.")
        self.process()
        return True
