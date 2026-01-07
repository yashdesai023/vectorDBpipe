"""
vectorDBpipe
A modular pipeline for text embedding and vector database storage.
"""

__version__ = "0.1.6"

from vectorDBpipe.config.config_manager import ConfigManager
from vectorDBpipe.pipeline import TextPipeline

__all__ = ["ConfigManager", "TextPipeline"]
