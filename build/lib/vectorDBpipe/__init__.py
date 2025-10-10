"""
vectorDBpipe
A modular pipeline for text embedding and vector database storage.
"""

from vectorDBpipe.config.config_manager import ConfigManager
from vectorDBpipe.pipeline import TextPipeline

__all__ = ["ConfigManager", "TextPipeline"]
