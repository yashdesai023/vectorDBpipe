"""
vectorDBpipe
An All-in-One Enterprise RAG Engine (Omni-RAG Architecture).
"""

__version__ = "0.2.3"

from vectorDBpipe.config.config_manager import ConfigManager

def __getattr__(name):
    if name == "TextPipeline":
        from vectorDBpipe.pipeline.text_pipeline import TextPipeline
        return TextPipeline
    if name == "VDBpipe":
        from vectorDBpipe.pipeline.vdbpipe import VDBpipe
        return VDBpipe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ConfigManager", "TextPipeline", "VDBpipe"]
