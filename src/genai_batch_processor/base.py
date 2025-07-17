# src/genai_batch_processor/base.py
from abc import ABC, abstractmethod

class BaseBatchProcessor(ABC):
    """A base interface for all GenAI batch processors."""
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """Executes the complete batch processing workflow."""
        pass

    @abstractmethod
    def validate_request(self, *args, **kwargs) -> bool:
        """Performs a pre-flight check on the request format."""
        pass