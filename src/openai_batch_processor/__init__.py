# This code allows users to import the core class directly from the package.
# Users can use 'from openai_batch_processor import OpenAIBatchProcessor' instead of
# 'from openai_batch_processor.processor import OpenAIBatchProcessor'.
from .processor import OpenAIBatchProcessor

# Define __all__ to explicitly control which names are exposed
# when using 'from openai_batch_processor import *'.
__all__ = ["OpenAIBatchProcessor"]