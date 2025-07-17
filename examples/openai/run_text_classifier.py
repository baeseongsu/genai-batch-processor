# -*- coding: utf-8 -*-
"""
Example script for running a text classification batch job using the OpenAIBatchProcessor.

This script demonstrates how to:
1.  Define a custom processor class for a specific task (text classification).
2.  Prepare input data for batch processing.
3.  Execute the batch processing job using the OpenAI Batch API.
4.  Retrieve, parse, and display the results.
5.  Handle errors and post-process the results.
"""

import os
import sys
from typing import Dict, List
from dotenv import load_dotenv

# --- Environment setup and path configuration ---
# Add the project root to Python path so this script can find the 'src' directory.
# This allows running the example directly without needing 'pip install .'.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load environment variables (OPENAI_API_KEY) from .env file.
load_dotenv()

# --- Import the base class ---
from src.genai_batch_processor import OpenAIBatchProcessor


# 1. Custom class implementation
class MyTextClassifier(OpenAIBatchProcessor):
    """
    Batch processor for text classification. It inherits from OpenAIBatchProcessor
    and implements the request creation logic for the classification task.
    """
    
    def _create_request(self, item: str, index: int, **kwargs) -> Dict:
        """
        Converts a string of text into an OpenAI Batch API request for classification.

        :param item: A text sentence to classify.
        :param index: Index for creating unique request ID.
        :param kwargs: Additional arguments passed from the `run` method (model in this case).
        :return: A dictionary formatted as an OpenAI Batch API request.
        """
        model = kwargs.get("model", "gpt-4.1-nano") # Default model setting

        return {
            "custom_id": f"request-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that classifies text as 'positive' or 'negative'."},
                    {"role": "user", "content": f"Classify the following sentence: '{item}'"}
                ],
                "max_tokens": 10,
                "temperature": 0.1,
            }
        }

# 2. Main execution block
if __name__ == "__main__":
    print("--- 🚀 Starting OpenAI text classification batch job ---")

    # --- Data and Model Parameters ---
    # A sample dataset for classification.
    my_sample_texts: List[str] = [
        "The weather today is absolutely beautiful!",
        "I'm feeling wonderful after a great night's sleep.",
        "This movie was a complete waste of time and money.",
        "The customer service I received was incredibly disappointing.",
        "I'm not sure how I feel about this new policy.",
        "An unforgettable experience, truly one for the books."
    ]

    # --- Job Execution ---
    # Instantiate the custom classifier.
    # OpenAIBatchProcessor constructor automatically reads OPENAI_API_KEY environment variable.
    classifier = MyTextClassifier() 
    
    # Run the entire batch process: validate, upload, create job, monitor, and get results.
    results, errors = classifier.run(
        data=my_sample_texts,
        endpoint="/v1/chat/completions",
        output_path_prefix="my_classification_job", # Result file name prefix
        poll_interval_seconds=30, # Check status every 30 seconds
        model="gpt-4.1-nano" # Specify model to use (passed to _create_request)
    )
    
    # 3. Result verification and post-processing
    print("\n--- ✅ Job Completed: Detailed Processing Results ---")
    print(f"Final batch status: {classifier.final_batch_status}")
    print(f"Total {len(results)} results, {len(errors)} errors.")

    if results:
        print("\n--- Processing Results ---")
        # Sort results by custom_id order to easily match with original data.
        sorted_results = sorted(results, key=lambda r: int(r['custom_id'].split('-')[1]))

        for res in sorted_results:
            custom_id = res.get('custom_id')
            # Extract index from custom_id (e.g., "request-0" -> 0)
            original_index = int(custom_id.split('-')[1])
            original_text = my_sample_texts[original_index]
            
            print(f"\n[Original] \"{original_text}\" (ID: {custom_id})")

            if res.get('error'):
                 print(f"  🚨 Error occurred: {res['error']['message']}")
            elif res['response']['status_code'] == 200:
                completion = res['response']['body']['choices'][0]['message']['content']
                print(f"  ✅ Classification result: {completion.strip()}")
            else:
                print(f"  ⚠️ API error response (status code: {res['response']['status_code']}): {res['response']['body']}")

    if errors:
        print("\n--- ❌ Critical errors during processing ---")
        for err in errors:
            print(err)

    print("--- ✨ All operations finished. ---")