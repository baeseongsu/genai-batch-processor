import os
import sys
from typing import Dict, List
from dotenv import load_dotenv

# --- Environment setup and path configuration ---
# Add the project root to Python path so this script can find the 'src' directory.
# This allows running the example directly without needing 'pip install .'.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load environment variables (OPENAI_API_KEY) from .env file.
load_dotenv()

# --- Import class from package ---
# Thanks to __init__.py, we can import the class concisely.
from src.openai_batch_processor import OpenAIBatchProcessor


# 1. Custom class implementation
class MyTextClassifier(OpenAIBatchProcessor):
    """
    Batch processing class for text classification tasks.
    Inherits from OpenAIBatchProcessor and implements the _create_request method.
    """
    
    def _create_request(self, item: str, index: int, **kwargs) -> Dict:
        """
        Converts string data into a classification request using gpt-3.5-turbo.

        :param item: Text sentence to classify.
        :param index: Index for creating unique request ID.
        :param kwargs: Additional arguments passed from the `run` method (model in this case).
        :return: Batch API request dictionary.
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
    # Large dataset to process (sample)
    my_sample_texts: List[str] = [
        "The weather today is absolutely beautiful!",
        "I'm feeling wonderful after a great night's sleep.",
        "This movie was a complete waste of time and money.",
        "The customer service I received was incredibly disappointing.",
        "I'm not sure how I feel about this new policy.",
        "An unforgettable experience, truly one for the books."
    ]

    print("--- Starting text classification batch job ---")

    # Create class instance using API key.
    # OpenAIBatchProcessor constructor automatically reads OPENAI_API_KEY environment variable.
    classifier = MyTextClassifier() 
    
    # Call the `run` method to execute the entire process
    results, errors = classifier.run(
        data=my_sample_texts,
        endpoint="/v1/chat/completions",
        output_path_prefix="my_classification_job", # Result file name prefix
        poll_interval_seconds=30, # Check status every 30 seconds
        model="gpt-4.1-nano" # Specify model to use (passed to _create_request)
    )
    
    # 3. Result verification and post-processing
    print("\n--- Job Summary ---")
    print(f"Final batch status: {classifier.final_batch_status}")
    print(f"Total {len(results)} results, {len(errors)} errors.")

    if results:
        print("\n--- Detailed Processing Results ---")
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
        print("\n--- Critical errors during processing ---")
        for err in errors:
            print(err)