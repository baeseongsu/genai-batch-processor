import os
import sys
import json
from typing import Dict, List
from dotenv import load_dotenv

# --- Environment Setup and Path Configuration ---
# Add the project root to the Python path so this script can find the 'src' directory.
# This allows running examples directly without 'pip install .'.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load environment variables (OPENAI_API_KEY) from .env file.
load_dotenv()

# --- Import Classes from Package ---
# Thanks to __init__.py, we can import classes concisely.
from src.openai_batch_processor import OpenAIBatchProcessor


# 1. Custom Class Implementation
class MultiChoiceExamProcessor(OpenAIBatchProcessor):
    """
    Batch processing class for multiple choice question solving tasks.
    Inherits from OpenAIBatchProcessor and implements the _create_request method.
    """

    def _create_request(self, item: List[Dict], index: int, **kwargs) -> Dict:
        """
        Convert pre-formatted message list to API request format.

        :param item: 'messages' list to be used for API request.
        :param index: Index for generating unique request ID.
        :param kwargs: Additional arguments passed from `run` method (in this case, model).
        :return: Batch API request dictionary.
        """
        model = kwargs.get("model", "gpt-4.1-nano")  # Default model setting

        return {
            "custom_id": f"request-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": item,  # Use message list from data source directly
                "max_tokens": 50,
                "temperature": 0.0,
                "response_format": {"type": "json_object"}  # Force JSON output
            }
        }

# 2. Main Execution Block
if __name__ == "__main__":
    # Dataset to process. Each item is a 'messages' list to be passed to the API.
    # Changed domain to 'medical'.
    my_medical_questions: List[List[Dict]] = [
        [ # First question (corresponds to one API call)
            {'role': 'system', 'content': 'You are a medical expert.'},
            {
                'role': 'user',
                'content': 'Answer the following multiple choice question. Choose the most appropriate option from the choices and output in the following JSON format: ```json {"answer": "{option_number}"}``` Do not output any text other than the answer.\nQuestion: Which blood type can a person with O blood type receive in a transfusion?\n1. Type A\n2. Type B\n3. Type AB\n4. Type O'
            }
        ]
        # If needed, other medical-related 'messages' lists can be added here.
    ]

    print("--- Starting Multiple Choice Question Solving Batch Job ---")

    # Create class instance. API key is automatically read from environment variables.
    exam_solver = MultiChoiceExamProcessor()

    # Call `run` method to execute the entire process
    results, errors = exam_solver.run(
        data=my_medical_questions,
        endpoint="/v1/chat/completions",
        output_path_prefix="my_medical_exam_job",  # Changed result file name prefix
        poll_interval_seconds=30,  # Check status every 30 seconds
        model="gpt-4.1-nano"  # Specify model to use
    )

    # 3. Result Verification and Post-processing
    print("\n--- Job Summary ---")
    print(f"Final batch status: {exam_solver.final_batch_status}")
    print(f"Total {len(results)} results, {len(errors)} errors.")

    if results:
        print("\n--- Detailed Processing Results ---")
        # Sort results by custom_id order for easy matching with original data
        sorted_results = sorted(results, key=lambda r: int(r['custom_id'].split('-')[1]))

        for res in sorted_results:
            custom_id = res.get('custom_id')
            # Extract index from custom_id (e.g., "request-0" -> 0)
            original_index = int(custom_id.split('-')[1])
            original_messages = my_medical_questions[original_index]
            
            # Extract only the question part from original user message briefly
            user_content = next((msg['content'] for msg in original_messages if msg['role'] == 'user'), "No question")
            try:
                question_preview = user_content.split('Question:')[1].strip().split('\n')[0]
            except IndexError:
                question_preview = "Unable to confirm question content"


            print(f"\n[Original] \"{question_preview}...\" (ID: {custom_id})")

            if res.get('error'):
                print(f"  🚨 Error occurred: {res['error']['message']}")
            elif res['response']['status_code'] == 200:
                try:
                    # Parse JSON content from response body
                    response_body = res['response']['body']
                    completion_json_str = response_body['choices'][0]['message']['content']
                    completion_data = json.loads(completion_json_str)
                    answer = completion_data.get("answer", "N/A")
                    print(f"  ✅ Model answer: {answer}")
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"  ⚠️ JSON response parsing error: {e}")
                    print(f"  - Original response: {res['response']['body']}")
            else:
                print(f"  ⚠️ API error response (status code: {res['response']['status_code']}): {res['response']['body']}")

    if errors:
        print("\n--- Critical Errors During Processing ---")
        for err in errors:
            print(err)