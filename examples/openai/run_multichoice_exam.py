# -*- coding: utf-8 -*-
"""
Example script for running a multiple choice exam batch job using the OpenAIBatchProcessor.

This script demonstrates how to:
1.  Define a custom processor class for a specific task (multiple choice question answering).
2.  Prepare input data for batch processing with pre-formatted messages.
3.  Execute the batch processing job using the OpenAI Batch API.
4.  Retrieve, parse, and display the results.
5.  Handle errors and post-process the results.
"""

import os
import sys
import json
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
class MultiChoiceExamProcessor(OpenAIBatchProcessor):
    """
    Batch processor for multiple choice question solving. It inherits from OpenAIBatchProcessor
    and implements the request creation logic for the exam solving task.
    """

    def _create_request(self, item: List[Dict], index: int, **kwargs) -> Dict:
        """
        Converts pre-formatted message list into an OpenAI Batch API request for exam solving.

        :param item: A list of messages to be used for the API request.
        :param index: Index for creating unique request ID.
        :param kwargs: Additional arguments passed from the `run` method (model in this case).
        :return: A dictionary formatted as an OpenAI Batch API request.
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

# 2. Main execution block
if __name__ == "__main__":
    print("--- 🚀 Starting OpenAI multiple choice exam batch job ---")

    # --- Data and Model Parameters ---
    # A sample dataset for exam solving. Each item is a 'messages' list to be passed to the API.
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

    # --- Job Execution ---
    # Instantiate the custom exam solver.
    # OpenAIBatchProcessor constructor automatically reads OPENAI_API_KEY environment variable.
    exam_solver = MultiChoiceExamProcessor()

    # Run the entire batch process: validate, upload, create job, monitor, and get results.
    results, errors = exam_solver.run(
        data=my_medical_questions,
        endpoint="/v1/chat/completions",
        output_path_prefix="my_medical_exam_job",  # Result file name prefix
        poll_interval_seconds=30,  # Check status every 30 seconds
        model="gpt-4.1-nano"  # Specify model to use (passed to _create_request)
    )

    # 3. Result verification and post-processing
    print("\n--- ✅ Job Completed: Detailed Processing Results ---")
    print(f"Final batch status: {exam_solver.final_batch_status}")
    print(f"Total {len(results)} results, {len(errors)} errors.")

    if results:
        print("\n--- Processing Results ---")
        # Sort results by custom_id order to easily match with original data.
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
        print("\n--- ❌ Critical errors during processing ---")
        for err in errors:
            print(err)

    print("--- ✨ All operations finished. ---")