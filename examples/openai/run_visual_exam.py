# -*- coding: utf-8 -*-
"""
Example script for running a visual examination batch job using the OpenAIBatchProcessor.

This script demonstrates how to:
1.  Define a custom processor class for a specific task (visual question answering).
2.  Prepare input data for batch processing with images.
3.  Execute the batch processing job using the OpenAI Batch API.
4.  Retrieve, parse, and display the results.
5.  Handle errors and post-process the results.
"""

import os
import sys
import json
import base64
from typing import Dict, List
from dotenv import load_dotenv

# Import Pillow library (required for image generation)
# Install with: pip install Pillow
try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow library is required. Please run 'pip install Pillow'.")
    sys.exit(1)

# --- Environment setup and path configuration ---
# Add the project root to Python path so this script can find the 'src' directory.
# This allows running the example directly without needing 'pip install .'.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load environment variables (OPENAI_API_KEY) from .env file.
load_dotenv()

# --- Import the base class ---
from src.genai_batch_processor import OpenAIBatchProcessor


# --- Helper functions: Image processing ---
def image_to_base64(image_path: str) -> str:
    """Read an image file and encode it to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Could not find image file '{image_path}'.")
        return None

def create_shape_image(path: str = "sample_shape.png"):
    """Generate an example shape (circle) image."""
    if os.path.exists(path):
        # print(f"Image '{path}' already exists.")
        return
        
    img_size = (256, 256)
    img = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a red circle in the center
    bbox = [64, 64, 192, 192]
    draw.ellipse(bbox, fill='red', outline='black')
    
    img.save(path)
    print(f"Generated example image '{path}'.")


# 1. Custom class implementation
class VisualExamProcessor(OpenAIBatchProcessor):
    """
    Batch processor for visual examination with images. It inherits from OpenAIBatchProcessor
    and implements the request creation logic for the visual question answering task.
    """

    def _create_request(self, item: Dict, index: int, **kwargs) -> Dict:
        """
        Converts a data item containing text and image paths into an OpenAI Batch API request for visual examination.

        :param item: Dictionary containing 'system_prompt', 'user_prompt', 'image_path'.
        :param index: Index for creating unique request ID.
        :param kwargs: Additional arguments passed from the `run` method (model in this case).
        :return: A dictionary formatted as an OpenAI Batch API request.
        """
        model = kwargs.get("model", "gpt-4o") # Default model setting

        # Encode image file to Base64
        base64_image = image_to_base64(item["image_path"])
        if not base64_image:
            raise FileNotFoundError(f"Could not process image file: {item['image_path']}")

        # Construct message content in Vision API format
        user_content = [
            {
                "type": "text",
                "text": item["user_prompt"]
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]

        # Prepare request body
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": user_content}
            ],
            "max_completion_tokens": 100, # Use 'max_completion_tokens' for o-series models
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        # Remove temperature parameter if model starts with 'o', as it might not be supported
        if model.lower().startswith('o'):
            if "temperature" in body:
                del body["temperature"]

        return {
            "custom_id": f"request-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }

# 2. Main execution block
if __name__ == "__main__":
    print("--- 🚀 Starting visual examination batch job ---")

    # Generate sample shape image for example execution
    create_shape_image("sample_shape.png")

    # --- Data and Model Parameters ---
    # A sample dataset for visual examination.
    my_visual_shape_questions: List[Dict] = [
        {
            "system_prompt": "You are a shape recognition expert. Look at the provided image and question, then select the most appropriate answer in JSON format.",
            "user_prompt": 'Answer the following multiple choice question. Choose the most appropriate option from the choices and output in the following JSON format: ```json {"answer": "{option_number}"}```\nQuestion: What shape is shown in the following image?\n1. Circle\n2. Triangle\n3. Square\n4. Star',
            "image_path": "sample_shape.png" # Path to the just-generated image file
        }
    ]

    print("Note: This script calls the actual OpenAI API and may incur charges.")

    # --- Job Execution ---
    # Instantiate the custom visual examiner.
    # OpenAIBatchProcessor constructor automatically reads OPENAI_API_KEY environment variable.
    exam_solver = VisualExamProcessor()

    # Run the entire batch process: validate, upload, create job, monitor, and get results.
    results, errors = exam_solver.run(
        data=my_visual_shape_questions,
        endpoint="/v1/chat/completions",
        output_path_prefix="my_job/visual_exam", # Result file name prefix
        poll_interval_seconds=30, # Check status every 30 seconds
        model="o4-mini" # Specify model to use (passed to _create_request)
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
            original_data = my_visual_shape_questions[original_index]
            question_preview = original_data['user_prompt'].split('\nQuestion:')[1].strip().split('\n')[0]

            print(f"\n[Original] \"{question_preview}...\" (ID: {custom_id})")

            if res.get('error'):
                print(f"  🚨 Error occurred: {res['error']['message']}")
            elif res['response']['status_code'] == 200:
                try:
                    response_body = res['response']['body']
                    
                    # Handle cases where content might be null (refusal)
                    message_content = response_body['choices'][0]['message']['content']
                    if message_content is None:
                        refusal = response_body['choices'][0]['message'].get('refusal', 'No content returned')
                        print(f"  ⚠️ Model refused to answer: {refusal}")
                    else:
                        completion_data = json.loads(message_content)
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

    # Clean up generated image file
    if os.path.exists("sample_shape.png"):
        os.remove("sample_shape.png")
        print("\n✨ Cleaned up 'sample_shape.png' file.")

    print("--- ✨ All operations finished. ---")
