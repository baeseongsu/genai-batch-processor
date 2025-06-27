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


# --- Environment Setup and Path Configuration ---
# Add the project root to the Python path so this script can find the 'src' directory.
# (e.g., this file is in the 'examples' folder alongside the 'src' folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
load_dotenv()

# --- Import the actual OpenAIBatchProcessor class ---
# Import the class from the provided 'src/openai_batch_processor/processor.py' file.
# Assumes that the 'src/openai_batch_processor/__init__.py' file is configured like
# 'from .processor import OpenAIBatchProcessor'.
try:
    from src.openai_batch_processor.processor import OpenAIBatchProcessor
except ImportError:
    print("Error: Could not find OpenAIBatchProcessor from 'src.openai_batch_processor.processor'.")
    print("Please check your project structure. This script should be located at the same level as the 'src' directory (e.g., in 'examples').")
    sys.exit(1)


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


# 1. Custom class implementation for visual problem solving
class VisualExamProcessor(OpenAIBatchProcessor):
    """
    Batch processing class for multiple choice question solving tasks with images.
    Inherits from the actual OpenAIBatchProcessor.
    """

    def _create_request(self, item: Dict, index: int, **kwargs) -> Dict:
        """
        Convert data containing text and image paths to Vision API request format.

        :param item: Dictionary containing 'system_prompt', 'user_prompt', 'image_path'.
        :param index: Index for generating unique request ID.
        :param kwargs: Additional arguments passed from `run` method (model).
        :return: Batch API request dictionary.
        """
        model = kwargs.get("model", "gpt-4.1-nano") # Specify Vision model

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

        return {
            "custom_id": f"request-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": 100,
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
        }

# 2. Main execution block
if __name__ == "__main__":
    # Generate sample shape image for example execution
    create_shape_image("sample_shape.png")

    # Dataset to process: includes image paths and prompts
    my_visual_shape_questions: List[Dict] = [
        {
            "system_prompt": "You are a shape recognition expert. Look at the provided image and question, then select the most appropriate answer in JSON format.",
            "user_prompt": 'Answer the following multiple choice question. Choose the most appropriate option from the choices and output in the following JSON format: ```json {"answer": "{option_number}"}```\nQuestion: What shape is shown in the following image?\n1. Circle\n2. Triangle\n3. Square\n4. Star',
            "image_path": "sample_shape.png" # Path to the just-generated image file
        }
    ]

    print("--- Starting Image-included Multiple Choice Question Solving Batch Job ---")
    print("Note: This script calls the actual OpenAI API and may incur charges.")

    # OPENAI_API_KEY environment variable must be set.
    exam_solver = VisualExamProcessor()

    # Execute actual batch API workflow
    results, errors = exam_solver.run(
        data=my_visual_shape_questions,
        endpoint="/v1/chat/completions",
        output_path_prefix="my_shape_exam_job",
        poll_interval_seconds=30, # Batch status check interval (seconds)
        model="gpt-4.1-nano" # Vision model to be passed to _create_request
    )

    # 3. Result verification and post-processing
    print("\n--- Job Summary ---")
    print(f"Final batch status: {exam_solver.final_batch_status}")
    print(f"Total {len(results)} results, {len(errors)} errors.")

    if results:
        print("\n--- Detailed Processing Results ---")
        # Sort results by custom_id order for easy matching with original data
        sorted_results = sorted(results, key=lambda r: int(r['custom_id'].split('-')[1]))

        for res in sorted_results:
            custom_id = res.get('custom_id')
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
        print("\n--- Critical Errors During Processing ---")
        for err in errors:
            print(err)

    # Clean up generated image file
    if os.path.exists("sample_shape.png"):
        os.remove("sample_shape.png")
        print("\nCleaned up 'sample_shape.png' file.")
