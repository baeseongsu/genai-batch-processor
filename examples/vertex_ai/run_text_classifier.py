# -*- coding: utf-8 -*-
"""
Example script for running a text classification batch job using the VertexAIBatchProcessor.

This script demonstrates how to:
1.  Define a custom processor class for a specific task (text classification).
2.  Prepare input data and cloud resources (GCS bucket).
3.  Execute the batch processing job using the Gemini API on Vertex AI.
4.  Retrieve, parse, and display the results.
5.  Clean up all created cloud resources.
"""

import os
import sys
import json
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

# --- Environment setup and path configuration ---
# Add the project root to the Python path to allow importing from the 'src' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load environment variables (like GOOGLE_CLOUD_PROJECT) from a .env file.
load_dotenv()

# --- Import the base class ---
from src.genai_batch_processor.vertex_ai_batch_processor import VertexAIBatchProcessor
from google.genai import types

# 1. Custom class implementation
class MyVertexTextClassifier(VertexAIBatchProcessor):
    """
    Batch processor for text classification. It inherits from VertexAIBatchProcessor
    and implements the request creation logic for the classification task.
    """
    def __init__(self, project_id: str, location: str, generation_config: Dict[str, Any], **kwargs):
        """
        Initializes the classifier.

        Args:
            project_id: Google Cloud project ID.
            location: Google Cloud location.
            generation_config: The generation configuration for the model.
            **kwargs: Additional configuration options, including thinking_budget.
        """
        super().__init__(project_id, location)
        if not generation_config:
            raise ValueError("A generation_config is required for the classifier.")
        self.generation_config = generation_config
        self.thinking_config = types.ThinkingConfig(thinking_budget=kwargs.get("thinking_budget", 0))

    def _create_request_data(self, item: str) -> Dict[str, Any]:
        """
        Converts a string of text into a Gemini API request for classification.

        Args:
            item: A text sentence to classify.

        Returns:
            A dictionary formatted as a GenerateContentRequest.
        """
        # The prompt instructs the model on its role and the task.
        prompt = (
            "You are a helpful assistant that classifies text as 'positive' or 'negative'. "
            f"Classify the following sentence: '{item}'"
        )

        request_data = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": self.generation_config
        }
        
        # Add thinking config if it has a non-zero budget
        if self.thinking_config.thinking_budget > 0:
            request_data["thinking_config"] = self.thinking_config
            
        return request_data

# 2. Main execution block
if __name__ == "__main__":
    print("--- 🚀 Starting Vertex AI text classification batch job ---")

    # --- Configuration ---
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    LOCATION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
    # Updated to use the Gemini 2.5 Flash model
    MODEL_ID = "gemini-2.5-flash" 

    if not PROJECT_ID:
        print("❌ Error: The GOOGLE_CLOUD_PROJECT environment variable is not set.")
        sys.exit(1)
        
    # Generate unique names for cloud resources to avoid conflicts.
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    bucket_name = f"{PROJECT_ID.lower()}-classification-test-{timestamp}"
    
    # Define GCS paths for input and output files.
    gcs_input_uri = f"gs://{bucket_name}/inputs/requests_{timestamp}.jsonl"
    gcs_output_uri = f"gs://{bucket_name}/outputs/" # Output must be a folder URI.
    local_save_path = f"results/vertex_classification_results_{timestamp}.csv"
    
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

    # Configuration for the generative model.
    # Low temperature for deterministic classification, low token count as we only need one word.
    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 1024,
        "top_p": 1.0,
    }

    # --- Job Execution ---
    # Instantiate the custom classifier.
    classifier = MyVertexTextClassifier(
        project_id=PROJECT_ID,
        location=LOCATION,
        generation_config=generation_config,
        thinking_budget=0  # Can be adjusted for thinking mode
    )

    try:
        # Create a GCS bucket for the job.
        print(f"\n--- Creating GCS bucket: gs://{bucket_name} ---")
        os.system(f"gsutil mb -l {LOCATION} -p {PROJECT_ID} gs://{bucket_name}")

        # Run the entire batch process: validate, upload, create job, monitor, and get results.
        results_df = classifier.run(
            model_id=MODEL_ID,
            data=my_sample_texts,
            input_uri=gcs_input_uri,
            output_uri=gcs_output_uri,
            local_save_path=local_save_path,
            generation_config=generation_config, # Pass for pre-flight validation check.
            poll_interval_seconds=60
        )

        # 3. Result verification and post-processing
        if results_df is not None:
            print(f"\n--- ✅ Job Completed: Detailed Processing Results (Model: {MODEL_ID}) ---")
            
            # The results are returned in the same order as the input data.
            for original_text, (_, row) in zip(my_sample_texts, results_df.iterrows()):
                print(f"\n[Original] \"{original_text}\"")
                try:
                    # The 'response' column in the DataFrame contains the parsed JSON from the API.
                    response_data = row['response']
                    
                    # Handle potential errors on a per-item basis.
                    if 'candidates' in response_data:
                        classification = response_data['candidates'][0]['content']['parts'][0]['text']
                        print(f"  ✅ Classification result: {classification.strip()}")
                    elif 'error' in response_data:
                        print(f"  🚨 API Error for this item: {response_data['error']['message']}")
                    else:
                        print(f"  ⚠️ Unexpected response format: {response_data}")

                except (KeyError, IndexError, TypeError) as e:
                    error_message = row.get('error', f'An exception occurred during parsing: {e}')
                    print(f"  🚨 Error processing this item: {error_message}")
        else:
            print("\n--- ❌ Job did not return results. ---")

    except Exception as e:
        print(f"\n--- ❌ An unexpected error occurred during the process: {e} ---")
        
    finally:
        # Clean up cloud resources with clear separation of responsibilities
        print("\n--- 🧹 Starting cleanup process ---")
        try:
            # 1. First, clean up processor resources (batch job and temporary files)
            if 'classifier' in locals():
                classifier.cleanup()
                
            # 2. Then, delete the GCS bucket created by this script
            print(f"\n--- Deleting GCS bucket: gs://{bucket_name} ---")
            os.system(f"gsutil -m rm -r gs://{bucket_name}")
            print(f"✅ Bucket gs://{bucket_name} deleted successfully")
                
        except Exception as cleanup_error:
            print(f"⚠️ Warning: Cleanup encountered an issue: {cleanup_error}")
            print("You may need to manually delete cloud resources to avoid charges.")
            
        print("--- ✨ All operations finished. ---")