"""
OpenAI Batch Processor Module

This module provides a batch processor for OpenAI's Batch API,
supporting both standard OpenAI and Azure OpenAI services.

Written by: seongsu@kaist.ac.kr and Claude 4 Sonnet
Date: 2025-07-17
"""

import os
import sys
import time
import json
import tempfile
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterable, Optional

from openai import OpenAI, BadRequestError
from openai import AzureOpenAI

from .base import BaseBatchProcessor


class OpenAIBatchProcessor(BaseBatchProcessor):
    """
    Abstract base class for processing asynchronous inference using the OpenAI Batch API.

    This class encapsulates the entire workflow including batch file creation, upload,
    batch job creation, status monitoring, result retrieval, and saving.

    Usage:
    1. Create a new class that inherits from this class.
    2. Implement the `_create_request` abstract method to fit your data format.
       This method should convert a single data item to the JSON line format required by the Batch API.
    3. Create an instance of your implemented class and call the `run` method.
    """

    def __init__(self, api_key: str = None, azure_endpoint: str = None, api_version: str = None, azure_deployment: str = None):
        """
        Initialize the class and set up the OpenAI or Azure OpenAI client.

        :param api_key: OpenAI API key. If None, uses the `OPENAI_API_KEY` environment variable for OpenAI or `AZURE_OPENAI_API_KEY` for Azure OpenAI.
        :param azure_endpoint: Azure OpenAI endpoint URL. If provided, will use Azure OpenAI.
        :param api_version: Azure OpenAI API version. Only used with Azure OpenAI.
        :param azure_deployment: Azure OpenAI deployment name. Only used with Azure OpenAI.
        """
        # Check if Azure OpenAI configuration is provided
        if azure_endpoint:
            if api_key is None:
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Azure OpenAI API key is required. Pass it as an argument or set the AZURE_OPENAI_API_KEY environment variable.")
            
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version or "2024-02-01",
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment
            )
            self.is_azure = True
            self.azure_deployment = azure_deployment
        else:
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required. Pass it as an argument or set the OPENAI_API_KEY environment variable.")
            
            self.client = OpenAI(api_key=api_key)
            self.is_azure = False
            self.azure_deployment = None
        
        self.input_file_id = None
        self.batch_id = None
        self.results = []
        self.errors = []
        self.final_batch_status = None

    @abstractmethod
    def _create_request(self, item: Any, index: int, **kwargs) -> Dict:
        """
        Convert a single data item from the user into OpenAI Batch API request format.
        This method must be implemented in subclasses.

        :param item: A single item from the data collection to be processed.
        :param index: Index of the data item. Useful for generating `custom_id`.
        :return: Dictionary corresponding to each line in the Batch API.
                 Example: {"custom_id": f"request-{index}", "method": "POST", ...}
        """
        pass

    def validate_request(self, sample_item: Any, **kwargs) -> bool:
        """
        Sends a single synchronous request to validate API parameters before starting a batch.

        :param sample_item: A single item from the dataset to use for validation.
        :param kwargs: Additional arguments, including the model name.
        :return: True if the request is successful, False otherwise.
        """
        print("\n--- Starting Parameter Validation Check ---")
        try:
            # Create a request for the single sample item. Index is 0 for simplicity.
            request_data = self._create_request(sample_item, 0, **kwargs)
            request_body = request_data['body']
            
            # For Azure OpenAI, use the deployment name as the model
            if self.is_azure and self.azure_deployment:
                request_body['model'] = self.azure_deployment
            
            print(f"Validating with model '{request_body.get('model')}' using a sample request...")

            # Make a synchronous call to the chat completions endpoint
            self.client.chat.completions.create(**request_body)
            
            print("✅ Validation successful: API parameters are valid for the selected model.")
            return True

        except BadRequestError as e:
            print("\n❌ Validation Failed: The API request was rejected.")
            print("This often happens due to unsupported parameters for the selected model.")
            print(f"Error Details: {e.message}")
            return False
        except Exception as e:
            print(f"\n❌ An unexpected error occurred during validation: {e}")
            return False

    def _prepare_and_upload_file(self, data: Iterable[Any], **kwargs) -> None:
        """
        Create a .jsonl file from the data and upload it to OpenAI.

        :param data: Iterable of data to be processed (e.g., list).
        """
        print("1. Starting batch input file preparation and upload...")
        
        # Use temporary file to store .jsonl data
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl", encoding='utf-8') as tmp_file:
            temp_filename = tmp_file.name
            for i, item in enumerate(data):
                request_line = self._create_request(item, i, **kwargs)
                
                # For Azure OpenAI, ensure the model field uses the deployment name
                if self.is_azure and self.azure_deployment:
                    request_line['body']['model'] = self.azure_deployment
                
                tmp_file.write(json.dumps(request_line) + "\n")
        
        print(f"Temporary file created: {temp_filename}")

        # Upload file
        try:
            with open(temp_filename, "rb") as f:
                batch_input_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            self.input_file_id = batch_input_file.id
            print(f"File upload successful. File ID: {self.input_file_id}")
        finally:
            # Delete temporary file
            os.remove(temp_filename)
            print("Temporary file deleted.")

    def _create_batch(self, endpoint: str, completion_window: str = "24h", metadata: Dict = None) -> None:
        """
        Create a batch job using the uploaded file.

        :param endpoint: API endpoint (e.g., "/v1/chat/completions").
        :param completion_window: Completion window (currently only "24h" is supported).
        :param metadata: Metadata to add to the batch.
        """
        if not self.input_file_id:
            raise RuntimeError("File must be uploaded first. Call _prepare_and_upload_file().")

        print("2. Starting batch job creation...")
        batch = self.client.batches.create(
            input_file_id=self.input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata
        )
        self.batch_id = batch.id
        print(f"Batch job creation successful. Batch ID: {self.batch_id}, Current status: {batch.status}")

    def _monitor_status(self, poll_interval_seconds: int = 30) -> None:
        """
        Periodically check the status until the batch job is completed.

        :param poll_interval_seconds: Status check interval in seconds.
        """
        if not self.batch_id:
            raise RuntimeError("Batch job has not been created. Call _create_batch().")

        print(f"3. Starting batch status monitoring... (Check interval: {poll_interval_seconds} seconds)")
        while True:
            batch = self.client.batches.retrieve(self.batch_id)
            self.final_batch_status = batch.status
            print(f"  - Current batch status: {self.final_batch_status} ({time.strftime('%Y-%m-%d %H:%M:%S')})")

            if self.final_batch_status in ["completed", "failed", "expired", "cancelled"]:
                print("Batch job has reached a terminal state.")
                break
            
            time.sleep(poll_interval_seconds)

    def _retrieve_and_save_results(self, output_path: str = None) -> None:
        """
        Retrieve and save the results of the completed batch.

        :param output_path: File path to save results. If None, results are not saved.
        """
        if not self.batch_id:
            raise RuntimeError("Batch ID is missing.")

        print("4. Starting batch result retrieval...")
        batch = self.client.batches.retrieve(self.batch_id)

        if batch.status != "completed":
            print(f"Cannot retrieve results as batch is not in 'completed' status. Final status: {batch.status}")
            if batch.error_file_id:
                print("Error file exists. Checking error content.")
            else:
                return

        # Process result file
        if batch.output_file_id:
            file_content_response = self.client.files.content(batch.output_file_id)
            content = file_content_response.text
            
            self.results = [json.loads(line) for line in content.strip().split('\n') if line]
            print(f"Successfully retrieved {len(self.results)} results.")

            if output_path:
                results_filename = f"{output_path}_results.json"
                output_dir = os.path.dirname(results_filename)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(results_filename, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, ensure_ascii=False, indent=4)
                print(f"Results saved to '{results_filename}' file.")
        else:
            print("No result file (output_file_id) available.")

        # Process error file
        if batch.error_file_id:
            error_content_response = self.client.files.content(batch.error_file_id)
            error_content = error_content_response.text
            
            self.errors = [json.loads(line) for line in error_content.strip().split('\n') if line]
            print(f"Found {len(self.errors)} errors in error file.")

            if output_path:
                errors_filename = f"{output_path}_errors.jsonl"
                output_dir = os.path.dirname(errors_filename)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(errors_filename, 'w', encoding='utf-8') as f:
                    f.write(error_content)
                print(f"Error content saved to '{errors_filename}' file.")
        else:
            print("No error file (error_file_id) available.")

    def run(self, data: Iterable[Any], endpoint: str, output_path_prefix: str = "batch_output", poll_interval_seconds: int = 60, validate: bool = True, **kwargs):
        """
        Execute the complete batch processing workflow.

        :param data: Iterable of data to be processed.
        :param endpoint: API endpoint (e.g., "/v1/chat/completions").
        :param output_path_prefix: File name prefix to use when saving result and error files. Can include a path.
        :param poll_interval_seconds: Status check interval in seconds.
        :param validate: If True, runs a pre-flight synchronous request to validate parameters. Note: Skipped for Azure OpenAI as batch service doesn't share single inference API keys.
        :param kwargs: Additional arguments to be passed to the `_create_request` method.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path_with_timestamp = f"{output_path_prefix}_{timestamp}"
        
        try:
            data_list = list(data)
            if not data_list:
                print("Data is empty, skipping process.")
                return [], []

            if validate and not self.is_azure:
                is_valid = self.validate_request(data_list[0], **kwargs)
                if not is_valid:
                    print("\nHalting batch process due to validation failure.")
                    return [], []
                print("--- Validation Complete ---")
            elif validate and self.is_azure:
                print("--- Skipping validation for Azure OpenAI (batch service doesn't share single inference API keys) ---")
            
            self._prepare_and_upload_file(data_list, **kwargs)
            self._create_batch(endpoint)
            self._monitor_status(poll_interval_seconds)
            self._retrieve_and_save_results(output_path_with_timestamp)
            
            print("\nBatch processing workflow completed.")

        except Exception as e:
            print(f"\nError occurred during workflow execution: {e}")
            import traceback
            traceback.print_exc()
        
        return self.results, self.errors


class AzureOpenAIBatchProcessor(OpenAIBatchProcessor):
    """
    Convenience class for Azure OpenAI batch processing.
    
    This class provides a simplified interface for Azure OpenAI by automatically
    setting up the Azure client configuration.
    """
    
    def __init__(self, api_key: str, azure_endpoint: str, azure_deployment: str, api_version: str = "2024-02-01"):
        """
        Initialize Azure OpenAI batch processor.
        
        :param api_key: Azure OpenAI API key.
        :param azure_endpoint: Azure OpenAI endpoint URL.
        :param azure_deployment: Azure OpenAI deployment name.
        :param api_version: Azure OpenAI API version.
        """
        super().__init__(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=azure_deployment
        )
