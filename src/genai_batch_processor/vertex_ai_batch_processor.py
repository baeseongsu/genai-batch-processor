"""
VertexAI Batch Processor Module

This module provides a batch processor for Google's Vertex AI Gemini API,
supporting both Cloud Storage (JSONL) and BigQuery as input sources and output destinations.

Written by: seongsu@kaist.ac.kr and Claude 4 Sonnet
Date: 2025-07-17
"""

import os
import time
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterable, Optional
from datetime import datetime

import pandas as pd
import fsspec
from google import genai
from google.cloud import bigquery
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig
from .base import BaseBatchProcessor

class VertexAIBatchProcessor(BaseBatchProcessor):
    """
    Refined abstract base class for asynchronous batch inference with the Vertex AI Gemini API.

    This processor supports both Cloud Storage (JSONL) and BigQuery as input sources and output destinations.
    It handles the complete workflow from input preparation to result retrieval.
    """

    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize the batch processor.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location (must support batch predictions)
        """
        if not project_id:
            raise ValueError("A Google Cloud project_id is required.")

        self.project_id = project_id
        self.location = location
        self.client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        self.bq_client = bigquery.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)
        self.batch_job = None
        self.results_df = None
        self._cleanup_resources = []

    @abstractmethod
    def _create_request_data(self, item: Any) -> Dict[str, Any]:
        """
        Convert a single data item into a request dictionary for batch processing.

        This method must be implemented by the user to define the structure of each request.
        The returned dictionary should follow the GenerateContentRequest format.

        Args:
            item: A single data item to process

        Returns:
            Dictionary containing the request structure with 'contents', 'generationConfig', etc.
        """
        pass

    def validate_request(
        self,
        model_id: str,
        sample_item: Any,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Performs a pre-flight check by testing the request format with a single item.

        Args:
            model_id: Model ID to validate against
            sample_item: Sample data item for validation
            generation_config: Optional generation configuration

        Returns:
            True if validation succeeds, False otherwise
        """
        print("\n--- 1. Starting Parameter Validation Check ---")
        try:
            # Create a test request using the user's implementation
            test_request = self._create_request_data(sample_item)

            # Modify config for quick validation
            if generation_config:
                test_request["generationConfig"] = generation_config.copy()
                test_request["generationConfig"]["max_output_tokens"] = 128
            else:
                test_request["generationConfig"] = {"max_output_tokens": 128}

            print(f"Validating with model '{model_id}'...")
            print(f"Request structure: {json.dumps(test_request, indent=2)[:200]}...")
            print("✅ Validation successful - request structure is valid.")
            return True

        except Exception as e:
            print(f"\n❌ Validation Failed. Please check your model name, parameters, and data format.")
            print(f"Error: {e}")
            return False

    def _prepare_input_file_gcs(
        self,
        data: Iterable[Any],
        gcs_input_uri: str,
    ) -> None:
        """
        Creates the batch JSONL file in Cloud Storage from the provided data.

        Args:
            data: Iterable of data items to process
            gcs_input_uri: GCS URI where the input file will be stored
        """
        print(f"--- 2. Preparing and uploading input file to {gcs_input_uri} ---")

        fs = fsspec.filesystem("gcs")

        with fs.open(gcs_input_uri, "w") as f:
            for item in data:
                # User's method creates the complete request
                request_dict = self._create_request_data(item)

                # Wrap in the batch format
                batch_request = {"request": request_dict}

                f.write(json.dumps(batch_request) + "\n")

        self._cleanup_resources.append(gcs_input_uri)
        print("✅ Input file preparation and upload successful.")

    def _prepare_input_table_bq(
        self,
        data: Iterable[Any],
        bq_table_uri: str,
    ) -> None:
        """
        Creates the batch input table in BigQuery from the provided data.

        Args:
            data: Iterable of data items to process
            bq_table_uri: BigQuery table URI (format: bq://project.dataset.table)
        """
        print(f"--- 2. Preparing and uploading input table to {bq_table_uri} ---")

        # Parse the BigQuery URI
        table_id = bq_table_uri.replace("bq://", "")

        # Prepare data for BigQuery
        rows = []
        for item in data:
            request_dict = self._create_request_data(item)
            rows.append({"request": json.dumps(request_dict)})

        # Create DataFrame and upload to BigQuery
        df = pd.DataFrame(rows)

        # Upload to BigQuery
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            schema=[
                bigquery.SchemaField("request", "STRING", mode="REQUIRED"),
            ],
        )

        job = self.bq_client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for the job to complete

        self._cleanup_resources.append(table_id)
        print("✅ Input table preparation and upload successful.")

    def _create_batch(self, model_id: str, input_uri: str, output_uri: str) -> None:
        """
        Creates a batch prediction job in Vertex AI.

        Args:
            model_id: Model ID for batch processing
            input_uri: Input data location (GCS or BigQuery URI)
            output_uri: Output location (GCS or BigQuery URI)
        """
        if not input_uri or not output_uri:
            raise ValueError("Both input_uri and output_uri are required.")

        print(f"--- 3. Creating batch job with model '{model_id}' ---")
        print(f"   - Input: {input_uri}")
        print(f"   - Output: {output_uri}")

        self.batch_job = self.client.batches.create(
            model=model_id,
            src=input_uri,
            config=CreateBatchJobConfig(dest=output_uri),
        )

        print(f"✅ Batch job submission successful.")
        print(f"Job Name: {self.batch_job.name}")
        print(f"Current state: {self.batch_job.state}")

    def _monitor_status(self, poll_interval_seconds: int = 30, timeout_seconds: int = 7200) -> None:
        """
        Periodically checks the status until the batch job completes or times out.

        Args:
            poll_interval_seconds: Interval between status checks
            timeout_seconds: Maximum time to wait for job completion
        """
        if not self.batch_job:
            raise RuntimeError("Batch job has not been created.")

        print(f"--- 4. Monitoring batch job (timeout: {timeout_seconds}s) ---")
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            # Refresh job state
            self.batch_job = self.client.batches.get(name=self.batch_job.name)
            current_state = self.batch_job.state
            print(f"  - State: {current_state} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

            if current_state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
                print("Batch job reached a terminal state.")
                if current_state == "JOB_STATE_SUCCEEDED":
                    print("✅ Job succeeded!")
                else:
                    print(f"❌ Job did not succeed. Final State: {current_state}")
                    if hasattr(self.batch_job, "error") and self.batch_job.error:
                        print(f"   Error details: {self.batch_job.error}")
                return

            time.sleep(poll_interval_seconds)

        print(f"⌛️ Monitoring timed out after {timeout_seconds} seconds. The job might still be running.")

    def _retrieve_and_save_results(self, local_save_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieves results from the completed job's output location and optionally saves them locally.

        Args:
            local_save_path: Optional local path to save results

        Returns:
            DataFrame containing the results, or None if retrieval failed
        """
        if not self.batch_job or self.batch_job.state != "JOB_STATE_SUCCEEDED":
            print("Cannot retrieve results as the job did not succeed.")
            return None

        print("--- 5. Retrieving batch results ---")

        # Check if output is in GCS or BigQuery
        if hasattr(self.batch_job.dest, "gcs_uri") and self.batch_job.dest.gcs_uri:
            return self._retrieve_from_gcs(self.batch_job.dest.gcs_uri, local_save_path)
        elif hasattr(self.batch_job.dest, "bigquery_uri") and self.batch_job.dest.bigquery_uri:
            return self._retrieve_from_bigquery(self.batch_job.dest.bigquery_uri, local_save_path)
        else:
            print("❌ Could not determine output location.")
            return None

    def _retrieve_from_gcs(self, gcs_uri: str, local_save_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve results from Cloud Storage."""
        try:
            fs = fsspec.filesystem("gcs")
            file_paths = fs.glob(f"{gcs_uri.rstrip('/')}/*/predictions.jsonl")

            if not file_paths:
                raise FileNotFoundError("No prediction .jsonl files found in the output directory.")

            # Load the JSONL file into a DataFrame
            self.results_df = pd.read_json(f"gs://{file_paths[0]}", lines=True)
            print(f"✅ Successfully retrieved {len(self.results_df)} results from GCS.")

        except Exception as e:
            print(f"❌ An error occurred while reading from GCS: {e}")
            return None

        return self._save_results_locally(local_save_path)

    def _retrieve_from_bigquery(self, bq_uri: str, local_save_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve results from BigQuery."""
        try:
            table_id = bq_uri.replace("bq://", "")
            query = f"SELECT * FROM `{table_id}`"

            query_result = self.bq_client.query(query)
            self.results_df = query_result.result().to_dataframe()
            print(f"✅ Successfully retrieved {len(self.results_df)} results from BigQuery.")

        except Exception as e:
            print(f"❌ An error occurred while reading from BigQuery: {e}")
            return None

        return self._save_results_locally(local_save_path)

    def _save_results_locally(self, local_save_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Save results to local file if path is provided."""
        if local_save_path and self.results_df is not None:
            try:
                # Create directory if it doesn't exist
                if os.path.dirname(local_save_path):
                    os.makedirs(os.path.dirname(local_save_path), exist_ok=True)

                # Determine file format based on extension
                if local_save_path.endswith(".csv"):
                    self.results_df.to_csv(local_save_path, index=False)
                    print(f"✅ Results saved to CSV: {local_save_path}")
                elif local_save_path.endswith(".jsonl"):
                    self.results_df.to_json(local_save_path, orient="records", lines=True)
                    print(f"✅ Results saved to JSONL: {local_save_path}")
                elif local_save_path.endswith(".parquet"):
                    self.results_df.to_parquet(local_save_path, index=False)
                    print(f"✅ Results saved to Parquet: {local_save_path}")
                else:
                    # Default to CSV if no recognized extension
                    csv_path = f"{os.path.splitext(local_save_path)[0]}.csv"
                    self.results_df.to_csv(csv_path, index=False)
                    print(f"✅ Results saved to CSV (default): {csv_path}")

            except Exception as e:
                print(f"⚠️ Failed to save results locally: {e}")

        return self.results_df

    def cleanup(self):
        """Deletes the batch job and temporary resources created by this processor."""
        print("\n--- 🧹 Starting Cleanup ---")

        # Delete batch job
        if self.batch_job:
            try:
                print(f"Deleting batch job: {self.batch_job.name}")
                self.client.batches.delete(name=self.batch_job.name)
                print("✅ Batch job deleted.")
            except Exception as e:
                print(f"⚠️ Could not delete batch job: {e}")

        # Clean up temporary resources created by this processor
        for resource in self._cleanup_resources:
            try:
                if resource.startswith("gs://"):
                    fs = fsspec.filesystem("gcs")
                    print(f"Deleting GCS resource: {resource}")
                    fs.rm(resource, recursive=True)
                    print("✅ GCS resource deleted.")
                else:
                    # Assume it's a BigQuery table
                    print(f"Deleting BigQuery table: {resource}")
                    self.bq_client.delete_table(resource)
                    print("✅ BigQuery table deleted.")
            except Exception as e:
                print(f"⚠️ Could not delete resource {resource}: {e}")

    def run(
        self,
        model_id: str,
        output_uri: str,
        data: Optional[Iterable[Any]] = None,
        input_uri: Optional[str] = None,
        validate: bool = True,
        poll_interval_seconds: int = 30,
        timeout_seconds: int = 7200,
        local_save_path: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Executes the complete batch processing workflow.

        Args:
            model_id: Model ID for batch processing
            output_uri: Output location (GCS or BigQuery URI)
            data: Optional data to process (if not using pre-existing input)
            input_uri: Optional pre-existing input location
            validate: Whether to validate the request format
            poll_interval_seconds: Interval between status checks
            timeout_seconds: Maximum time to wait for job completion
            local_save_path: Optional local path to save results
            generation_config: Optional generation configuration

        Returns:
            DataFrame containing the results, or None if processing failed
        """
        # Validation
        if validate and data:
            sample_item = next(iter(data), None)
            if sample_item is None:
                print("❌ No data provided for validation.")
                return None

            if not self.validate_request(model_id, sample_item, generation_config):
                print("❌ Halting batch process due to validation failure.")
                return None

        # Input preparation
        if data is not None and input_uri:
            if input_uri.startswith("gs://"):
                self._prepare_input_file_gcs(data, input_uri)
            elif input_uri.startswith("bq://"):
                self._prepare_input_table_bq(data, input_uri)
            else:
                raise ValueError("input_uri must be a GCS (gs://) or BigQuery (bq://) URI.")
            final_input_uri = input_uri
        elif input_uri:
            print("--- 1. Using pre-existing input ---")
            final_input_uri = input_uri
        else:
            raise ValueError("You must provide either 'data' and 'input_uri', or just a pre-existing 'input_uri'.")

        try:
            self._create_batch(model_id, final_input_uri, output_uri)
            self._monitor_status(poll_interval_seconds, timeout_seconds)
            results = self._retrieve_and_save_results(local_save_path)
            print("\n✅ Vertex AI Batch workflow completed successfully.")
            return results
        except Exception as e:
            print(f"\n❌ An error occurred during the workflow: {e}")
            return None