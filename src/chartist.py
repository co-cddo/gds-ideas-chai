import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pydantic import BaseModel
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


@dataclass
class ModelInput:
    """Configuration class for AWS Bedrock model inputs.

    This class defines the structure and parameters for model invocation requests
    following AWS Bedrock's expected format.

    See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

    Attributes:
        messages (List[dict]): List of message objects with role and content
        anthropic_version (str): Version string for Anthropic models
        max_tokens (int): Maximum number of tokens in the response
        system (Optional[str]): System message for the model
        stop_sequences (Optional[List[str]]): Custom stop sequences
        temperature (Optional[float]): Sampling temperature
        top_p (Optional[float]): Nucleus sampling parameter
        top_k (Optional[int]): Top-k sampling parameter
        tools (Optional[List[dict]]): Tool definitions for structured outputs
        tool_choice (Optional[ToolChoice]): Tool selection configuration
    """

    # These are required
    messages: List[dict]
    anthropic_version: str = "bedrock-2023-05-31"
    max_tokens: int = 2000

    # Moddable parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    system: Optional[str] = None
    stop_sequences: Optional[List[str]] | None = None

    # tools: Optional[List[dict]] | None = None
    # tool_choice: Optional[ToolChoice] = None

    def to_dict(self):
        result = {k: v for k, v in self.__dict__.items() if v is not None}
        # if self.tool_choice:
        #    result["tool_choice"] = self.tool_choice.__dict__
        return result

    def to_json(self):
        return json.dumps(self.to_dict())


class chartist:
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize chartist with required configuration

        Parameters:
        region_name (str): AWS region name
        """
        # Load environment variables
        load_dotenv()

        # Get AWS profile from environment
        self.aws_profile = os.getenv("AWS_PROFILE")
        self.region_name = region_name

        self.visualizations = None

        # Configure AWS session
        try:
            self.session = boto3.Session(
                profile_name=self.aws_profile, region_name=self.region_name
            )

            # Initialize bedrock client
            self.bedrock = self.session.client(service_name="bedrock-runtime")

            print(
                f"Successfully configured AWS session using profile: {self.aws_profile}"
            )

        except Exception as e:
            print(f"Error configuring AWS session: {str(e)}")
            raise

    @staticmethod
    def get_preset_params(preset: str = "default") -> dict:
        """
        Set predefined parameter presets

        Parameters:
        preset: One of ["default", "creative", "precise", "detailed"]

        Returns:
        dict: Model parameters for the specified preset
        """
        presets = {
            "default": {"temperature": 0.5, "top_k": 250, "max_tokens": 2000},
            "creative": {"temperature": 1.0, "top_k": 300, "max_tokens": 3000},
            "precise": {"temperature": 0.2, "top_k": 100, "max_tokens": 2000},
            "detailed": {"temperature": 0.5, "top_k": 250, "max_tokens": 4000},
        }
        return presets.get(preset, presets["default"])

    @staticmethod
    def get_preset_description(preset: str) -> str:
        """Get description of what each preset does"""
        descriptions = {
            "default": "Balanced settings for general analysis",
            "creative": "Higher temperature for more creative suggestions",
            "precise": "Lower temperature for more focused analysis",
            "detailed": "Extended token limit for comprehensive analysis outputs",
        }
        return descriptions.get(preset, "Unknown preset")

    def generate_ideas(
        self, df: pd.DataFrame, prompt: str, preset: str = "default", **model_params
    ):
        """
        Send DataFrame to Claude via AWS Bedrock with preset or custom parameters and generate some ideas
        for charts.

        Parameters:
        df: DataFrame to analyze
        prompt: What you want Claude to do with the data
        preset: Preset configuration to use ("default", "creative", "precise", "detailed")
        **model_params: Optional parameters to override preset values
        """
        # Get preset parameters
        params = self.get_preset_params(preset)

        # Override preset with any custom parameters
        params.update(model_params)

        # Prepare DataFrame information
        data_info = {
            "data_preview": df.head().to_string(),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
            "summary": df.describe().to_string(),
            "sample_records": df.head(10).to_dict("records"),
        }

        # Construct the message
        message_content = f"""
            Here is the DataFrame information:
            
            Shape: {data_info['shape']}
            
            Columns and Types:
            {pd.Series(data_info['dtypes']).to_string()}
            
            Preview of first few rows:
            {data_info['data_preview']}
            
            Statistical Summary:
            {data_info['summary']}
            
            Please suggest different visualizations that would be useful for analyzing this data.
            For each visualization, provide a clear description of what it shows and why it's useful.
            Consider all possible meaningful visualizations that could provide insights from this data.
            """

        # Create ModelInput directly
        model_input = ModelInput(
            messages=[{"role": "user", "content": message_content}],
            **model_params,  # Pass any override parameters
        )

        try:
            response = self.bedrock.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=model_input.to_json(),
            )

            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"]

        except Exception as e:
            return f"Error processing request: {str(e)}"

    def visualise_ideas(self, df: pd.DataFrame, claude_response: str):
        """
        Generate Plotly visualization code based on Claude's response.

        Parameters:
        df: The full DataFrame to visualize
        claude_response: The response from generate_ideas

        Returns:
        Dict: Dictionary of visualization details including description and plotly code
        """
        prompt = f"""
            Based on the previous visualization suggestions:

            {claude_response}

            Please generate a JSON dictionary where each key is a sequential number (starting from 1) and each value is an object containing:
            1. 'description': The original description of the visualization
            2. 'plotly_code': The complete Plotly code using ONLY Plotly Graph Objects (go), not Plotly Express.

            Generate code for ALL suggested visualizations from the previous response.

            Requirements for the code:
            - Use 'import plotly.graph_objects as go' and 'from plotly.subplots import make_subplots'
            - Use Graph Objects (go) for all visualizations
            - Include proper layout configurations (titles, axes labels, etc.)
            - Assume the DataFrame is named 'df'
            - Include color configurations and hover templates
            - Ensure all code is complete and can be executed independently

            Format the response as a valid JSON string that can be parsed directly.
            Include ALL visualizations mentioned in the previous response.
        """

        # Create ModelInput for code generation
        model_input = ModelInput(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Lower temperature for more precise code generation
            max_tokens=4000,  # Increased tokens for detailed code
        )

        try:
            # Get response from Claude
            response = self.bedrock.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=model_input.to_json(),
            )

            # Parse response
            response_body = json.loads(response["body"].read())
            response_text = response_body["content"][0]["text"]

            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]

            # Parse JSON string
            visualizations = json.loads(json_str)

            return visualizations

        except Exception as e:
            return f"Error generating visualizations: {str(e)}"

    def generate_and_store_visualizations(self, df: pd.DataFrame, prompt: str):
        """
        Generate and store visualizations for later use

        Parameters:
        df: DataFrame to visualize
        prompt: Prompt for visualization suggestions

        Returns:
        dict: Dictionary of visualizations
        """
        try:
            # Get visualization ideas
            ideas_response = self.generate_ideas(df, prompt)

            # Generate visualization code
            visualizations = self.visualise_ideas(df, ideas_response)

            # Store visualizations in instance
            if isinstance(visualizations, dict):
                self.visualizations = visualizations
                return self.visualizations
            else:
                print("Error: Failed to generate valid visualizations")
                return None

        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            return None

    def list_visualizations(self, visualizations: dict):
        """
        List all available visualizations and their descriptions
        """
        if not visualizations or not isinstance(visualizations, dict):
            print("No valid visualizations available")
            return

        print("\nAvailable Visualizations:")
        for num, viz in visualizations.items():
            print(f"\n{num}. {viz['description']}")

    # Add helper method to execute visualization
    def execute_visualization(self, df: pd.DataFrame, viz_number: str):
        """
        Execute a specific visualization and return the figure

        Parameters:
        df: DataFrame to visualize
        viz_number: Visualization number as string

        Returns:
        plotly.graph_objects.Figure: The generated figure
        """
        if not self.visualizations:
            print("No visualizations stored. Generate visualizations first.")
            return None

        try:
            if viz_number in self.visualizations:
                viz = self.visualizations[viz_number]
                print(f"\nExecuting visualization {viz_number}:")
                print(f"Description: {viz['description']}\n")

                # Create namespace with required imports
                namespace = {"df": df, "go": go, "make_subplots": make_subplots}

                # Execute code
                exec(viz["plotly_code"], namespace)

                # Return the figure from the namespace
                if "fig" in namespace:
                    return namespace["fig"]
                else:
                    print("No figure found in visualization code")
                    return None
            else:
                print(f"Visualization {viz_number} not found")
                return None

        except Exception as e:
            print(f"Error executing visualization: {str(e)}")
            return None

    @staticmethod
    def create_sample_dataset():
        """Create a sample dataset for testing"""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        sample_df = pd.DataFrame(
            {
                "date": dates,
                "sales": np.random.normal(1000, 100, 100),
                "customers": np.random.randint(50, 200, 100),
                "satisfaction_score": np.random.uniform(3.5, 5.0, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

        return sample_df


def main():
    # Create an instance of the chartist class
    chart = chartist()

    # Create sample dataset
    sample_df = chart.create_sample_dataset()

    # Example prompt
    prompt = "Can you analyze the sales trends and provide basic statistical insights? Also, is there any correlation between sales and customer numbers?"

    # Get response from Claude
    response = chart.ask_claude(sample_df, prompt)

    # Print the response
    print(response)


if __name__ == "__main__":
    main()
