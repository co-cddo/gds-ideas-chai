# Import base requirements for data handling and AWS
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Optional, Any, Dict, List
from uuid import uuid4
import boto3
from dotenv import load_dotenv
from pathlib import Path

# Import agent dependencies
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_json_chat_agent,
)

# Import custom classes and tools
from chAI.config import Config
from chAI.bedrock import BedrockHandler
from tools.visualisation_formatter import create_formatting_tool
from tools.image_analysis_formatter import create_analysis_formatter_tool
from tools.save_plotly import create_save_plotly_tool
from tools.default_charts import PlotlyTemplates
from chAI.constants import (
    LLMModel,
    AWSRegion,
    APIVersion,
    MaxTokens,
    DataFrameLimits,
    ChartType,
)
from chAI.types import DataFrameInfo
from chAI.requests import DataFrameHandler, DataFrameJSONEncoder  # , ImageHandler

logger = logging.getLogger()

import pandas as pd
import base64


class ChAIError(Exception):
    """Base exception for chAI errors"""

    pass


class chAI:
    def __init__(self, region_name: AWSRegion = AWSRegion.US_EAST_1):
        """
        Initialises the chAI class with required configurations and tools.

        Args:
            region_name (str, optional): AWS region name. Defaults to "us-east-1".

        Notes:
            - Sets up configuration using Config class
            - Initialises Bedrock handler and runtime
            - Loads LLM model and prompt
            - Sets up visuasation tools and templates
            - Creates agent executor
        """
        logger.info("chAI Start")
        self.config = Config()
        self.bedrock = BedrockHandler(self.config)
        self.bedrock_runtime = self.bedrock.set_runtime()
        self.llm = self.bedrock.get_llm()
        self.prompt = hub.pull("hwchase17/react-chat-json")

        # Initialise handlers
        self.dataframe_handler = DataFrameHandler()
        # self.image_handler = ImageHandler()

        self.tools = [
            create_formatting_tool(),
            create_analysis_formatter_tool(),
            create_save_plotly_tool(),
        ]
        self.plotly_templates = PlotlyTemplates()
        self.agent_executor = self.set_agent_executor()

        # Set up holder for visualisations
        self.visualisations = None

    def set_agent_executor(self, verbose=False, handle_parse=True):
        """
        Sets up the LangChain agent executor with specified tools and configurations.

        Args:
            verbose (bool, optional): Enable verbose output. Defaults to False.
            handle_parse (bool, optional): Enable parsing error handling. Defaults to True.

        Returns:
            AgentExecutor: Configured agent executor instance.

        Raises:
            Exception: If there's an error setting up the agent executor.
        """
        logger.info("Setting up chAI agent")
        try:
            agent = create_json_chat_agent(self.llm, self.tools, self.prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=verbose,
                handle_parsing_errors=handle_parse,
            )
            logger.info("chAI agent successfully set up")
            return executor
        except Exception as e:
            logger.error(f"Error setting up chAI agent: {str(e)}")
            raise

    def encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encodes an image file to Base64 format for Claude's multi-modal API.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64 encoded image string.

        Raises:
            Exception: If there's an error reading or encoding the image.
        """
        logger.info("Encoding image to base64")
        try:
            # Create image content string
            path = Path(image_path)
            with path.open("rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise ChAIError(f"Failed to encode image: {e}")

    def analyse_image(
        self, base64_data: str, custom_prompt: Optional[str] = None
    ) -> str:
        """
        Analyses an image using AWS Bedrock's Claude model and returns a structured analysis.

            This method is designed to handle full base64 image data directly, bypassing the typical
            agent tool limitations that might truncate the image data. It sends the image to Claude
            along with a structured prompt for analysis.

            Args:
                base64_data (str): The base64-encoded image data to be analyzed.
                custom_prompt (str, optional): Additional specific requirements or questions to be
                    addressed in the analysis. Defaults to None.

            Returns:
                str: A structured analysis of the image including:
                    - General description
                    - Chart type identification
                    - Axes analysis
                    - Key insights and patterns
                    - Plotly recreation code
                    - Required data structure description
                    If an error occurs, returns an error message string.

            Raises:
                No exceptions are raised directly; all exceptions are caught and returned as error messages.

            Example:
                >>> analyser = YourClass(config)
                >>> base64_image = "base64_encoded_image_data"
                >>> custom_requirements = "Focus on trend analysis and seasonal patterns"
                >>> result = analyser.analyse_image(base64_image, custom_requirements)
                >>> print(result)
                # Description
                This image shows a time series plot...
                # Chart Analysis
                ## Type
                Line plot...

            Notes:
                - The method uses AWS Bedrock's Claude model specified in the configuration
                - The analysis follows a structured format with markdown-style sections
                - Maximum token limit is set to 2000 for the response
                - Requires valid AWS credentials and permissions to access Bedrock
        """
        try:
            if not base64_data:
                return "Error: No image data provided"

            analysis_prompt = f"""Analyse this image and provide a detailed analysis using the following structure:

                # Description
                [Provide a detailed description of what the image shows]

                # Chart Analysis
                ## Type
                [Specify the type of visualisation (e.g., bar chart, line plot, scatter plot)]

                ## Axes
                [List all axes and what they represent]

                ## Insights
                [Consider the following specific requirements in your analysis:
                {custom_prompt if custom_prompt else "No additional specific requirements stated"}
            
                Based on these requirements (if provided) and the image, provide detailed insights such as key patterns, trends or insights visible in the chart]

                # Plotly Recreation
                ## Code
                ```python
                [Provide a complete Plotly code snippet that could recreate this visualisation including the visible values for each variable]
                ```

                ## Data Structure
                [Describe the data structure needed for the Plotly code]"""

            body = {
                "anthropic_version": APIVersion.BEDROCK.value,
                "max_tokens": MaxTokens.DEFAULT,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_data,
                                },
                            },
                            {"type": "text", "text": analysis_prompt},
                        ],
                    }
                ],
            }

            body_bytes = json.dumps(body).encode("utf-8")

            logger.debug(f"Request body: {json.dumps(body, indent=2)}")

            # Call Bedrock
            response = self.bedrock_runtime.invoke_model(
                modelId=self.config.LLM_MODEL, body=body_bytes
            )

            response_body = json.loads(response.get("body").read())
            return response_body.get("content", [])[0].get("text", "")

        except Exception as e:
            logger.error(f"Error in analyse_image: {str(e)}")
            raise ChAIError(f"Failed to analyse image: {e}")

    def handle_request(
        self,
        data: Optional[pd.DataFrame] = None,
        prompt: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        chart_type: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Processes user requests based on input type and generates appropriate visualisations.

        Args:
            data (Optional[pd.DataFrame]): Input data for analysis.
            prompt (Optional[str]): User instructions for visualisation.
            image_path (Optional[Union[str, Path]]): Path to image for analysis.
            chart_type (Optional[ChartType]): Specific chart type from ChartType enum.
            output_path (Optional[Union[str, Path]]): Output directory for saved visualisations.
            **kwargs (Any): Additional keyword arguments for the LLM.

        Returns:
            Dict[str, Any]: Agent response containing:
                - Analysis results
                - Generated visualisations
                - Output file paths

        Raises:
        ChAIError: If there's an error processing the request.

        Notes:
            - Handles different input types (DataFrame, image, chart type)
            - Limits DataFrame processing to 100 rows
            - Uses appropriate templates based on chart type specified
            - Saves visualisations to specified output path
        """

        base_prompt = f"""
            User Prompt:
            {prompt}
            """

        if isinstance(data, pd.DataFrame):
            logger.info("Detected DataFrame input. Preparing to analyse...")
            final_prompt = self.dataframe_handler.dataframe_request(data, base_prompt)

        elif isinstance(image_path, str):
            logger.info("Detected image location input. Preparing to review...")
            final_prompt = self._handle_image_request(image_path, output_path)
            print(final_prompt)

        elif chart_type:
            logger.info(f"Processing chart type request: {chart_type}")
            final_prompt = self._handle_chart_request(chart_type, prompt, output_path)

        else:
            raise ValueError("No valid input provided")

        # Send to the agent executor
        try:
            logger.info("Sending prompt and data to agent executor...")
            response = self.agent_executor.invoke({"input": final_prompt})
            print(response)
            return response["output"]
        except Exception as e:
            logger.error(f"Error in handle_request: {str(e)}")
            raise ChAIError(f"Failed to process request: {e}")

    def _handle_image_request(
        self, image_path: Union[str, Path], output_path: Optional[Union[str, Path]]
    ) -> str:
        """Handle image analysis request."""
        image_base64 = self.encode_image(image_path)
        image_response = self.analyse_image(image_base64)

        return f"""
            Review the supplied image information and create an interactive visualisation that matches it as closely as possible.
            
            IMPORTANT: You MUST use the format_image_analysis_output tool first to structure the image analysis.
            
            1. Format the analysis:
            - Use the format_image_analysis_output tool with this exact input: {image_response}
            - Store the formatted JSON output for use in subsequent steps
            - This step is mandatory and must be completed first
            
            2. Using the formatted analysis from step 1:
            - Reference the chart_analysis.type to identify the visualisation type
            - Use chart_analysis.axes to understand data relationships
            - Review insights from chart_analysis.insights
            - Check plotly_recreation for specific implementation details
            
            3. Create a professional visualisation that:
            - Matches the identified chart type from the formatted analysis
            - Replicates the color scheme and styling
            - Maintains proper layout and formatting
            - Includes appropriate labels and legends
            - Uses 'plotly_white' as the base template
            
            4. Ensure the visualisation includes:
            - Proper title and axis formatting
            - Consistent text positioning and styling
            - Appropriate data representation
            - Professional appearance and readability
            
            5. Use save_plotly_visualisation to create an HTML file in the output_path folder.
            {f'Use this exact output path: {str(output_path)}' if output_path else 'Use the default path in save_plotly_visualisation'}
            
            6. Return a JSON dictionary with exactly this structure:
            {{
                "analysis": "## Insights\\n1. <insight1>\\n2. <insight2>\\n...",
                "path": The exact path returned by save_plotly_visualisation,
                "code": The complete plotly code used to create the visualisation
            }}
            
            IMPORTANT: The analysis section in the JSON response should maintain the exact markdown formatting with the "## Insights" header and numbered list format.
            IMPORTANT: Use the exact path and code returned by the save_plotly_visualisation tool in your response.
            
            You MUST use the format_image_analysis_output tool for step 1 before proceeding with the visualisation creation.
            Focus on creating a visualisation that accurately represents the original image while maintaining professional standards and interactive functionality.
        """

    def _handle_chart_request(
        self,
        chart_type: ChartType,
        prompt: Optional[str],
        output_path: Optional[Union[str, Path]],
    ) -> str:
        """Handle specific chart type request."""
        templates = self.plotly_templates.get_templates()
        chart_type_mapping = {
            ChartType.BAR: "bar_chart",
            ChartType.HISTOGRAM: "histogram",
            ChartType.SCATTER: "scatter_plot",
            ChartType.LINE: "line_chart",
        }

        template_key = chart_type_mapping.get(chart_type, "bar_chart")
        if template_key != chart_type_mapping.get(chart_type):
            logger.warning(
                f"Unsupported chart type: {chart_type}. Defaulting to bar chart."
            )

        template_code = templates[template_key]
        print(template_code)

        return f"""
            Create a default {chart_type} chart visualisation using this template as reference:

            # Template Code:
            {template_code}
            
            1. Use this template code as a starting point
            2. Consider the following specific requirements in adapting the code to the user's needs:
            {prompt if prompt else "No additional specific requirements stated"}
            3. Maintain the professional styling while adjusting for your specific needs and those of the user.
            4. Use the save_plotly_visualisatio tool to create an HTML file in the output_path folder.
            {f'Use this exact output path: {str(output_path)}' if output_path else 'Use the default path in the save_plotly_visualisation tool'}
            5. The save_plotly_visualisation tool will return a dictionary containing the path and code. Use this dictionary 
            to build a JSON dictionary like the one below:
            {{
                "path": "The exact path returned by the save_plotly_visualisation tool",
                "code": "The complete plotly code used to create the visualisation by the save_plotly_visualisation tool"
            }}
            
            DO NOT include any text before or after the JSON.
            DO NOT add any explanations or descriptions.
            ONLY return the JSON dictionary.
        """
