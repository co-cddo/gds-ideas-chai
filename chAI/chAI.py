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
from chAI.requests import DataFrameHandler, DataFrameJSONEncoder

logger = logging.getLogger()

import pandas as pd
import base64


class ChAIError(Exception):
    """Base exception for chAI errors"""

    pass


class chAI:
    def __init__(self, region_name: AWSRegion = AWSRegion.US_EAST_1):
        """
        Initializes the chAI class with required configurations and tools.

        Args:
            region_name (str, optional): AWS region name. Defaults to "us-east-1".

        Notes:
            - Sets up configuration using Config class
            - Initializes Bedrock handler and runtime
            - Loads LLM model and prompt
            - Sets up visualization tools and templates
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
            prompt (Optional[str]): User instructions for visualization.
            image_path (Optional[Union[str, Path]]): Path to image for analysis.
            chart_type (Optional[ChartType]): Specific chart type from ChartType enum.
            output_path (Optional[Union[str, Path]]): Output directory for saved visualizations.
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

        elif chart_type:
            logger.info(f"Processing chart type request: {chart_type.value}")
            final_prompt = self._handle_chart_request(chart_type, prompt, output_path)

        else:
            raise ValueError("No valid input provided")

        # Send to the agent executor
        try:
            logger.info("Sending prompt and data to agent executor...")
            response = self.agent_executor.invoke({"input": final_prompt})
            print(response)
            return response
        except Exception as e:
            logger.error(f"Error in handle_request: {str(e)}")
            raise ChAIError(f"Failed to process request: {e}")

    def _handle_dataframe_request(self, data: pd.DataFrame, base_prompt: str) -> str:
        """Handle DataFrame analysis request."""
        if len(data) > DataFrameLimits.MAX_ROWS:
            logger.info(
                f"DataFrame has more than {DataFrameLimits.MAX_ROWS} rows. Trimming for processing."
            )
            data = data.head(DataFrameLimits.MAX_ROWS)

        dataframe_json = data.to_json(orient="split")

        dataframe_prompt = f"""
            DataFrame Information (Top {len(data)} Rows):
            The following is a JSON representation of the DataFrame. Use this to suggest suitable visualisations:
            {dataframe_json}

            Instructions:
            1. Analyse the DataFrame to understand its structure and content
            2. Suggest meaningful visualisations based on the data and user's instructions
            3. For each visualisation, include:
            - Clear purpose
            - Chart type
            - Variables used
            - Expected insights
            4. Use the format_visualisation_output tool to structure your response
            5. Make sure to provide concrete, specific suggestions based on the actual data

            Remember to use the formatting tool for your final output.
            """

        return f"{base_prompt}\n\n{dataframe_prompt}"

    def _handle_image_request(
        self, image_path: Union[str, Path], output_path: Optional[Union[str, Path]]
    ) -> str:
        """Handle image analysis request."""
        image_base64 = self.encode_image(image_path)
        image_response = self.analyse_image(image_base64)
        template_examples = PlotlyTemplates.get_template_prompt()

        return f"""
            Use the image_analysis_formatter tool to standardise the output and create an interactive visualisation using appropriate default chart templates as reference.
            
            1. First, format the analysis using image_analysis_formatter with these parameters:
            - image_information: {image_response}
            
            2. Based on the chart type identified in the analysis, use the appropriate template from below:
            {template_examples}
            
            3. Use the matching template as a reference for structure and styling, particularly for:
            - Layout organization
            - Title and axis label formatting
            - Template style ('plotly_white')
            - Figure update_layout parameters
            - Text positioning and formatting
            
            4. Modify the template code with:
            - The actual data from the image
            - Similar color schemes where appropriate
            - Matching chart type specifications
            - Equivalent text positioning
            
            5. Use save_plotly_visualisation to create an HTML file in the output_path folder. 
            The requested output_path folder is {output_path}. 
            If this is empty or None, then use the default path in plotly_visualisation.
            
            6. Return both:
            - The formatted analysis from step 1
            - The path to the saved visualisation
            - The modified Plotly code used
            
            Remember to maintain the professional appearance of the default templates while incorporating the specific data and styling from the image.
            Choose the most appropriate template based on the chart type identified in your analysis.
        """

    def _handle_chart_request(
        self,
        chart_type: ChartType,
        prompt: Optional[str],
        output_path: Optional[Union[str, Path]],
    ) -> str:
        """Handle specific chart type request."""
        templates = PlotlyTemplates.get_templates()
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

        return f"""
            Create a default {chart_type.value} chart visualisation using this template as reference:

            # Template Code:
            {template_code}
            
            1. Use this template code as a starting point
            2. Consider the following specific requirements in adapting the code to the user's needs:
            {prompt if prompt else "No additional specific requirements stated"}
            3. Maintain the professional styling while adjusting for your specific needs and those of the user.
            4. Use save_plotly_visualisation to save the chart as an HTML file in the output_path folder.
            The requested output_path folder is {output_path}. 
            If this is empty or None, then use the default path in plotly_visualisation.
            
            Return:
            - The modified Plotly code used
            - The path to the saved visualisation
        """
