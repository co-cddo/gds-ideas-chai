# Import base requirements for data handling and AWS
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4
import boto3
from dotenv import load_dotenv

# Import agent dependencies
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    tool,
    create_json_chat_agent,
)

# Import custom classes and tools
from chAI.config import Config
from chAI.bedrock import BedrockHandler
from tools.visualisation_formatter import create_formatting_tool
from tools.image_analysis_formatter import create_analysis_formatter_tool
from tools.save_plotly import create_save_plotly_tool
from tools.default_charts import PlotlyTemplates

logger = logging.getLogger()

import pandas as pd
import base64


class chAI:
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialise chAI with required configurations and tools.
        """
        logger.info("chAI Start")
        self.config = Config()
        self.bedrock = BedrockHandler(self.config)
        self.bedrock_runtime = self.bedrock.set_runtime()
        self.llm = self.bedrock.get_llm()
        self.prompt = hub.pull("hwchase17/react-chat-json")

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
        logger.info("Setting up chAI executor")
        try:
            agent = create_json_chat_agent(self.llm, self.tools, self.prompt)
            logger.debug("Structured chat agent created")

            executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=verbose,
                handle_parsing_errors=handle_parse,
            )
            logger.info("Agent executor successfully set up")
            return executor
        except Exception as e:
            logger.error(f"Error setting up agent executor: {str(e)}")
            raise

    def parse_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Extract useful information from a DataFrame and return it as a JSON-like dictionary suitable for an LLM to review.

        Parameters:
        df: DataFrame to analyse.

        Returns:
        dict: A structured dictionary containing DataFrame details.
        """
        logger.info("Parsing DataFrame into structured JSON dictionary")
        try:
            try:
                summary = df.describe(include="all").to_dict()
            except ValueError as e:
                logger.warning(f"Could not generate full summary statistics: {e}")
                summary = {}

            data_info = {
                "columns": [
                    {"name": col, "dtype": str(dtype)}
                    for col, dtype in df.dtypes.items()
                ],
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "summary": summary,
                "sample_data": df.head(10).to_dict(orient="records"),
            }
            logger.debug(f"DataFrame parsed: {data_info}")
            return data_info
        except Exception as e:
            logger.error(f"Error parsing DataFrame: {str(e)}")
            raise

    def encode_image(self, image_path) -> str:
        """
        Encode one or more images into Base64 ready for Claude's multi-modal API call. Avoids agent since agent will truncate base64 image payload.

        Parameters:
            image_paths str: Path to the chart image

        Returns:
            image_contents str: base64 encoded image string data

        Raises:
            Exception: If there's an error processing the request or calling Bedrock
        """
        logger.info("Encoding image to base64")
        try:
            # Create image content string
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            return encoded_image

        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise

    def analyse_image(self, base64_data, custom_prompt=None):
        """
        Necessary because trying to pass it into an agent tool fails as agent will truncate the base64 image
        data.

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
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
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
            logger.error(f"Full error in analyse_image: {str(e)}")
            return f"Error analysing image: {str(e)}"

    def handle_request(
        self,
        data=None,
        prompt=None,
        image_path=None,
        chart_type=None,
        output_path=None,
        **kwargs,
    ):
        """
        Handle user input by deciding how to process it, based on the type of data provided.

        Parameters:
        data: Input data for the agent. Could be a DataFrame or other types.
        prompt: String prompt to guide the agent.
        image_paths: List of image paths to convert and pass for analysis
        output_path: String to set the output path for any plotly-based visualisations in html files. If none, will store in home directory.

        Returns:
        Response generated by the agent executor.
        """

        base_prompt = f"""
            User Prompt:
            {prompt}
            """

        if isinstance(data, pd.DataFrame):
            logger.info("Detected DataFrame input. Preparing to analyse...")

            # Limit the DataFrame to the top 100 rows to reduce LLM costs
            max_rows = 100
            if len(data) > max_rows:
                logger.info(
                    f"DataFrame has more than {max_rows} rows. Trimming for processing."
                )
                data = data.head(max_rows)

            # Convert DataFrame to JSON string ready for LLM review
            dataframe_json = data.to_json(orient="split")

            # Construct the combined prompt - TODO: maybe instead of creating a separate prompt, we add the bits
            # to the prompt depending on what is being analysed as we do with citations.

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

            final_prompt = f"""
                {base_prompt}

                {dataframe_prompt}
                """

        if isinstance(image_path, str):
            logger.info("Detected image location input. Preparing to review...")

            # Get encoded image data
            image_base64 = self.encode_image(image_path)
            image_response = self.analyse_image(image_base64)

            logger.info(f"Claude markdown response: {image_response}")

            # Get the template examples
            template_examples = PlotlyTemplates.get_template_prompt()

            final_prompt = f"""
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

        if chart_type:
            templates = PlotlyTemplates.get_templates()
            chart_type_mapping = {
                "bar": "bar_chart",
                "histogram": "histogram",
                "scatter": "scatter_plot",
                "line": "line_chart",
            }
            chart_type_lower = chart_type.lower()
            if chart_type_lower not in chart_type_mapping:
                logger.warning(
                    f"Unsupported chart type: {chart_type}. Defaulting to bar chart."
                )
                template_key = "bar_chart"
            else:
                template_key = chart_type_mapping[chart_type_lower]

            template_code = templates[template_key]

            final_prompt = f"""
            Create a default {chart_type if chart_type else 'bar'} chart visualisation using this template as reference:

            # Template Code:
            {template_code}
            
            1. Use this template code as a starting point
            2. Consider the following specific requirements in adapting the code to the user's needs:
            {prompt if prompt else "No additional specific requirements stated"}
            3. Maintain the professional styling while adjusting for your specific needs and those of the user.
            4. Use save_plotly_visualisation to save the chart as an HTML file in the output_path folder.
            The requested output_path folder is {output_path}. 
            If this is empty or None, then use the default path in plotly_visualisation.
            
            4. Return:
            - The modified Plotly code used
            - The path to the saved visualisation
            """

        # Send to the agent executor
        try:
            logger.info("Sending prompt and data to agent executor...")
            response = self.agent_executor.invoke({"input": final_prompt})
            print(response)
            return response
        except Exception as e:
            logger.error(f"Error in agent executor: {str(e)}")
            return f"Error processing visualisation suggestions: {str(e)}"
