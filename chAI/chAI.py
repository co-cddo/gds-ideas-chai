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
import pandas as pd
import base64

# Import agent dependencies
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_json_chat_agent,
)

# Import custom classes and tools
from .config import Config
from .bedrock import BedrockHandler
from .tools import (
    create_formatting_tool,
    create_analysis_formatter_tool,
    create_save_plotly_tool,
)
from .constants import (
    LLMModel,
    AWSRegion,
)
from .requests import (
    DataFrameHandler,
    ImageHandler,
    TypeHandler,
)

logger = logging.getLogger()


class ChAIError(Exception):
    """Base exception for chAI errors"""

    pass


class chAI:
    def __init__(self):
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
        self.image_handler = ImageHandler()
        self.type_handler = TypeHandler()

        self.tools = [
            create_formatting_tool(),
            create_analysis_formatter_tool(),
            create_save_plotly_tool(),
        ]
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

    def handle_request(
        self,
        data: Optional[pd.DataFrame] = None,
        prompt: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        chart_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Processes user requests based on input type and generates appropriate visualisations.

        Args:
            data (Optional[pd.DataFrame]): Input data for analysis.
            prompt (Optional[str]): User instructions for visualisation.
            image_path (Optional[Union[str, Path]]): Path to image for analysis.
            chart_type (Optional[ChartType]): Specific chart type from ChartType enum.
            **kwargs (Any): Additional keyword arguments for the LLM.

        Returns:
            Dict[str, Any]: Agent response containing 1 or more of:
                - Analysis results
                - Generated visualisations plotly code
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
            final_prompt = self.image_handler.image_request(
                image_path=image_path,
                bedrock_runtime=self.bedrock_runtime,
                model_id=self.config.LLM_MODEL.value,
                custom_prompt=prompt,
            )

        elif chart_type:
            logger.info(f"Processing chart type request: {chart_type}")
            final_prompt = self.type_handler.chart_request(
                chart_type=chart_type, custom_prompt=prompt
            )

        else:
            raise ValueError("No valid input provided")

        # Send to the agent executor
        try:
            logger.info("Sending prompt and data to agent executor...")
            response = self.agent_executor.invoke({"input": final_prompt})
            return response["output"]
        except Exception as e:
            logger.error(f"Error in handle_request: {str(e)}")
            raise ChAIError(f"Failed to process request: {e}")
