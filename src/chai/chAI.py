# Import base requirements for data handling and AWS
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

# Import agent dependencies
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_json_chat_agent,
)
from pydantic import BaseModel, Field

from .bedrock import BedrockHandler

# Import custom classes and tools
from .config import Config
from .constants import ChartType
from .requests import (
    DataFrameHandler,
    ImageHandler,
    TypeHandler,
)
from .tools import (
    create_analysis_formatter_tool,
    create_formatting_tool,
    create_save_plotly_tool,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*API key must be provided when using hosted LangSmith API.*",
)

logger = logging.getLogger(__name__)


class ChAIError(Exception):
    """Base exception for chAI errors"""

    pass


class ChAITeapot(BaseModel):
    """
    Pydantic model for the teapot component of ChAI responses.
    Different fields will be populated based on the request type.
    """

    # Raw response text (private)
    raw_text: str = Field(alias="raw_text")

    # Fields for all visualisation suggestions from dataset_requests
    suggestions: Optional[str] = None

    # Fields for image_request and chart_request
    analysis: Optional[str] = None
    code: Optional[str] = None
    path: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    def __str__(self) -> str:
        """Return the raw text when the object is printed"""
        return self.raw_text


class chAI:
    def __init__(
        self,
        aws_profile: Optional[str] = None,
        llm_region: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initialises the chAI class with required configurations and tools.

        Args:
            aws_profile: Optional AWS profile name. If provided, overrides environment variable.
            llm_region: Optional LLM region. If provided, overrides environment variable.
            llm_model: Optional LLM model. If provided, overrides environment variable.

        Notes:
            - AWS profile is loaded from environment variables via Config class
            - Sets up Bedrock handler and runtime
            - Loads LLM model and prompt
            - Sets up visualization tools and templates
            - Creates agent executor
        """
        logger.info("chAI Start")

        self.config = Config(
            aws_profile=aws_profile, llm_region=llm_region, llm_model=llm_model
        )
        self.bedrock = BedrockHandler(self.config)
        self.bedrock_runtime = self.bedrock.runtime_client
        self.llm = self.bedrock.llm
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

    def set_agent_executor(self, verbose=False, handle_parse=True) -> AgentExecutor:
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

    def steep(
        self,
        data: Optional[pd.DataFrame] = None,
        prompt: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        chart_type: Optional[ChartType] = None,
        **kwargs: Any,
    ) -> ChAITeapot:
        """
        Processes user requests based on input type and generates appropriate visualisations.

        Args:
            data (Optional[pd.DataFrame]): Input data for analysis.
            prompt (Optional[str]): User instructions for visualisation.
            image_path (Optional[Union[str, Path]]): Path to image for analysis.
            chart_type (Optional[ChartType]): Specific chart type from ChartType enum.
            **kwargs (Any): Additional keyword arguments for the LLM.

        Returns:
            ChAITeapot: A response object that:
                - Can be printed as a string to show the full response
                - Has structured components directly accessible:
                    - For DataFrame and visualisation requests: .suggestions
                    - For image requests: .analysis, .code, .path
                    - For chart requests: .code, .path

        Raises:
            ChAIError: If there's an error processing the request.
            ValueError: If no valid input is provided.

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
            request_type = "dataframe"

        elif isinstance(image_path, str):
            logger.info("Detected image location input. Preparing to review...")
            final_prompt = self.image_handler.image_request(
                image_path=image_path,
                bedrock_runtime=self.bedrock_runtime,
                model_id=self.config.LLM_MODEL.value,
                custom_prompt=prompt,
            )
            request_type = "image"

        elif chart_type:
            logger.info(f"Processing chart type request: {chart_type}")
            final_prompt = self.type_handler.chart_request(
                chart_type=chart_type, custom_prompt=prompt
            )
            request_type = "chart"

        else:
            raise ValueError("No valid input provided")

        # Send to the agent executor
        try:
            logger.info("Sending prompt and data to agent executor...")
            response = self.agent_executor.invoke({"input": final_prompt})
            raw_output = response["output"]

            # Process the output based on the request type
            teapot_data = self._process_output(raw_output, request_type)

            # Create and return the response object
            raw_text = (
                json.dumps(raw_output) if isinstance(raw_output, dict) else raw_output
            )
            return ChAITeapot(raw_text=raw_text, **teapot_data)
        except Exception as e:
            logger.error(f"Error in steep: {str(e)}")
            raise ChAIError(f"Failed to process request: {e}")

    def _process_output(self, raw_output: Any, request_type: str) -> Dict[str, Any]:
        """
        Process the raw output based on the request type and extract structured components.

        This method parses the agent's response according to the request type and
        extracts relevant components into a dictionary structure that matches the ChAITeapot model.

        Args:
            raw_output (Any): Raw output from the agent, can be string or dictionary
            request_type (str): Type of request ("dataframe", "image", or "chart")

        Returns:
            Dict[str, Any]: Dictionary with extracted components based on request type:
                - For DataFrame requests: {"suggestions": str}
                - For image requests: {"analysis": str, "code": str, "path": str}
                - For chart requests: {"code": str, "path": str}
        """
        result = {}

        if request_type == "dataframe":
            # For DataFrame requests, use raw output as suggestions
            result["suggestions"] = raw_output

        elif request_type == "image":
            # For image requests, the output should be a JSON dictionary
            if isinstance(raw_output, dict):
                # If it's already a dictionary, use it directly
                if "analysis" in raw_output:
                    result["analysis"] = raw_output["analysis"]
                if "code" in raw_output:
                    result["code"] = raw_output["code"]
                if "path" in raw_output:
                    result["path"] = raw_output["path"]
            else:
                # If not a dictionary, try parsing as JSON
                try:
                    json_data = json.loads(raw_output)
                    if "analysis" in json_data:
                        result["analysis"] = json_data["analysis"]
                    if "code" in json_data:
                        result["code"] = json_data["code"]
                    if "path" in json_data:
                        result["path"] = json_data["path"]
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Image response was not valid JSON")
                    # If not JSON, use raw output as analysis
                    result["analysis"] = raw_output

        elif request_type == "chart":
            # For chart requests, the output should be a JSON dictionary
            if isinstance(raw_output, dict):
                # Map JSON keys directly to result
                if "code" in raw_output:
                    result["code"] = raw_output["code"]
                if "path" in raw_output:
                    result["path"] = raw_output["path"]
            else:
                try:
                    json_data = json.loads(raw_output)
                    # Map JSON keys directly to result
                    if "code" in json_data:
                        result["code"] = json_data["code"]
                    if "path" in json_data:
                        result["path"] = json_data["path"]
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Chart response was not valid JSON")
                    # If not JSON, use raw output as code
                    result["code"] = raw_output

        return result
