import logging
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class Config:
    AWS_PROFILE: str = os.getenv("AWS_PROFILE")
    """AWS profile to use"""
    LLM_REGION: str = os.getenv("LLM_REGION")
    """the region of the llm"""
    LLM_MODEL: str = os.getenv("LLM_MODEL")
    """the name of the LLM used in the agent"""

    def __post_init__(self):
        # Check is any of the values are None.
        missing_fields = [
            field_name
            for field_name, field_value in self.__dict__.items()
            if field_value is None
        ]
        # If they are raise a value error
        if missing_fields:
            logger.error(
                f"There were missing fields {missing_fields} when trying to initialise log for agent."
            )
            raise ValueError(
                f"Field{'s' if len(missing_fields) > 1 else ''} '{missing_fields}' cannot be None"
            )

        logger.info("Loaded config for agent successfully.")


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
