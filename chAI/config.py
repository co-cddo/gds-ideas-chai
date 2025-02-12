import logging
import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# Adding helper function so we can test this with pytest
def get_env_var(var_name: str) -> str:
    """Helper function to get environment variables"""
    return os.getenv(var_name)


@dataclass
class Config:
    AWS_PROFILE: str = field(default_factory=lambda: get_env_var("AWS_PROFILE"))
    """AWS profile to use"""
    LLM_REGION: str = field(default_factory=lambda: get_env_var("LLM_REGION"))
    """the region of the llm"""
    LLM_MODEL: str = field(default_factory=lambda: get_env_var("LLM_MODEL"))
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
