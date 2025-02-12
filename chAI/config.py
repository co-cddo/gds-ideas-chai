import logging
import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from chAI.constants import LLMModel, AWSRegion

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there's an error in the configuration"""

    pass


# Adding helper function so we can test this with pytest
def get_env_var(var_name: str) -> Optional[str]:
    """
    Helper function to get environment variables.

    Args:
        var_name (str): Name of the environment variable to retrieve.

    Returns:
        Optional[str]: Value of the environment variable or None if not found.
    """
    return os.getenv(var_name)


def validate_aws_profile(profile: Optional[str]) -> str:
    """
    Validates AWS profile name.

    Args:
        profile (Optional[str]): AWS profile name to validate.

    Returns:
        str: Validated profile name.

    Raises:
        ConfigurationError: If profile is None or empty.
    """
    if not profile:
        raise ConfigurationError("AWS_PROFILE cannot be None or empty")
    return profile


def validate_llm_region(region: Optional[str]) -> AWSRegion:
    """
    Validates and converts region string to AWSRegion enum.
    Accepts either the region value (e.g., "us-east-1") or the enum name (e.g., "US_EAST_1").

    Args:
        region (Optional[str]): Region string to validate.

    Returns:
        AWSRegion: Validated region enum.

    Raises:
        ConfigurationError: If region is invalid or None.
    """
    if not region:
        raise ConfigurationError("LLM_REGION cannot be None or empty")

    try:
        # First try to create from value (e.g., "us-east-1")
        return AWSRegion(region)
    except ValueError:
        try:
            # If that fails, try to get by name (e.g., "US_EAST_1")
            return AWSRegion[region]
        except KeyError:
            raise ConfigurationError(
                f"Invalid region: {region}\n"
                f"Must be one of these values: {[r.value for r in AWSRegion]}\n"
                f"Or one of these names: {[r.name for r in AWSRegion]}"
            )


def validate_llm_model(model: Optional[str]) -> LLMModel:
    """
    Validates and converts model string to LLMModel enum.

    Args:
        model (Optional[str]): Model string to validate.

    Returns:
        LLMModel: Validated model enum.

    Raises:
        ConfigurationError: If model is invalid or None.
    """
    if not model:
        raise ConfigurationError("LLM_MODEL cannot be None or empty")

    try:
        # First try to create from value in env
        return LLMModel(model)
    except ValueError:
        try:
            # If that fails, try to get by name
            return LLMModel[model]
        except KeyError:
            raise ConfigurationError(
                f"Invalid model: {model}. Must be one of {[m.value for m in LLMModel]} "
                f"or one of {[m.name for m in LLMModel]}"
            )


@dataclass
class Config:
    """
    Configuration class for the application.

    Attributes:
        AWS_PROFILE (str): AWS profile name
        LLM_REGION (AWSRegion): AWS region for the LLM
        LLM_MODEL (LLMModel): LLM model identifier
    """

    AWS_PROFILE: str = field(
        default_factory=lambda: validate_aws_profile(get_env_var("AWS_PROFILE"))
    )
    LLM_REGION: AWSRegion = field(
        default_factory=lambda: validate_llm_region(get_env_var("LLM_REGION"))
    )
    LLM_MODEL: LLMModel = field(
        default_factory=lambda: validate_llm_model(get_env_var("LLM_MODEL"))
    )

    def __post_init__(self):
        """
        Validates the configuration after initialisation.

        Raises:
            ConfigurationError: If any required fields are missing or invalid.
        """
        try:
            self.validate()
            logger.info("Loaded config for agent successfully.")
        except Exception as e:
            logger.error(f"Configuration error: {str(e)}")
            raise

    def validate(self) -> None:
        """
        Validates all configuration values.

        Raises:
            ConfigurationError: If any validation fails.
        """
        missing_fields = []

        if not isinstance(self.AWS_PROFILE, str):
            missing_fields.append("AWS_PROFILE")

        if not isinstance(self.LLM_REGION, AWSRegion):
            missing_fields.append("LLM_REGION")

        if not isinstance(self.LLM_MODEL, LLMModel):
            missing_fields.append("LLM_MODEL")

        if missing_fields:
            raise ConfigurationError(
                f"Invalid configuration for fields: {missing_fields}"
            )

        logger.info("Loaded config for agent successfully.")
