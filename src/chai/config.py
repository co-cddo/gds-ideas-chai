import logging
import os
from dataclasses import dataclass
from typing import Optional

from .constants import AWSRegion, LLMModel

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there's an error in the configuration"""

    pass


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

    AWS_PROFILE: str
    LLM_REGION: AWSRegion
    LLM_MODEL: LLMModel

    def __init__(
        self,
        aws_profile: Optional[str] = None,
        llm_region: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize Config with optional direct values.

        Args:
            aws_profile: Optional AWS profile name. If provided, overrides environment variable.
            llm_region: Optional LLM region. If provided, overrides environment variable.
            llm_model: Optional LLM model. If provided, overrides environment variable.
        """
        # Create some placeholders here so depending on how user has configured their system, tells where variables are coming from
        user_specified = []
        env_specified = []

        # Process aws_profile
        if aws_profile:
            self.AWS_PROFILE = validate_aws_profile(aws_profile)
            user_specified.append(f"AWS_PROFILE: {self.AWS_PROFILE}")
        else:
            self.AWS_PROFILE = validate_aws_profile(os.getenv("AWS_PROFILE"))
            env_specified.append(f"AWS_PROFILE: {self.AWS_PROFILE}")

        # Process llm_region
        if llm_region:
            self.LLM_REGION = validate_llm_region(llm_region)
            user_specified.append(f"LLM_REGION: {self.LLM_REGION.value}")
        else:
            self.LLM_REGION = validate_llm_region(os.getenv("LLM_REGION"))
            env_specified.append(f"LLM_REGION: {self.LLM_REGION.value}")

        # Process llm_model
        if llm_model:
            self.LLM_MODEL = validate_llm_model(llm_model)
            user_specified.append(f"LLM_MODEL: {self.LLM_MODEL.name}")
        else:
            self.LLM_MODEL = validate_llm_model(os.getenv("LLM_MODEL"))
            env_specified.append(f"LLM_MODEL: {self.LLM_MODEL.name}")

        # Log the sources of configuration values
        if user_specified:
            logger.info(f"Using user-specified variables: {', '.join(user_specified)}")
        if env_specified:
            logger.info(f"Using .env variables: {', '.join(env_specified)}")

        # Validate the configuration
        self.validate()
        logger.info("Loaded config for agent successfully.")

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
