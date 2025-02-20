from typing import Optional
import logging
import boto3
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrock
from .constants import AWSRegion, LLMModel
from .config import Config, ConfigurationError

logger = logging.getLogger(__name__)


class BedrockHandlerError(Exception):
    """Base exception for BedrockHandler errors"""

    pass


class BedrockHandler:
    def __init__(self, config: Config):
        """
        Initialize BedrockHandler with configuration.

        Args:
            config (Config): Configuration object containing AWS settings

        Raises:
            ConfigurationError: If configuration is invalid
        """
        logger.info("Initializing BedrockHandler")
        self.region: AWSRegion = config.LLM_REGION
        self.model_id: LLMModel = config.LLM_MODEL
        self.profile: str = config.AWS_PROFILE
        self._validate_config()
        self._runtime: Optional[boto3.client] = None
        self._llm: Optional[ChatBedrock] = None

        logger.debug(f"LLM Region: {self.region.value}")
        logger.debug(f"LLM Model ID: {self.model_id.value}")

    def _validate_config(self) -> None:
        """
        Validate configuration values.

        Raises:
            ConfigurationError: If any configuration value is invalid
        """
        if not isinstance(self.model_id, LLMModel):
            raise ConfigurationError(f"Invalid model ID: {self.model_id}")
        if not isinstance(self.region, AWSRegion):
            raise ConfigurationError(f"Invalid region: {self.region}")
        if not isinstance(self.profile, str):
            raise ConfigurationError(f"Invalid profile: {self.profile}")

    def get_llm(self) -> ChatBedrock:
        """
        Create a new ChatBedrock LLM instance.

        Returns:
            ChatBedrock: New LLM instance

        Raises:
            BedrockHandlerError: If creation fails
        """
        logger.info("Creating ChatBedrock LLM instance")
        try:
            llm = ChatBedrock(
                model_id=self.model_id.value, region_name=self.region.value
            )
            logger.info("Successfully created ChatBedrock LLM instance")
            return llm
        except Exception as e:
            logger.error(f"Error creating ChatBedrock LLM instance: {str(e)}")
            raise BedrockHandlerError(f"LLM creation failed: {e}")

    def set_runtime(self) -> boto3.client:
        """
        Create a new Bedrock runtime client.

        Returns:
            boto3.client: New Bedrock runtime client

        Raises:
            BedrockHandlerError: If creation fails
        """
        try:
            return boto3.Session(profile_name=self.profile).client(
                service_name="bedrock-runtime", region_name=self.region.value
            )
        except Exception as e:
            logger.error(f"Error creating Bedrock runtime client: {str(e)}")
            raise BedrockHandlerError(f"Bedrock client creation failed: {e}")
