import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrock

from .config import Config, ConfigurationError
from .constants import AWSRegion, LLMModel

logger = logging.getLogger(__name__)


class BedrockHandlerError(Exception):
    """Base exception for BedrockHandler errors"""

    pass


class BedrockHandler:
    def __init__(self, config: Config):
        """
        Initialise BedrockHandler with configuration.

        Args:
            config (Config): Configuration object containing AWS settings

        Raises:
            ConfigurationError: If configuration is invalid
        """
        logger.info("Initialising BedrockHandler")
        self.region: AWSRegion = config.LLM_REGION
        self.model_id: LLMModel = config.LLM_MODEL
        self.profile: str = config.AWS_PROFILE
        self._validate_config()

        # Create a single session to be reused across all of chAI
        self.session = boto3.Session(profile_name=self.profile)

        # These are necessary for instance-level caching, ensures subsequent calls are returned directly.
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

    @property  # Cache this as part of the class to improve performance on load
    def runtime_client(self) -> boto3.client:
        """
        Get or create Bedrock runtime client.

        Returns:
            boto3.client: Bedrock runtime client

        Raises:
            BedrockHandlerError: If client creation fails
        """
        if self._runtime is None:
            try:
                self._runtime = self.set_runtime()
            except ClientError as e:
                logger.error(f"Failed to create Bedrock runtime client: {e}")
                raise BedrockHandlerError(f"Bedrock client creation failed: {e}")
        return self._runtime

    @property  # Cache this as part of the class to improve performance on load
    def llm(self) -> ChatBedrock:
        """
        Get or create ChatBedrock LLM instance.

        Returns:
            ChatBedrock: LLM instance

        Raises:
            BedrockHandlerError: If LLM creation fails
        """
        if self._llm is None:
            try:
                self._llm = self.get_llm()
            except Exception as e:
                logger.error(f"Failed to create ChatBedrock LLM instance: {e}")
                raise BedrockHandlerError(f"LLM creation failed: {e}")
        return self._llm

    def get_llm(self) -> ChatBedrock:
        """
        Create a new ChatBedrock LLM instance.

        Returns:
            ChatBedrock: New LLM instance

        Raises:
            BedrockHandlerError: If creation fails
        """
        logger.info("Creating ChatBedrock LLM instance")
        runtime = self.runtime_client
        try:
            llm = ChatBedrock(
                model_id=self.model_id.value,
                client=runtime,
                region_name=self.region.value,
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
