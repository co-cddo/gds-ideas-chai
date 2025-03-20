from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from chai.bedrock import BedrockHandler, BedrockHandlerError
from chai.config import Config, ConfigurationError
from chai.constants import AWSRegion, LLMModel


@pytest.fixture
def mock_config():
    """Create a mock configuration"""
    config = Mock(spec=Config)
    config.LLM_REGION = AWSRegion.US_EAST_1
    config.LLM_MODEL = LLMModel.CLAUDE_SONNET_3_5
    config.AWS_PROFILE = "test-profile"
    return config


@pytest.fixture
def bedrock_handler(mock_config):
    """Create a BedrockHandler instance with mock config"""
    with patch("boto3.Session") as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        handler = BedrockHandler(mock_config)
        handler.session = mock_session
        return handler


def test_initialization_successful(mock_config):
    """Test successful initialization of BedrockHandler"""
    with patch("boto3.Session") as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        handler = BedrockHandler(mock_config)

        assert handler.region == AWSRegion.US_EAST_1
        assert handler.model_id == LLMModel.CLAUDE_SONNET_3_5
        assert handler.profile == "test-profile"
        assert handler._runtime is None
        assert handler._llm is None
        mock_session_class.assert_called_once_with(profile_name="test-profile")


@pytest.mark.parametrize(
    "invalid_config",
    [
        # Invalid model type
        {
            "LLM_MODEL": "invalid-model",
            "LLM_REGION": AWSRegion.US_EAST_1,
            "AWS_PROFILE": "test",
        },
        # Invalid region type
        {
            "LLM_MODEL": LLMModel.CLAUDE_SONNET_3_5,
            "LLM_REGION": "invalid-region",
            "AWS_PROFILE": "test",
        },
        # Invalid profile type
        {
            "LLM_MODEL": LLMModel.CLAUDE_SONNET_3_5,
            "LLM_REGION": AWSRegion.US_EAST_1,
            "AWS_PROFILE": 123,
        },
    ],
)
def test_initialization_with_invalid_config(invalid_config):
    """Test initialization with invalid configuration"""
    mock_config = Mock(spec=Config)
    for key, value in invalid_config.items():
        setattr(mock_config, key, value)

    with pytest.raises(ConfigurationError):
        BedrockHandler(mock_config)


def test_runtime_client_creation_successful(bedrock_handler):
    """Test successful creation of runtime client"""
    mock_client = Mock()
    mock_session = Mock()
    mock_session.client.return_value = mock_client

    with patch("boto3.Session") as mock_session_class:
        mock_session_class.return_value = mock_session

        client = bedrock_handler.runtime_client

        assert client == mock_client
        mock_session_class.assert_called_once_with(profile_name=bedrock_handler.profile)
        mock_session.client.assert_called_once_with(
            service_name="bedrock-runtime", region_name=bedrock_handler.region.value
        )


def test_runtime_client_creation_failure(bedrock_handler):
    """Test handling of runtime client creation failure"""
    with patch("chai.bedrock.boto3.Session") as mock_session:
        mock_session.side_effect = ClientError(
            error_response={"Error": {"Code": "InvalidProfile"}},
            operation_name="CreateClient",
        )

        with pytest.raises(BedrockHandlerError) as exc_info:
            _ = bedrock_handler.runtime_client

        assert "Bedrock client creation failed" in str(exc_info.value)


def test_runtime_client_caching(bedrock_handler):
    """Test that runtime client is cached"""
    mock_client = Mock()
    mock_session = Mock()
    mock_session.client.return_value = mock_client

    with patch("boto3.Session") as mock_session_class:
        mock_session_class.return_value = mock_session

        # First access creates the client
        client1 = bedrock_handler.runtime_client
        # Second access should return cached client
        client2 = bedrock_handler.runtime_client

        assert client1 == client2
        mock_session_class.assert_called_once()


def test_llm_creation_successful(bedrock_handler):
    """Test successful creation of LLM instance"""
    mock_runtime = Mock()
    bedrock_handler._runtime = mock_runtime
    mock_llm = Mock()

    with patch("chai.bedrock.ChatBedrock", return_value=mock_llm) as mock_chat_bedrock:
        llm = bedrock_handler.get_llm()

        assert llm == mock_llm
        mock_chat_bedrock.assert_called_once_with(
            model_id=bedrock_handler.model_id.value,
            client=mock_runtime,
            region_name=bedrock_handler.region.value,
        )


def test_runtime_client_creation_error(bedrock_handler):
    """Test handling of runtime client creation failure"""
    with patch("chai.bedrock.boto3.Session") as mock_session:
        mock_session.side_effect = ClientError(
            error_response={"Error": {"Code": "InvalidProfile"}},
            operation_name="CreateClient",
        )

        with pytest.raises(BedrockHandlerError) as exc_info:
            _ = bedrock_handler.runtime_client

        assert "Bedrock client creation failed" in str(exc_info.value)


def test_llm_caching(bedrock_handler):
    """Test that LLM instance is cached"""
    mock_llm = Mock()
    mock_client = Mock()

    # We need to patch both the ChatBedrock and boto3.Session
    with (
        patch("chai.bedrock.ChatBedrock", return_value=mock_llm) as mock_chat_bedrock,
        patch("boto3.Session") as mock_session_class,
    ):
        # Set up the mock session
        mock_session = Mock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        # Set the runtime client directly to avoid boto3 calls
        bedrock_handler._runtime = mock_client

        # First access creates the LLM
        llm1 = bedrock_handler.llm  # Use the property, not get_llm()
        # Second access should return cached LLM
        llm2 = bedrock_handler.llm

        # Verify the LLM was only created once
        assert llm1 == llm2
        mock_chat_bedrock.assert_called_once()


# Try different errors when setting runtime
@pytest.mark.parametrize(
    "error,expected_message",
    [
        (
            ClientError({"Error": {"Code": "InvalidProfile"}}, "CreateClient"),
            "Bedrock client creation failed",
        ),
        (Exception("Network error"), "Bedrock client creation failed"),
    ],
)
def test_set_runtime_errors(bedrock_handler, error, expected_message):
    """Test different error scenarios in set_runtime"""
    with patch("boto3.Session") as mock_session_class:
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = error
        mock_session_class.return_value = mock_session_instance

        with pytest.raises(BedrockHandlerError) as exc_info:
            bedrock_handler.set_runtime()

        assert expected_message in str(exc_info.value)
