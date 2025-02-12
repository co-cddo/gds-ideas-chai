import pytest
from unittest.mock import Mock, patch
import boto3
import botocore
from langchain_aws import ChatBedrock
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chAI.bedrock import BedrockHandler


@pytest.fixture
def mock_config():
    """Create a mock config object"""
    config = Mock()
    config.LLM_REGION = "us-west-2"
    config.LLM_MODEL = "anthropic.claude-v2"
    config.AWS_PROFILE = "test-profile"
    return config


@pytest.fixture
def bedrock_handler(mock_config):
    """Create a BedrockHandler instance with mock config"""
    return BedrockHandler(mock_config)


def test_initialisation(mock_config):
    """Test successful initialisation of BedrockHandler"""
    handler = BedrockHandler(mock_config)
    assert handler.region == "us-west-2"
    assert handler.model_id == "anthropic.claude-v2"
    assert handler.profile == "test-profile"


def test_get_llm_success(bedrock_handler):
    """Test successful creation of ChatBedrock instance"""
    with patch("chAI.bedrock.ChatBedrock") as mock_chat_bedrock:
        # Call the method
        bedrock_handler.get_llm()

        # Verify ChatBedrock was called with correct parameters
        mock_chat_bedrock.assert_called_once_with(
            model_id=bedrock_handler.model_id, region_name=bedrock_handler.region
        )


def test_get_llm_failure(bedrock_handler):
    """Test handling of ChatBedrock creation failure"""
    with patch("chAI.bedrock.ChatBedrock") as mock_chat_bedrock:
        # Configure the mock to raise an exception
        mock_chat_bedrock.side_effect = Exception("Connection error")

        # Verify that the exception is raised
        with pytest.raises(Exception) as exc_info:
            bedrock_handler.get_llm()

        assert "Connection error" in str(exc_info.value)


def test_set_runtime(bedrock_handler):
    """Test successful creation of bedrock-runtime client"""
    mock_client = Mock()
    mock_session = Mock()
    mock_session.client.return_value = mock_client

    with patch("boto3.Session") as mock_session_class:
        mock_session_class.return_value = mock_session

        result = bedrock_handler.set_runtime()

        # Verify the session was created with correct parameters
        mock_session_class.assert_called_once_with(profile_name=bedrock_handler.profile)

        # Verify the client was created with correct parameters
        mock_session.client.assert_called_once_with(
            service_name="bedrock-runtime", region_name=bedrock_handler.region
        )


def test_set_runtime_with_invalid_profile(bedrock_handler):
    """Test handling of invalid AWS profile"""
    with patch("boto3.Session") as mock_session_class:
        # Simulate boto3 error for invalid profile
        mock_session_class.side_effect = botocore.exceptions.ProfileNotFound(
            profile="invalid-profile"
        )

        with pytest.raises(botocore.exceptions.ProfileNotFound):
            bedrock_handler.set_runtime()


@pytest.mark.parametrize(
    "config_params",
    [
        {"LLM_REGION": None, "LLM_MODEL": "model", "AWS_PROFILE": "profile"},
        {"LLM_REGION": "", "LLM_MODEL": "model", "AWS_PROFILE": "profile"},
    ],
)
def test_initialisation_with_invalid_params(config_params):
    """Test initialisation with invalid parameters"""
    mock_config = Mock()
    for key, value in config_params.items():
        setattr(mock_config, key, value)

    # Just test that initialsation works
    handler = BedrockHandler(mock_config)
    assert handler is not None


def test_get_llm_with_invalid_region(bedrock_handler):
    """Test get_llm with invalid region"""
    with patch("chAI.bedrock.ChatBedrock") as mock_chat_bedrock:
        mock_chat_bedrock.side_effect = botocore.exceptions.ClientError(
            error_response={"Error": {"Code": "InvalidRegion"}},
            operation_name="CreateModel",
        )

        with pytest.raises(botocore.exceptions.ClientError):
            bedrock_handler.get_llm()


def test_integration_get_llm_and_set_runtime(bedrock_handler):
    """Test integration between get_llm and set_runtime"""
    mock_client = Mock()
    mock_session = Mock()
    mock_session.client.return_value = mock_client

    with patch("boto3.Session") as mock_session_class, patch(
        "chAI.bedrock.ChatBedrock"
    ) as mock_chat_bedrock:

        mock_session_class.return_value = mock_session

        # Get both LLM and runtime client
        bedrock_handler.get_llm()
        runtime = bedrock_handler.set_runtime()

        # Verify the calls were made correctly
        mock_chat_bedrock.assert_called_once_with(
            model_id=bedrock_handler.model_id, region_name=bedrock_handler.region
        )
        assert runtime == mock_client
