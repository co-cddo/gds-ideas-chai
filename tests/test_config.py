import os

import pytest

from chai.config import (
    Config,
    ConfigurationError,
    validate_aws_profile,
    validate_llm_model,
    validate_llm_region,
)
from chai.constants import AWSRegion, LLMModel


@pytest.fixture(autouse=True)
def mock_env():
    """Mock entire environment"""
    original_environ = dict(os.environ)
    os.environ.clear()
    yield
    os.environ.clear()
    os.environ.update(original_environ)


# Test validation functions
def test_validate_aws_profile():
    """Test AWS profile validation"""
    assert validate_aws_profile("valid-profile") == "valid-profile"

    with pytest.raises(ConfigurationError, match="AWS_PROFILE cannot be None or empty"):
        validate_aws_profile(None)

    with pytest.raises(ConfigurationError, match="AWS_PROFILE cannot be None or empty"):
        validate_aws_profile("")


def test_validate_llm_region():
    """Test LLM region validation"""
    # Test valid regions
    assert validate_llm_region("us-east-1") == AWSRegion.US_EAST_1
    assert validate_llm_region("us-west-2") == AWSRegion.US_WEST_2

    # Test invalid regions
    with pytest.raises(ConfigurationError, match="LLM_REGION cannot be None or empty"):
        validate_llm_region(None)

    with pytest.raises(ConfigurationError, match="Invalid region"):
        validate_llm_region("invalid-region")


def test_validate_llm_model():
    """Test LLM model validation"""
    # Test valid models
    assert (
        validate_llm_model("anthropic.claude-3-5-sonnet-20240620-v1:0")
        == LLMModel.CLAUDE_SONNET_3_5
    )

    # Test invalid models
    with pytest.raises(ConfigurationError, match="LLM_MODEL cannot be None or empty"):
        validate_llm_model(None)

    with pytest.raises(ConfigurationError, match="Invalid model"):
        validate_llm_model("invalid-model")


# Test Config class
def test_config_successful_initialization(mock_env):
    """Test successful initialization with all environment variables set"""
    os.environ.update(
        {
            "AWS_PROFILE": "test-profile",
            "LLM_REGION": "us-west-2",
            "LLM_MODEL": "anthropic.claude-v3:5",
        }
    )

    config = Config()
    assert config.AWS_PROFILE == "test-profile"
    assert config.LLM_REGION == AWSRegion.US_WEST_2
    assert config.LLM_MODEL == LLMModel.CLAUDE_SONNET_3_5


def test_config_direct_assignment():
    """Test successful initialization with direct assignment"""
    config = Config(
        aws_profile="direct-profile",
        llm_region="eu-west-1",
        llm_model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    assert config.AWS_PROFILE == "direct-profile"
    assert config.LLM_REGION == AWSRegion.EU_WEST_1
    assert config.LLM_MODEL == LLMModel.CLAUDE_SONNET_3_5


@pytest.mark.parametrize(
    "env_vars,expected_error",
    [
        (
            {
                "AWS_PROFILE": "test",
                "LLM_MODEL": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            },
            "LLM_REGION cannot be None or empty",
        ),
        ({"AWS_PROFILE": "test"}, "LLM_REGION cannot be None or empty"),
        (
            {
                "AWS_PROFILE": "test",
                "LLM_REGION": "invalid-region",
                "LLM_MODEL": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            },
            "Invalid region",
        ),
        (
            {
                "AWS_PROFILE": "test",
                "LLM_REGION": "us-west-2",
                "LLM_MODEL": "invalid-model",
            },
            "Invalid model",
        ),
    ],
)
def test_config_validation_errors(mock_env, env_vars, expected_error):
    """Test different validation error scenarios"""
    os.environ.update(env_vars)

    with pytest.raises(ConfigurationError, match=expected_error):
        Config()


def test_invalid_type_assignment():
    """Test assignment of invalid types"""
    with pytest.raises(ConfigurationError):
        Config(
            aws_profile=123,
            llm_region="us-west-2",
            llm_model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        )


@pytest.mark.parametrize(
    "invalid_region",
    [
        "invalid-region",
        "",
        None,
        123,
    ],
)
def test_invalid_region_values(invalid_region):
    """Test various invalid region values"""
    with pytest.raises(ConfigurationError):
        validate_llm_region(invalid_region)


@pytest.mark.parametrize(
    "invalid_model",
    [
        "invalid-model",
        "",
        None,
        123,
    ],
)
def test_invalid_model_values(invalid_model):
    """Test various invalid model values"""
    with pytest.raises(ConfigurationError):
        validate_llm_model(invalid_model)
