import pytest
import os
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chAI.config import Config


@pytest.fixture(autouse=True)
def mock_env():
    """Mock entire environment"""
    original_environ = dict(os.environ)
    os.environ.clear()
    yield
    os.environ.clear()
    os.environ.update(original_environ)


def test_config_successful_initialisation(mock_env):
    """Test successful initialisation with all environment variables set"""
    os.environ.update(
        {
            "AWS_PROFILE": "test-profile",
            "LLM_REGION": "us-west-2",
            "LLM_MODEL": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        }
    )

    config = Config()
    assert config.AWS_PROFILE == "test-profile"
    assert config.LLM_REGION == "us-west-2"
    assert config.LLM_MODEL == "anthropic.claude-3-5-sonnet-20240620-v1:0"


def test_config_missing_single_field(mock_env):
    """Test initialisation fails when one environment variable is missing"""
    os.environ.update(
        {
            "AWS_PROFILE": "test-profile",
            "LLM_REGION": "us-west-2",
        }
    )

    with pytest.raises(ValueError) as exc_info:
        Config()
    assert "LLM_MODEL" in str(exc_info.value)


def test_config_missing_multiple_fields(mock_env):
    """Test initialisation fails when multiple environment variables are missing"""
    os.environ.update(
        {
            "AWS_PROFILE": "test-profile",
        }
    )

    with pytest.raises(ValueError) as exc_info:
        Config()
    assert any(field in str(exc_info.value) for field in ["LLM_REGION", "LLM_MODEL"])


def test_config_missing_all_fields(mock_env):
    """Test initialisation fails when all environment variables are missing"""
    with pytest.raises(ValueError) as exc_info:
        Config()
    assert any(
        field in str(exc_info.value)
        for field in ["AWS_PROFILE", "LLM_REGION", "LLM_MODEL"]
    )


def test_config_direct_assignment(mock_env):
    """Test successful initialisation with direct assignment"""
    config = Config(
        AWS_PROFILE="direct-profile",
        LLM_REGION="eu-west-1",
        LLM_MODEL="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    assert config.AWS_PROFILE == "direct-profile"
    assert config.LLM_REGION == "eu-west-1"
    assert config.LLM_MODEL == "anthropic.claude-3-5-sonnet-20240620-v1:0"


@pytest.mark.parametrize(
    "env_vars,expected_missing",
    [
        (
            {
                "AWS_PROFILE": "test",
                "LLM_MODEL": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            },
            ["LLM_REGION"],
        ),
        (
            {"AWS_PROFILE": "test"},
            ["LLM_REGION", "LLM_MODEL"],
        ),
    ],
)
def test_config_parametrised(mock_env, env_vars, expected_missing):
    """Test different combinations of missing fields"""
    os.environ.update(env_vars)

    with pytest.raises(ValueError) as exc_info:
        Config()
    for field in expected_missing:
        assert field in str(exc_info.value)
