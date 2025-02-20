import os
import sys
import pytest
from unittest.mock import Mock, patch, ANY
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chai.constants import ChartType
from chai.requests import TypeHandler
from chai.chAI import chAI, ChAIError
from chai.config import Config
from chai.constants import LLMModel, AWSRegion


@pytest.fixture
def mock_image_handler():
    with patch("chai.chAI.ImageHandler") as mock:
        handler_instance = Mock()
        handler_instance.image_request.return_value = "Image prompt"
        mock.return_value = handler_instance
        yield mock


@pytest.fixture
def mock_config():
    """Mock Config class"""
    with patch("chai.chAI.Config") as mock:
        config_instance = Mock(spec=Config)
        config_instance.LLM_MODEL = LLMModel.CLAUDE_SONNET_3_5
        config_instance.LLM_REGION = AWSRegion.US_EAST_1
        config_instance.AWS_PROFILE = "test-profile"
        mock.return_value = config_instance
        yield mock


@pytest.fixture
def mock_bedrock():
    """Mock BedrockHandler"""
    with patch("chai.chAI.BedrockHandler") as mock:  # Updated patch path
        bedrock_instance = Mock()
        bedrock_instance.set_runtime.return_value = Mock()
        bedrock_instance.get_llm.return_value = Mock()
        mock.return_value = bedrock_instance
        yield mock


@pytest.fixture
def chai_instance(mock_config, mock_bedrock):
    """Create a chAI instance with mocked dependencies"""
    return chAI()


@pytest.fixture
def mock_agent_executor():
    with patch("chai.chAI.AgentExecutor") as mock:
        mock.return_value = Mock()
        yield mock


@pytest.fixture
def mock_create_json_chat_agent():
    with patch("chai.chAI.create_json_chat_agent") as mock:
        mock.return_value = Mock()
        yield mock


@pytest.fixture
def mock_dataframe_handler():
    with patch("chai.chAI.DataFrameHandler") as mock:
        handler_instance = Mock()
        handler_instance.dataframe_request.return_value = "DataFrame prompt"
        mock.return_value = handler_instance
        yield mock


@pytest.fixture
def mock_image_handler():
    with patch("chai.chAI.ImageHandler") as mock:
        handler_instance = Mock()
        handler_instance.image_request.return_value = "Image prompt"
        mock.return_value = handler_instance
        yield mock


@pytest.fixture
def mock_type_handler():
    with patch("chai.chAI.TypeHandler") as mock:
        handler_instance = Mock()
        handler_instance.chart_request.return_value = "Chart type prompt"
        mock.return_value = handler_instance
        yield mock


def test_chai_initialization(mock_config, mock_bedrock):
    """Test successful initialization of chAI class"""
    chai = chAI()

    assert chai.config is not None
    assert chai.bedrock is not None
    assert chai.bedrock_runtime is not None
    assert chai.llm is not None
    assert len(chai.tools) == 3
    assert chai.agent_executor is not None
    assert chai.visualisations is None

    # Verify the mocks were called correctly
    mock_config.assert_called_once()
    mock_bedrock.assert_called_once()


def test_set_agent_executor_success(
    chai_instance, mock_create_json_chat_agent, mock_agent_executor
):
    """Test successful creation of agent executor"""
    result = chai_instance.set_agent_executor(verbose=True, handle_parse=True)
    mock_create_json_chat_agent.assert_called_once_with(
        chai_instance.llm, chai_instance.tools, chai_instance.prompt
    )
    mock_agent_executor.assert_called_once_with(
        agent=mock_create_json_chat_agent.return_value,
        tools=chai_instance.tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    assert result == mock_agent_executor.return_value


def test_set_agent_executor_failure(chai_instance, mock_create_json_chat_agent):
    """Test handling of exceptions during agent executor creation"""
    mock_create_json_chat_agent.side_effect = Exception("Test error")
    with pytest.raises(Exception) as exc_info:
        chai_instance.set_agent_executor()
    assert str(exc_info.value) == "Test error"


def test_handle_request_dataframe(chai_instance, mock_dataframe_handler):
    """Test handling DataFrame input"""
    test_df = pd.DataFrame({"col1": [1, 2, 3]})
    chai_instance.agent_executor = Mock()
    chai_instance.agent_executor.invoke.return_value = {"output": "Test response"}
    # Setting handler manually because mocks aren't deal with properly
    chai_instance.dataframe_handler = mock_dataframe_handler.return_value
    result = chai_instance.handle_request(data=test_df, prompt="Test prompt")
    assert result == "Test response"
    chai_instance.dataframe_handler.dataframe_request.assert_called_once()


def test_handle_request_image(chai_instance, mock_image_handler, tmp_path):
    """Test handling image input"""
    # Create temporary test image empty file because needs a real file to work with
    test_image_path = tmp_path / "test_image.jpg"
    test_image_path.write_text("")
    chai_instance.agent_executor = Mock()
    chai_instance.agent_executor.invoke.return_value = {"output": "Test response"}
    # Setting handler manually because mocks aren't deal with properly
    chai_instance.image_handler = mock_image_handler.return_value
    result = chai_instance.handle_request(
        image_path=str(test_image_path), prompt="Test prompt"
    )
    assert result == "Test response"
    chai_instance.image_handler.image_request.assert_called_once()


def test_handle_request_chart_type(chai_instance, mock_type_handler):
    """Test handling chart type input"""
    test_chart_type = "bar"
    chai_instance.agent_executor = Mock()
    chai_instance.agent_executor.invoke.return_value = {"output": "Test response"}
    # Again setting handler manually because mocks aren't deal with properly
    chai_instance.type_handler = mock_type_handler.return_value
    result = chai_instance.handle_request(
        chart_type=test_chart_type, prompt="Test prompt"
    )
    assert result == "Test response"
    chai_instance.type_handler.chart_request.assert_called_once()


def test_handle_request_no_input(chai_instance):
    """Test handling no valid input"""
    with pytest.raises(ValueError) as exc_info:
        chai_instance.handle_request(prompt="Test prompt")
    assert str(exc_info.value) == "No valid input provided"


def test_handle_request_execution_error(chai_instance):
    """Test handling executor error"""
    # Setup
    test_df = pd.DataFrame({"col1": [1, 2, 3]})
    chai_instance.agent_executor = Mock()
    chai_instance.agent_executor.invoke.side_effect = Exception("Test error")

    # Execute and Assert
    with pytest.raises(ChAIError) as exc_info:
        chai_instance.handle_request(data=test_df, prompt="Test prompt")
    assert str(exc_info.value) == "Failed to process request: Test error"
