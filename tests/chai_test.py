import pytest
from unittest.mock import Mock, patch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chAI.constants import ChartType
from chAI.requests import TypeHandler
from chAI.chAI import chAI, ChAIError

# Sample template for testing
SAMPLE_TEMPLATE = """
import plotly.graph_objects as go
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y))

# Update layout
fig.update_layout(
    title='Sample Plot',
    xaxis_title='X',
    yaxis_title='Y'
)
"""


@pytest.fixture
def type_handler():
    return TypeHandler()


@pytest.fixture
def mock_plotly_templates():
    templates = {
        "bar_chart": SAMPLE_TEMPLATE,
        "scatter_plot": SAMPLE_TEMPLATE,
        "line_chart": SAMPLE_TEMPLATE,
        "histogram": SAMPLE_TEMPLATE,
    }
    return templates


class TestTypeHandler:
    """Tests for TypeHandler class"""

    def test_init(self, type_handler):
        """Test TypeHandler initialization"""
        assert isinstance(type_handler, TypeHandler)
        assert type_handler.logger is not None

    @patch("chAI.requests.PlotlyTemplates")
    def test_get_template_valid_type(
        self, mock_plotly, type_handler, mock_plotly_templates
    ):
        """Test getting template for valid chart type"""
        mock_plotly.return_value.get_templates.return_value = mock_plotly_templates

        template = type_handler.get_template(ChartType.BAR)
        assert template == SAMPLE_TEMPLATE

        template = type_handler.get_template(ChartType.SCATTER)
        assert template == SAMPLE_TEMPLATE

    @patch("chAI.requests.PlotlyTemplates")
    def test_get_template_invalid_type(
        self, mock_plotly, type_handler, mock_plotly_templates
    ):
        """Test getting template for invalid chart type defaults to bar chart"""
        mock_plotly.return_value.get_templates.return_value = mock_plotly_templates

        template = type_handler.get_template("invalid_type")
        assert template == SAMPLE_TEMPLATE  # Should return bar_chart template

    @patch("chAI.requests.PlotlyTemplates")
    def test_get_template_error(self, mock_plotly, type_handler):
        """Test error handling in get_template"""
        mock_plotly.return_value.get_templates.side_effect = Exception("Template error")

        with pytest.raises(Exception):
            type_handler.get_template(ChartType.BAR)

    def test_chart_request_valid(self, type_handler):
        """Test chart request with valid inputs"""
        with patch.object(type_handler, "get_template", return_value=SAMPLE_TEMPLATE):
            prompt = type_handler.chart_request(
                chart_type=ChartType.BAR, custom_prompt="Make it red"
            )

            assert isinstance(prompt, str)
            assert "Make it red" in prompt
            assert "BAR" in prompt
            assert SAMPLE_TEMPLATE in prompt

    def test_chart_request_no_prompt(self, type_handler):
        """Test chart request without custom prompt"""
        with patch.object(type_handler, "get_template", return_value=SAMPLE_TEMPLATE):
            prompt = type_handler.chart_request(chart_type=ChartType.BAR)

            assert isinstance(prompt, str)
            assert "No additional specific requirements stated" in prompt
            assert SAMPLE_TEMPLATE in prompt

    def test_chart_request_error(self, type_handler):
        """Test error handling in chart request"""
        with patch.object(
            type_handler, "get_template", side_effect=Exception("Template error")
        ):
            with pytest.raises(Exception):
                type_handler.chart_request(ChartType.BAR)


class TestChAIIntegration:
    """Integration tests for chAI with TypeHandler"""

    @pytest.fixture
    def chai_instance(self):
        with patch("chAI.chAI.BedrockHandler"), patch("chAI.chAI.hub.pull"), patch(
            "chAI.chAI.create_json_chat_agent"
        ):
            return chAI()

    def test_handle_request_chart_type(self, chai_instance):
        """Test handling chart type request"""
        with patch.object(
            chai_instance.type_handler, "chart_request"
        ) as mock_chart_request, patch.object(
            chai_instance.agent_executor, "invoke"
        ) as mock_invoke:

            mock_chart_request.return_value = "test prompt"
            mock_invoke.return_value = {
                "output": {"path": "test.html", "code": "test code"}
            }

            result = chai_instance.handle_request(
                chart_type=ChartType.BAR, prompt="Make it red"
            )

            assert isinstance(result, dict)
            assert "path" in result
            assert "code" in result

            mock_chart_request.assert_called_once_with(
                chart_type=ChartType.BAR, custom_prompt="Make it red"
            )

    def test_handle_request_error(self, chai_instance):
        """Test error handling in handle_request"""
        with patch.object(
            chai_instance.type_handler,
            "chart_request",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(ChAIError):
                chai_instance.handle_request(chart_type=ChartType.BAR)


def test_chart_type_enum():
    """Test ChartType enum values"""
    assert ChartType.BAR.value == "bar"
    assert ChartType.SCATTER.value == "scatter"
    assert ChartType.LINE.value == "line"
    assert ChartType.HISTOGRAM.value == "histogram"
