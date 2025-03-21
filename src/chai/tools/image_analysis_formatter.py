import logging
import re
from typing import Dict

from langchain.tools import Tool

logger = logging.getLogger(__name__)


class ImageAnalysisFormatter:
    """Tool for formatting DataFrame analysis output"""

    @staticmethod
    def format_image_analysis_output(image_information: str) -> Dict:
        """
        Convert markdown-formatted image analysis into a structured JSON dictionary.

        Parameters:
            image_information (str): Markdown-formatted analysis from Claude

        Returns:
            Dict: Structured analysis in JSON format
        """
        try:
            # Initialize the structure
            analysis_dict = {
                "description": "",
                "chart_analysis": {"type": "", "axes": [], "insights": []},
                "plotly_recreation": {"code": "", "data_structure": ""},
            }

            # Extract description
            description_match = re.search(
                r"# Description\n(.*?)\n#", image_information, re.DOTALL
            )
            if description_match:
                analysis_dict["description"] = description_match.group(1).strip()

            # Extract chart type
            type_match = re.search(r"## Type\n(.*?)\n##", image_information, re.DOTALL)
            if type_match:
                analysis_dict["chart_analysis"]["type"] = type_match.group(1).strip()

            # Extract axes
            axes_match = re.search(r"## Axes\n(.*?)\n##", image_information, re.DOTALL)
            if axes_match:
                axes_text = axes_match.group(1).strip()
                # Split by newlines and clean up
                axes = [
                    axis.strip().replace("[", "").replace("]", "")
                    for axis in axes_text.split("\n")
                    if axis.strip()
                ]
                analysis_dict["chart_analysis"]["axes"] = axes

            # Extract insights
            insights_match = re.search(
                r"## Insights\n(.*?)\n#", image_information, re.DOTALL
            )
            if insights_match:
                insights_text = insights_match.group(1).strip()
                # Split by newlines and clean up
                insights = [
                    insight.strip().replace("[", "").replace("]", "")
                    for insight in insights_text.split("\n")
                    if insight.strip()
                ]
                analysis_dict["chart_analysis"]["insights"] = insights

            # Extract Plotly code
            code_match = re.search(r"```python\n(.*?)```", image_information, re.DOTALL)
            if code_match:
                analysis_dict["plotly_recreation"]["code"] = code_match.group(1).strip()

            # Extract data structure
            data_structure_match = re.search(
                r"## Data Structure\n(.*?)\$", image_information, re.DOTALL
            )
            if data_structure_match:
                analysis_dict["plotly_recreation"]["data_structure"] = (
                    data_structure_match.group(1).strip()
                )

            return analysis_dict

        except Exception as e:
            logger.error(f"Error formatting image analysis: {str(e)}")
            return {"error": f"Failed to format image analysis: {str(e)}"}


def create_analysis_formatter_tool() -> Tool:
    formatter = ImageAnalysisFormatter()

    return Tool(
        name="format_image_analysis_output",
        func=formatter.format_image_analysis_output,
        description="Formats markdown image analysis into structured JSON",
    )
