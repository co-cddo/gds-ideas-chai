import logging
import time
from pathlib import Path

import plotly.io as pio
from langchain.tools import Tool

logger = logging.getLogger(__name__)


class PlotlyPlotter:
    """Tool for saving plotly code into an HTML"""

    @staticmethod
    def save_plotly_visualisation(plotly_code: str, output_path: str = None) -> str:
        """
        Execute Plotly code and save the resulting visualisation as an HTML file.

        Parameters:
            plotly_code (str): String containing valid Plotly Python code
            output_path (str): Optional path where to save the HTML file

        Returns:
            str: Path to the saved HTML file
        """
        try:
            # Create a safe execution environment
            local_vars = {}

            # Execute the Plotly code to store the figure in local variables
            exec(plotly_code, globals(), local_vars)

            # Find the figure object in local variables
            fig = None
            for var in local_vars.values():
                if str(type(var)).find("plotly.graph_objs._figure.Figure") != -1:
                    fig = var
                    break

            if fig is None:
                raise ValueError("No Plotly figure found in the generated code")

            # Generate output path if not provided
            if output_path is None:
                output_dir = Path.cwd() / "plotly_visualisations"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"visualisation_{int(time.time())}.html"

            # Save the figure
            pio.write_html(fig, str(output_path))

            return {"path": str(output_path), "code": plotly_code}

        except Exception as e:
            logger.error(f"Error saving Plotly visualisation: {str(e)}")
            return f"Failed to save visualisation: {str(e)}"


def create_save_plotly_tool() -> Tool:
    saver = PlotlyPlotter()

    return Tool(
        name="save_plotly_visualisation",
        func=saver.save_plotly_visualisation,
        description="Saves Plotly visualisation code as an HTML file and returns the path and code",
    )
