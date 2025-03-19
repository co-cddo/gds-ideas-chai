from langchain.tools import Tool


class DataFrameOutputFormatter:
    """Tool for formatting DataFrame analysis output"""

    @staticmethod
    def format_visualisation_output(analysis: str) -> str:
        """
        Format visualisation suggestions in a structured way

        Args:
            analysis (str): Raw analysis text

        Returns:
            str: Formatted analysis
        """
        try:
            # Split into individual visualisation sections
            sections = analysis.split("\n\n")
            formatted_output = ["📊 VISUALISATION SUGGESTIONS:\n"]

            current_section = []
            for section in sections:
                if section.strip():
                    # Check if this is a new visualisation section
                    if any(section.strip().startswith(str(i)) for i in range(1, 10)):
                        # Add previous section if it exists
                        if current_section:
                            formatted_output.append("\n".join(current_section))
                            formatted_output.append("-" * 50)
                            current_section = []

                        # Start new section
                        title = (
                            section.split(".")[1].strip() if "." in section else section
                        )
                        current_section.append(f"📈 {title.upper()}")

                    # Format bullet points
                    elif section.strip().startswith("-"):
                        points = section.split("\n")
                        for point in points:
                            if point.strip():
                                formatted_point = point.replace("   - ", "• ")
                                current_section.append(f"  {formatted_point.strip()}")
                    else:
                        current_section.append(section)

            # Add last section
            if current_section:
                formatted_output.append("\n".join(current_section))

            return "\n\n".join(formatted_output)

        except Exception as e:
            return f"Error formatting output: {str(e)}\nOriginal analysis:\n{analysis}"


def create_formatting_tool() -> Tool:
    formatter = DataFrameOutputFormatter()

    return Tool(
        name="format_visualisation_output",
        func=formatter.format_visualisation_output,
        description="Formats DataFrame visualisation suggestions in a clear, structured way. Use this tool to format your final output when asked to visualise possible charts from a dataframe",
        return_direct=True,
    )
