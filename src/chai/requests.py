import base64
import json
import logging
from datetime import datetime
from json import JSONEncoder
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .constants import APIVersion, ChartType, DataFrameLimits, MaxTokens
from .tools.default_charts import PlotlyTemplates

logger = logging.getLogger(__name__)


class DataFrameJSONEncoder(JSONEncoder):
    """Custom JSON encoder to handle pandas and numpy types"""

    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


class DataFrameHandler:
    def __init__(
        self,
        max_rows: int = DataFrameLimits.MAX_ROWS,
        include_summary: bool = True,
        sample_size: int = 10,
    ):
        self.max_rows = max_rows
        self.include_summary = include_summary
        self.sample_size = sample_size

    @staticmethod
    def _serialize_value(v: Any) -> Any:
        """
        Convert a value to JSON-serializable format.

        Args:
            v: Value to serialize

        Returns:
            JSON-serializable version of the value
        """
        if pd.isna(v):
            return None
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.isoformat()
        return str(v) if not isinstance(v, (int, float, bool, str)) else v

    def _serialize_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert DataFrame to JSON-serializable format.
        Handles timestamp and other non-serializable types.
        """
        records = df.head(self.sample_size).to_dict(orient="records")
        return [
            {k: self._serialize_value(v) for k, v in record.items()}
            for record in records
        ]

    def parse_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract useful information from a DataFrame.
        """
        logger.info("Parsing DataFrame into structured JSON dictionary")
        try:
            data_info = {
                "columns": [
                    {"name": col, "dtype": str(dtype)}
                    for col, dtype in df.dtypes.items()
                ],
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "sample_data": self._serialize_dataframe(df),
            }

            if self.include_summary:
                try:
                    summary_df = df.describe(include="all")
                    summary = {
                        col: {
                            k: self._serialize_value(v)
                            for k, v in summary_df[col].items()
                        }
                        for col in summary_df.columns
                    }
                    data_info["summary"] = summary
                except ValueError as e:
                    logger.warning(f"Could not generate summary statistics: {e}")
                    data_info["summary"] = {}

            return data_info
        except Exception as e:
            logger.error(f"Error parsing DataFrame: {str(e)}")
            raise

    def dataframe_request(self, data: pd.DataFrame, base_prompt: str) -> str:
        """
        Handle DataFrame analysis request.
        """
        if len(data) > self.max_rows:
            logger.info(
                f"DataFrame has more than {self.max_rows} rows. Trimming for processing."
            )
            data = data.head(self.max_rows)

        data_info = self.parse_dataframe(data)

        # Use custom JSON encoder
        sample_data_json = json.dumps(
            data_info["sample_data"], indent=2, cls=DataFrameJSONEncoder
        )

        dataframe_prompt = f"""
            DataFrame Information:
            Shape: {data_info["shape"]["rows"]} rows, {data_info["shape"]["columns"]} columns
            
            Columns:
            {", ".join(col["name"] for col in data_info["columns"])}
            
            Sample Data:
            {sample_data_json}

            Instructions:
            1. Analyse the DataFrame structure and content above
            2. Suggest meaningful visualisations based on the data and user's instructions
            3. For each visualisation, include:
            - Clear purpose
            - Chart type
            - Variables used
            - Expected insights
            4. Use the format_visualisation_output tool to structure your response
            5. Make sure to provide concrete, specific suggestions based on the actual data

            Remember to use the formatting tool for your final output.
            """

        return f"{base_prompt}\n\n{dataframe_prompt}"


class ImageHandler:
    """Handles image analysis and visualisation recreation requests."""

    def __init__(self):
        """Initialize ImageHandler."""
        self.logger = logging.getLogger(__name__)

    def encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encodes an image file to Base64 format.

        Args:
            image_path (Union[str, Path]): Path to the image file.

        Returns:
            str: Base64 encoded image string.

        Raises:
            Exception: If there's an error reading or encoding the image.
        """
        self.logger.info("Encoding image to base64")
        try:
            path = Path(image_path)
            with path.open("rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error encoding image: {str(e)}")
            raise

    def analyse_image(
        self,
        base64_data: str,
        bedrock_runtime: Any,
        model_id: str,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """
        Analyses an image using AWS Bedrock's Claude model.

        Args:
            base64_data (str): Base64-encoded image data
            bedrock_runtime: AWS Bedrock runtime client
            model_id (str): Model ID to use for analysis
            custom_prompt (Optional[str]): Additional analysis requirements

        Returns:
            str: Structured analysis of the image

        Raises:
            Exception: If analysis fails
        """
        try:
            if not base64_data:
                return "Error: No image data provided"

            analysis_prompt = f"""Analyse this image and provide a detailed analysis using the following structure:

                # Description
                [Provide a detailed description of what the image shows]

                # Chart Analysis
                ## Type
                [Specify the type of visualisation (e.g., bar chart, line plot, scatter plot)]

                ## Axes
                [List all axes and what they represent]

                ## Insights
                [Consider the following specific requirements in your analysis:
                {custom_prompt if custom_prompt else "No additional specific requirements stated"}
            
                Based on these requirements (if provided) and the image, provide detailed insights such as key patterns, trends or insights visible in the chart]

                # Plotly Recreation
                ## Code
                ```python
                [Provide a complete Plotly code snippet that could recreate this visualisation including the visible values for each variable]
                ```

                ## Data Structure
                [Describe the data structure needed for the Plotly code]"""

            body = {
                "anthropic_version": APIVersion.BEDROCK.value,
                "max_tokens": MaxTokens.DEFAULT,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_data,
                                },
                            },
                            {"type": "text", "text": analysis_prompt},
                        ],
                    }
                ],
            }

            response = bedrock_runtime.invoke_model(
                modelId=model_id, body=json.dumps(body).encode("utf-8")
            )

            response_body = json.loads(response.get("body").read())
            return response_body.get("content", [])[0].get("text", "")

        except Exception as e:
            self.logger.error(f"Error in analyse_image: {str(e)}")
            raise

    def image_request(
        self,
        image_path: Union[str, Path],
        bedrock_runtime: Any,
        model_id: str,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """
        Handle complete image analysis request.

        Args:
            image_path (Union[str, Path]): Path to image file
            bedrock_runtime: AWS Bedrock runtime client
            model_id (str): Model ID to use for analysis
            custom_prompt (Optional[str]): Additional analysis requirements

        Returns:
            str: Formatted prompt for visualisation creation

        Raises:
            Exception: If request handling fails
        """
        try:
            # Encode image
            image_base64 = self.encode_image(image_path)

            # Analyze image
            image_response = self.analyse_image(
                image_base64, bedrock_runtime, model_id, custom_prompt
            )

            # Return formatted prompt
            return f"""
                You are a data visualisation expert tasked with recreating an image using Plotly.

                First, analyze the image using these tools in order:
                1. format_image_analysis_output tool
                Input: {image_response}
                Store the JSON output for the next steps.

                2. save_plotly_visualisation tool
                Use this after creating your visualisation code.

                Steps to create the visualisation:
                1. From the formatted analysis:
                ```
                - Use chart_analysis.type for visualisation type
                - Use chart_analysis.axes for data relationships
                - Use chart_analysis.insights for key features
                - Use plotly_recreation for implementation details
                ```

                2. Create your visualisation with these requirements:
                ```
                - Match the identified chart type
                - Replicate color scheme and styling
                - Use 'plotly_white' template
                - Include proper labels and legends
                - Ensure professional formatting
                - Add clear title and axis labels
                ```

                Return ONLY a JSON dictionary in this exact format:
                ```json
                {{
                    "analysis": "## Insights\\n1. <insight1>\\n2. <insight2>\\n...",
                    "path": "<path returned by save_plotly_visualisation>",
                    "code": "<complete plotly code used>"
                }}
                ```

                Critical requirements:
                - Maintain markdown formatting in analysis section
                - Use exact path and code from save_plotly_visualisation tool response
                - Return only the JSON dictionary, no additional text
                - Ensure accurate recreation of the original image
            """

        except Exception as e:
            self.logger.error(f"Error handling image request: {str(e)}")
            raise


class TypeHandler:
    """Handles chart type-specific visualization requests."""

    def __init__(self):
        """Initialize TypeHandler."""
        self.logger = logging.getLogger(__name__)
        self.plotly_templates = PlotlyTemplates()

    def get_template(self, chart_type: ChartType) -> str:
        """
        Get the appropriate template for the requested chart type.

        Args:
            chart_type (ChartType): Type of chart requested

        Returns:
            str: Template code for the requested chart type

        Raises:
            Exception: If template retrieval fails
        """
        try:
            templates = self.plotly_templates.get_templates()
            chart_type_mapping = {
                ChartType.BAR: "bar_chart",
                ChartType.HISTOGRAM: "histogram",
                ChartType.SCATTER: "scatter_plot",
                ChartType.LINE: "line_chart",
            }

            if chart_type not in chart_type_mapping:
                self.logger.warning(
                    f"Unsupported chart type: {chart_type}. Defaulting to bar chart."
                )
                template_key = "bar_chart"
            else:
                template_key = chart_type_mapping[chart_type]

            return templates[template_key]

        except Exception as e:
            self.logger.error(f"Error getting template: {str(e)}")
            raise

    def chart_request(
        self,
        chart_type: ChartType,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """
        Handle complete chart type request.

        Args:
            chart_type (ChartType): Type of chart to create
            custom_prompt (Optional[str]): Additional requirements for the chart

        Returns:
            str: Formatted prompt for visualization creation

        Raises:
            Exception: If request handling fails
        """
        try:
            # Get appropriate template
            template_code = self.get_template(chart_type)

            # Return formatted prompt
            return f"""
                You are a data visualization expert using Plotly to create a {chart_type} chart.

                Here is your template code:
                ```python
                {template_code}
                ```

                Requirements:
                {custom_prompt if custom_prompt else "No additional specific requirements stated"}

                Instructions:
                1. Modify the template code to meet the requirements
                2. Use the save_plotly_visualisation tool to save your visualization
                3. Return ONLY a JSON dictionary containing the tool's response

                The JSON must be in this exact format:
                ```json
                {{
                    "path": "<path returned by tool>",
                    "code": "<complete code used>"
                }}
                ```

                Critical requirements:
                - Use the save_plotly_visualisation tool to save the visualization
                - Return only the JSON dictionary with path and code
                - No additional text or explanations
                - Ensure professional appearance and functionality
            """

        except Exception as e:
            self.logger.error(f"Error handling chart request: {str(e)}")
            raise
