from typing import Dict, Any, List
import logging
import json
import pandas as pd
from datetime import datetime
from json import JSONEncoder
from chAI.constants import DataFrameLimits

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
            Shape: {data_info['shape']['rows']} rows, {data_info['shape']['columns']} columns
            
            Columns:
            {', '.join(col['name'] for col in data_info['columns'])}
            
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
