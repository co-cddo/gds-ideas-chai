from typing import TypedDict, Any, Dict, List


class DataFrameInfo(TypedDict):
    columns: List[Dict[str, str]]
    shape: Dict[str, int]
    summary: Dict[str, Any]
    sample_data: List[Dict[str, Any]]
