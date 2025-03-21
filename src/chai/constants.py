from enum import Enum


class LLMModel(str, Enum):
    CLAUDE_HAIKU_3_5 = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_SONNET_3_5 = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_SONNET_3_7 = "anthropic.claude-3-7-sonnet-20250219-v1:0"


class AWSRegion(str, Enum):
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"


class ChartType(str, Enum):
    BAR = "bar"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    LINE = "line"


class APIVersion(str, Enum):
    BEDROCK = "bedrock-2023-05-31"


class MaxTokens:
    DEFAULT = 2000
    LARGE = 4000


class DataFrameLimits:
    MAX_ROWS = 100
