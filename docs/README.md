![Banner](img/chai_cropped.png)

# chAI
An LLM-powered charting package to enable visualisation suggestion, image review and analysis, copycat functionality to replicate a provided chart in plotly, and output of standardised plotly code for bar, histogram, scatter and line charts.

Currently the library has only been tested with compatible Anthropic models. See the [AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for models that have compatible input modalities for image analysis.


## Prerequisites
- An environment with Python 3.13 or greater
- Foundation model access approved within AWS for models referenced in constants.py
- OPTIONAL: A .env file with an entry for AWS_PROFILE=. This profile/user should have sufficient permissions to execute a API calls without requiring MFA authentication in e.g. a notebook. Alternatively, users can configure the chAI runtime at initialisation.

## Installation
To use chAI:
- Create and activate a new virtual environment with Python 3.11 or greater
- Ensure you are in the root folder of the repo
- In the terminal type:

```pip install chai-llm```

You can now use chAI in your python notebook!

## Usage
chAI comes with 3 LLM models for selection by default - please note the capabilities of each one when deciding which is most appropriate for your use case:

- Claude 3.5 Haiku: Claude 3.5 Haiku offers quick code suggestions and completions to accelerate development workflows.
- Claude 3.5 Sonnet: Claude 3.5 Sonnet in addition to code completion is well suited to image analysis, particularly visual reasoning and interpreting charts and graphs, and can accurately transcribe text from imperfect images.
- Claude 3.7 Sonnet: Claude 3.7 Sonnet takes the capabilities of 3.5 further by combining the ordinary LLM with a reasoning model and is generally an upgraded version of Claude 3.5.

### Configuration
You can provide configuration parameters explicitly when initializing the chAI object. For convenience, we've defined Enum values in the AWSRegion and LLMModel classes that represent commonly used AWS regions and Claude model variants suitable for chAI:

```
from chai import chAI
from chai.constants import AWSRegion, LLMModel

chai = chAI(
    aws_profile="your-aws-profile",
    llm_region=AWSRegion.US_EAST_1,
    llm_model=LLMModel.CLAUDE_SONNET_3_7,
)
```

Alternatively, you can initialise chAI with an .env file with the following variables if you wish to avoid sharing credentials information in your code:
- AWS_PROFILE=your-aws-profile
- LLM_REGION=your-aws-region (you can use the Enum here)
- LLM_MODEL=your-chosen-model (you can use the Enum here)

and then run:
```
from chai import chAI
chai = chAI()
```

Refer to the chAI_functionality_examples.ipynb notebook to see the workflows in action. A sample input chart is provided for you to experiment with.

### Intelligent Visualisation Recommendations
This takes an input dataframe and reviews the first 100 rows of data to suggest potential visualisations for analysis. For each suggested chart:
- Purpose: This section outlines the goal of the visualisation. It describes what the chart is meant to show.
- Chart Type: This specifies the suitable visual you will use to present the data.
- Variables: This section defines the data being plotted on the x and y axes.
- Expected Insights: This part describes the specific insights you may gain from the visualisation.

```
# Example Code
sample_df = create_sample_dataset()
prompt = "Analyse this dataset and suggest visualisations to explore trends and insights."
response = chai.steep(
    data=sample_df,
    prompt=prompt,
)
print(response.suggestions)
```
![Visuals Ideas Example](img/visual_ideas.png)

### Chart Analysis and Replication with chAI
This takes a stored PNG image and attempts to 1. replicate the image using plotly into an HTML file 2. analyse the image. The analytical review can be boosted by providing some additional context as to what the image might show (for example the chart relates to outcomes of a specific subset of data).
This returns a JSON dictionary containing:
- analysis: A summary of what the LLM's analytical review
- code: The plotly code required to replicate the "look" of the chart (or as close as possible)
- path: The path where the HTML file has been saved

```
# Example Code
prompt = """
    Please analyse this chart and provide a comprehensive review with the following structure:

    1. Key Findings:
    - What are the most significant patterns or insights?
    - What are the highest and lowest values?
    - Are there any notable disparities or trends?

    2. Comparative Analysis:
    - Compare the different outcome groups
    - Identify any significant gaps or differences
    - Highlight any unexpected patterns

    Please provide your analysis in clear, business-friendly language suitable for stakeholders.
    """
image_analysis = chai.steep(prompt=prompt, image_path="../tests/img/satisfaction.png")
print(image_analysis.analysis) # Access the analysis
print(image_analysis.code) # View the generated Plotly code
print(image_analysis.path) # View the path to the locally generated chart
```
![Analysis and Code Example](img/analysis_code.png)

### Chart Template Generation
chAI has plotly code for several chart templates in its backend:
- Bar: Set chart_type to 'bar'
- Histogram: Set chart_type to 'histogram'
- Scatter: Set chart_type to 'scatter'
- Line: Set chart_type to 'line'

chAI can take user prompts to design a specified chart and provide required plotly code.

```
# Example Code
chart_prompt = "I want a red chart with bold axis titles. There should also be a legend showing each distinct category"
chart_generation = chai.steep(prompt=chart_prompt, chart_type="scatter")
print(chart_generation.code) # View the generated Plotly code
```
![Scatter Chart Example](img/request_scatter.png)


## Development
The project uses [uv](https://docs.astral.sh/uv/) for development. To contribute:

```
git clone https://github.com/co-cddo/gds-ideas-chai.git
cd gds-ideas-chai
uv sync --dev 
```