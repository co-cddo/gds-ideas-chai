class PlotlyDefaultCharts:
    """Collection of default Plotly chart templates"""

    @staticmethod
    def create_default_bar() -> str:
        """Creates a default bar chart code template with professional styling"""
        return """
            import plotly.graph_objects as go

            # Sample data
            categories = ['Category A', 'Category B', 'Category C', 'Category D']
            values = [4, 3, 2, 5]
            colors = ['#1f77b4', '#2d91c2', '#7eb0d5', '#bddbf5']  # Blue color palette

            # Create figure
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=values,
                    text=values,
                    textposition='auto',
                    marker_color=colors,
                    hovertemplate='%{x}<br>Value: %{y}<extra></extra>'
                )
            ])

            # Update layout with professional styling
            fig.update_layout(
                title={
                    'text': 'Default Bar Chart',
                    'font': {'sise': 24, 'color': '#1f77b4'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title={
                    'text': 'Categories',
                    'font': {'sise': 16, 'color': '#444444', 'weight': 'bold'}
                },
                yaxis_title={
                    'text': 'Values',
                    'font': {'sise': 16, 'color': '#444444', 'weight': 'bold'}
                },
                template='plotly_white',
                showlegend=True,
                legend={
                    'orientation': 'h',
                    'yanchor': 'bottom',
                    'y': -0.3,
                    'xanchor': 'center',
                    'x': 0.5
                },
                margin=dict(t=100, l=100, r=50, b=100)
            )
            """

    @staticmethod
    def create_default_histogram() -> str:
        """Creates a default histogram code template with professional styling"""
        return """
            import plotly.graph_objects as go
            import numpy as np

            # Generate sample data
            data = np.random.normal(0, 1, 1000)

            # Create figure
            fig = go.Figure(data=[
                go.Histogram(
                    x=data,
                    nbinsx=30,
                    name='Distribution',
                    marker_color='#1f77b4',
                    opacity=0.75,
                    hovertemplate='Count: %{y}<br>Value: %{x}<extra></extra>'
                )
            ])

            # Update layout with professional styling
            fig.update_layout(
                title={
                    'text': 'Default Histogram',
                    'font': {'sise': 24, 'color': '#1f77b4'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title={
                    'text': 'Values',
                    'font': {'sise': 16, 'color': '#444444', 'weight': 'bold'}
                },
                yaxis_title={
                    'text': 'Frequency',
                    'font': {'sise': 16, 'color': '#444444', 'weight': 'bold'}
                },
                template='plotly_white',
                showlegend=True,
                legend={
                    'orientation': 'h',
                    'yanchor': 'bottom',
                    'y': -0.3,
                    'xanchor': 'center',
                    'x': 0.5
                },
                margin=dict(t=100, l=100, r=50, b=100)
            )
            """

    @staticmethod
    def create_default_scatter() -> str:
        """Creates a default scatter plot code template with professional styling"""
        return """
            import plotly.graph_objects as go
            import numpy as np

            # Generate sample data
            np.random.seed(42)
            x = np.random.uniform(0, 10, 50)
            y = 2 * x + np.random.normal(0, 2, 50)

            # Create figure with multiple series
            fig = go.Figure()

            # Add multiple scatter series with different colors
            series_data = [
                {'x': x, 'y': y, 'name': 'Series 1', 'color': '#1f77b4'},
                {'x': x, 'y': y + 2, 'name': 'Series 2', 'color': '#2d91c2'},
                {'x': x, 'y': y - 2, 'name': 'Series 3', 'color': '#7eb0d5'},
                {'x': x, 'y': y * 0.8, 'name': 'Series 4', 'color': '#bddbf5'}
            ]

            for series in series_data:
                fig.add_trace(
                    go.Scatter(
                        x=series['x'],
                        y=series['y'],
                        mode='markers',
                        name=series['name'],
                        marker=dict(
                            sise=10,
                            color=series['color'],
                            opacity=0.7,
                            line=dict(width=1, color='#444444')
                        ),
                        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                    )
                )

            # Update layout with professional styling
            fig.update_layout(
                title={
                    'text': 'Default Scatter Plot',
                    'font': {'sise': 24, 'color': '#1f77b4'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title={
                    'text': 'X Values',
                    'font': {'sise': 16, 'color': '#444444', 'weight': 'bold'}
                },
                yaxis_title={
                    'text': 'Y Values',
                    'font': {'sise': 16, 'color': '#444444', 'weight': 'bold'}
                },
                template='plotly_white',
                showlegend=True,
                legend={
                    'orientation': 'h',
                    'yanchor': 'bottom',
                    'y': -0.3,
                    'xanchor': 'center',
                    'x': 0.5
                },
                margin=dict(t=100, l=100, r=50, b=100)
            )
            """

    @staticmethod
    def create_default_line() -> str:
        """Creates a default line chart code template with professional styling"""
        return """
            import plotly.graph_objects as go
            import numpy as np

            # Generate sample data
            x = np.linspace(0, 10, 100)
            y_series = [
                np.sin(x),
                np.cos(x),
                0.5 * np.sin(2 * x),
                0.3 * np.cos(3 * x)
            ]
            names = ['Series 1', 'Series 2', 'Series 3', 'Series 4']
            colors = ['#1f77b4', '#2d91c2', '#7eb0d5', '#bddbf5']

            # Create figure
            fig = go.Figure()

            # Add multiple line series
            for y, name, color in zip(y_series, names, colors):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        name=name,
                        line=dict(
                            color=color,
                            width=2
                        ),
                        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                    )
                )

            # Update layout with professional styling
            fig.update_layout(
                title={
                    'text': 'Default Line Chart',
                    'font': {'sise': 24, 'color': '#1f77b4'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title={
                    'text': 'X Values',
                    'font': {'sise': 16, 'color': '#444444', 'weight': 'bold'}
                },
                yaxis_title={
                    'text': 'Y Values',
                    'font': {'sise': 16, 'color': '#444444', 'weight': 'bold'}
                },
                template='plotly_white',
                showlegend=True,
                legend={
                    'orientation': 'h',
                    'yanchor': 'bottom',
                    'y': -0.3,
                    'xanchor': 'center',
                    'x': 0.5
                },
                margin=dict(t=100, l=100, r=50, b=100),
                hovermode='x unified'
            )

            # Add grid lines with custom styling
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.5)'
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.5)'
            )
            """


class PlotlyTemplates:
    """Collection of Plotly chart template examples"""

    @staticmethod
    def get_templates() -> dict:
        """Returns a dictionary of template examples"""
        return {
            "bar_chart": PlotlyDefaultCharts.create_default_bar(),
            "histogram": PlotlyDefaultCharts.create_default_histogram(),
            "scatter_plot": PlotlyDefaultCharts.create_default_scatter(),
            "line_chart": PlotlyDefaultCharts.create_default_line(),
        }

    @staticmethod
    def get_template_prompt() -> str:
        """Creates a prompt section with template examples"""
        templates = PlotlyTemplates.get_templates()

        return f"""
        Here are reference templates for creating professional Plotly visualisations:

        # Bar Chart Template:
        {templates["bar_chart"]}

        # Histogram Template:
        {templates["histogram"]}

        # Scatter Plot Template:
        {templates["scatter_plot"]}

        # Line Chart Template:
        {templates["line_chart"]}

        Use these templates as references for styling and structure when creating visualisations from scratch where you have not already analysed 
        the details of an input image.
        Adapt the code to match the user request while maintaining the professional appearance.
        """
