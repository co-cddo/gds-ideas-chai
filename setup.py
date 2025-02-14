from setuptools import setup, find_packages

VERSION = "0.1"
DESCRIPTION = "An LLM powered charting package"
LONG_DESCRIPTION = """An LLM-powered charting package to enable visualisation 
suggestion, image review and analysis, copycat functionality to 
replicate a provided chart in plotly, and output of standardised plotly code for bar, 
histogram, scatter and line charts."""


setup(
    name="chAI",
    version=VERSION,
    author="Jose Orjales",
    author_email="jose.orjales@digital.cabinet-office.gov.uk",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "pip>=24.0",
        "boto3>=1.36.7",
        "ipykernel>=6.29.5",
        "ipywidgets>=8.1.5",  # Need notebook support so can have notebook approach as well as in a .py file
        "langchain>=0.3.16",
        "langchain-aws>=0.2.11",
        "langchain-community>=0.3.16",
        "langchain-core>=0.3.32",
        "langchain-experimental>=0.3.4",
        "langchain-text-splitters>=0.3.5",
        "nbformat>=5.10.4",
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "plotly>=5.24.1",
        "python-dotenv>=1.0.1",
    ],
    python_requires=">=3.11",
    extras_require={
        "test": ["pytest"],
        "dev": [
            "black",
            "mkdocs",
            "mkdocs-material",
            "mkdocs-glightbox",
            "setuptools",
            "build",
            "twine",
        ],
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
    ],
    url="https://github.com/co-cddo/c-af-chartist",
    project_urls={
        "Bug Tracker": "https://github.com/co-cddo/c-af-chartist/issues",
    },
    include_package_data=True,
)
