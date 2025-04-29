from setuptools import setup, find_packages

setup(
    name="jssp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "streamlit",
        "plotly",
    ],
    python_requires=">=3.7",
) 