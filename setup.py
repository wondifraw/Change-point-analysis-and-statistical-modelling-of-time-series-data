from setuptools import setup, find_packages

setup(
    name="brent-oil-changepoint-analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ],
    python_requires=">=3.8",
    author="Data Analysis Team",
    description="Change point analysis for Brent oil price data"
)