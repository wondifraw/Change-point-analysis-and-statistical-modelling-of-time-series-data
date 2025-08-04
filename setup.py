"""
Setup script for Change Point Analysis Project
Enables pip installation and development setup
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="brent-oil-changepoint-analysis",
    version="2.0.0",
    author="Change Point Analysis Team",
    author_email="team@changepointanalysis.com",
    description="Advanced Bayesian change point detection for Brent oil price analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/Change-point-analysis-and-statistical-modelling-of-time-series-data",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
        ],
        "bayesian": [
            "pymc3>=3.11.0",
            "theano-pymc>=1.1.0",
            "arviz>=0.12.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "changepoint-analysis=changepoint_detection:main",
            "oil-dashboard=src.task3.backend.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.html", "*.md", "*.yml", "*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/username/Change-point-analysis-and-statistical-modelling-of-time-series-data/issues",
        "Source": "https://github.com/username/Change-point-analysis-and-statistical-modelling-of-time-series-data",
        "Documentation": "https://change-point-analysis.readthedocs.io/",
    },
)