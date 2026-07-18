"""
Setup script for the AI-GEO-NGO package.
"""
from setuptools import setup, find_packages

setup(
    name="ai-geo-ngo",
    version="0.1.0",
    description="AI-Enabled Geospatial Decision Support System for WASH planning",
    author="WASH Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Dependencies are managed in requirements.txt
    ],
    python_requires=">=3.9",
)
