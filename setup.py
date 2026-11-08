from setuptools import setup, find_packages

setup(
    name="safegrid_ai",
    version="1.0.0",
    author="SafeGrid AI Team",
    description="Safety evacuation planning using LLMs based on CubiCasa5k architectural blueprints",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/safegrid_ai",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "networkx>=3.0",
        "shapely>=2.0",
        "svgpathtools>=1.6",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "numpy>=1.24",
        "pandas>=1.5",
        "google-generativeai>=0.8",
        "langchain>=0.2",
        "langchain-google-genai>=2.0",
        "tabulate>=0.9",
        "tqdm>=4.60",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
