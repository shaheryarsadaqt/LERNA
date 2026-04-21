from setuptools import setup, find_packages

setup(
    name="lerna",
    version="3.0.0",
    packages=find_packages(),
    author="Shaheryar Sadaqat",
    author_email="research@lerna.ai",
    description="LERNA: Learning Efficiency Ratio Navigation & Adaptation - Energy-efficient LLM fine-tuning",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://gitlab.com/f43434f-group1/lerna",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3,<2.6",
        "transformers>=4.44,<5.0",
        "datasets>=2.18.0",
        "accelerate>=0.30.0",
        "pynvml>=11.5.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "pyyaml>=6.0",
    ],
    extras_require={
        "wandb": ["wandb>=0.15.0"],
        "viz": ["matplotlib>=3.5.0", "seaborn>=0.11.0", "plotly>=5.13.0"],
        "dev": ["pytest>=7.0", "black", "ruff"],
        "all": [
            "wandb>=0.15.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.13.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
