from setuptools import setup, find_packages

setup(
    name="lerna",
    version="0.1.0",
    packages=find_packages(),
    author="Your Name",
    author_email="your.email@example.com",
    description="Lerna project for GLUE benchmark experiments",
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "datasets>=2.0.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.19.0",
        "tqdm>=4.0.0",
    ],
)
