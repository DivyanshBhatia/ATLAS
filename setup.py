from setuptools import setup, find_packages

setup(
    name="atlas-peft",
    version="0.1.0",
    description="ATLAS: Adaptation Theory — Limits, Approximation, and Selection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
