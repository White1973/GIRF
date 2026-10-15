"""Compatibility setuptools entry point; prefer ``pip install -e .``."""

from setuptools import find_packages, setup


setup(
    name="qwenvl-girf",
    version="0.1.0",
    description="GIRF multimodal positional-relation research implementation",
    python_requires=">=3.10",
    packages=find_packages(include=["qwenvl", "qwenvl.*", "experiments", "experiments.*"]),
)
