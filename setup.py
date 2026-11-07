from setuptools import find_namespace_packages, setup

setup(
    name="fcformer",
    version="1.0.0",
    description="FC-Former: Fully-Connected Transformer for Multi-Source Image Fusion",
    license="GPL-3.0",
    packages=find_namespace_packages(include=["fcformer", "fcformer.*"]),
    python_requires=">=3.8",
    install_requires=["torch", "einops", "timm>=0.9", "safetensors"],
)
