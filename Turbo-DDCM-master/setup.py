from setuptools import setup, find_packages

setup(
    name="turbo_ddcm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "diffusers>=0.32.0",
        "transformers",
        "pillow",
        "numpy",
        "matplotlib",
    ],
)

