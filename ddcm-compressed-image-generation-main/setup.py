from setuptools import setup, find_packages

setup(
    name="ddcm",
    version="0.1.0",
    packages=find_packages(exclude=['assets', '__pycache__', 'extras']),
    package_dir={'': '.'},
    py_modules=['ddcm_api'],
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "diffusers>=0.31.0",
        "transformers>=4.37.2",
        "pillow",
        "numpy",
        "matplotlib",
        "tqdm",
    ],
    python_requires=">=3.8",
)

