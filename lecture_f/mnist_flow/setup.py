from setuptools import find_packages
from setuptools import setup

setup(
    name="mnist-flow-trainer",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "lightning==2.1.2",
        "torchvision==0.14.0",
        "matplotlib==3.8.2",
        "pandas==2.1.4",
        "wandb==0.16.1",
        "jsonargparse[signatures]==4.27.1",
        "rich==13.7.0",
    ],
    description="REALNVP-like flow on MNIST",
)
