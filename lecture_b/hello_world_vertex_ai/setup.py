from setuptools import find_packages
from setuptools import setup

setup(
    name="hello-world",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch==1.13.1",
        "torchvision==0.14.1",
        "lightning==2.0.9",
        "python-json-logger",
    ],
    description="Hello World",
)
