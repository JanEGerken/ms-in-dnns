from setuptools import find_packages
from setuptools import setup

setup(
    name="hello-world",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["lightning==2.1.2"],
    description="Hello World",
)
