from setuptools import find_packages, setup

setup(
    name="medmnist-equivariant",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "lightning==2.1.2",
        "torchvision==0.14.0",
        "matplotlib==3.8.2",
        "wandb==0.16.1",
        "jsonargparse[signatures]==4.27.1",
        "rich==13.7.0",
        "medmnist>=2.0.0",
        "escnn>=1.0.0",
    ],
    description="Equivariant neural networks on MedMNIST",
)
