from setuptools import find_packages, setup

setup(
    name="cifar10-simple",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torchvision==0.14.0",
        "wandb==0.16.1",
        "tqdm",
    ],
    description="Simplified CIFAR10 classifier for adversarial attacks exercise",
)
