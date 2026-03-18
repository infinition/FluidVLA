from setuptools import setup, find_packages

setup(
    name="fluidvla",
    version="0.1.0",
    description="Transformer-free Vision-Language-Action model using reaction-diffusion PDEs",
    author="Fabien Polly",
    url="https://github.com/infinition/FluidVLA",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "numpy",
        "tqdm",
        "tensorboard",
        "Pillow",
    ],
)
