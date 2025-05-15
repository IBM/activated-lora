from setuptools import setup, find_packages

setup(
    name='alora',
    version='0.1.0',
    description='Activated LoRA implementation utilizing HuggingFace packages.',
    author='Kristjan Greenewald',
    author_email='kristjan.h.greenewald@ibm.com',
    packages=find_packages(),
    install_requires=[
        "click",  "peft>=0.14","transformers" # Add any required dependencies here
    ],
    python_requires='>=3.8',
)
