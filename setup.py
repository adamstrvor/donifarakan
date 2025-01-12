from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="donifarakan",
    version="0.0.8",
    author="Adama Seydou Traore",
    author_email="adamaseydoutraore86@gmail.com",
    description="Donifarakan is a federated learning framework designed specially for the finance sector (banks, fintech companies, etc.), where the stakeholders will train a generalized model on their local data without sharing them in order to make predictions, prevent market risks, assess news impacts on the stock market, and more.  ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adamstrvor/donifarakan",  # Optional
    packages=find_packages(),
    install_requires=requirements, 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)