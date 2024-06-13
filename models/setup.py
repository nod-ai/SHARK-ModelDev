import json
import os
from pathlib import Path

from setuptools import find_namespace_packages, setup


#### TURBINE MODELS SETUP ####


TURBINE_MODELS_DIR = os.path.realpath(os.path.dirname(__file__))
TURBINE_ROOT_DIR = Path(TURBINE_MODELS_DIR).parent
print(TURBINE_ROOT_DIR)
VERSION_INFO_FILE = os.path.join(TURBINE_ROOT_DIR, "version_info.json")


with open(
    os.path.join(
        TURBINE_MODELS_DIR,
        "README.md",
    ),
    "rt",
) as f:
    README = f.read()


def load_version_info():
    with open(VERSION_INFO_FILE, "rt") as f:
        return json.load(f)


version_info = load_version_info()
PACKAGE_VERSION = version_info["package-version"]

setup(
    name=f"turbine-models",
    version=f"{PACKAGE_VERSION}",
    author="SHARK Authors",
    author_email="dan@nod.ai",
    description="SHARK Turbine Machine Learning Model Zoo",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nod-ai/SHARK-Turbine",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_namespace_packages(
        include=[
            "turbine_models",
            "turbine_models.*",
        ],
    ),
    install_requires=[
        "Shark-Turbine",
        "protobuf",
        "sentencepiece",
        "transformers==4.37.1",
        "accelerate",
        "diffusers==0.29.0.dev0",
        "azure-storage-blob",
        "einops",
    ],
)
