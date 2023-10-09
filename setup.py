# Copyright 2023 Stella Laurenzo
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import os
import distutils.command.build
from pathlib import Path
import sys

from setuptools import find_namespace_packages, setup

THIS_DIR = os.path.realpath(os.path.dirname(__file__))
VERSION_INFO_FILE = os.path.join(THIS_DIR, "version_info.json")


with open(
    os.path.join(
        THIS_DIR,
        "README.md",
    ),
    "rt",
) as f:
    README = f.read()


def load_version_info():
    with open(VERSION_INFO_FILE, "rt") as f:
        return json.load(f)


try:
    version_info = load_version_info()
except FileNotFoundError:
    print("version_info.json not found. Using defaults", file=sys.stderr)
    version_info = {}

PACKAGE_VERSION = version_info.get("package-version")
if not PACKAGE_VERSION:
    PACKAGE_VERSION = f"0.9.1dev1"


packages = find_namespace_packages(
    include=[
        "shark_turbine",
        "shark_turbine.*",
    ],
    where="python",
)

print("Found packages:", packages)

# Lookup version pins from requirements files.
requirement_pins = {}


def load_requirement_pins(requirements_file: str):
    with open(Path(THIS_DIR) / requirements_file, "rt") as f:
        lines = f.readlines()
    pin_pairs = [line.strip().split("==") for line in lines if "==" in line]
    requirement_pins.update(dict(pin_pairs))


load_requirement_pins("requirements.txt")
load_requirement_pins("pytorch-cpu-requirements.txt")


def get_version_spec(dep: str):
    if dep in requirement_pins:
        return f">={requirement_pins[dep]}"
    else:
        return ""


# Override build command so that we can build into _python_build
# instead of the default "build". This avoids collisions with
# typical CMake incantations, which can produce all kinds of
# hilarity (like including the contents of the build/lib directory).
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = "_python_build"


setup(
    name=f"shark-turbine",
    version=f"{PACKAGE_VERSION}",
    author="SHARK Authors",
    author_email="stella@nod.ai",
    description="SHARK Turbine Machine Learning Deployment Tools",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nod-ai/SHARK-Turbine",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],

    package_dir={
        "": "python",
    },
    packages=packages,
    entry_points={
        "torch_dynamo_backends": [
            "turbine_cpu = shark_turbine.dynamo.backends.cpu:backend",
        ],
    },
    install_requires=[
        "numpy",
        f"iree-compiler{get_version_spec('iree-compiler')}",
        f"iree-runtime{get_version_spec('iree-runtime')}",
        # Use the [torch-cpu-nightly] spec to get a more recent/specific version.
        "torch>=2.1.0",
    ],
    extras_require={
        "torch-cpu-nightly": [f"torch{get_version_spec('torch')}"],
        "testing": [
            "pytest",
            "pytest-xdist",
        ],
    },
    cmdclass={"build": BuildCommand},
)
