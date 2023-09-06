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
    PACKAGE_VERSION = f"0.dev0"


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
    ],
    extras_require={
        "torch": [f"torch{get_version_spec('torch')}"],
        "testing": [
            "pytest",
            "pytest-xdist",
        ],
    },
    cmdclass={"build": BuildCommand},
)
