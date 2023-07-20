# Copyright 2023 Stella Laurenzo
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import os
import distutils.command.build
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
    PACKAGE_VERSION = f"0.dev0.1"


packages = find_namespace_packages(
    include=[
        "shark_turbine",
        "shark_turbine.*",
    ],
    where="python",
)

print("Found packages:", packages)


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
    install_requires=[
        "numpy",
        f"shark-turbine-iree-compiler=={PACKAGE_VERSION}",
        f"shark-turbine-iree-runtime=={PACKAGE_VERSION}",
    ],
    entry_points={
        "torch_dynamo_backends": [
            "turbine_cpu = shark_turbine.dynamo.backends.cpu:backend",
        ],
    },
    extras_require={},
    cmdclass={"build": BuildCommand},
)
