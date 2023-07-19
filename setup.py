# Copyright 2023 Stella Laurenzo
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import os
import distutils.command.build
from setuptools import find_namespace_packages, setup

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
    version=f"0.1",
    package_dir={
        "": "python",
    },
    packages=packages,
    install_requires=[
        "numpy",
        # "iree-compiler",
        # "iree-runtime",
    ],
    entry_points={
        "torch_dynamo_backends": [
            "turbine_cpu = shark_turbine.dynamo.backends.cpu:backend",
        ],
    },
    extras_require={},
    cmdclass={"build": BuildCommand},
)
