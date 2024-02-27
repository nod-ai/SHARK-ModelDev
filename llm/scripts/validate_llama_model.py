# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

from turbine_llm.config import *
from turbine_llm.data import *
from turbine_llm.models.llama import *


def main(args: list[str]):
    config = load_gguf_file(args[0])
    hp = LlamaHParams.from_gguf_props(config.properties)
    model = LlamaModelV1(config.root_theta, hp)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
