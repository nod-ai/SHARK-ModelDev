# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type conversion utilities.

During export, we must convert between two dimensions of types:

1. Flat/Tree: User-facing code works on trees. Compiler functions
   are flat.
2. IR-Values vs Tensors.

This module helps navigate these different worlds.
"""

from typing import Any, Sequence

from iree.compiler.ir import (
    Type as IrType,
    Value,
)

from .procedural import (
    Intrinsic,
    ProcedureTrace,
)
