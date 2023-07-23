# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class GeneralError(Exception):
    ...


class MismatchedDeviceSetClearError(AssertionError):
    def __init__(self):
        super().__init__("Calls to Device.set()/clear() are mismatched or unbalanced.")


class NoCurrentDeviceError(Exception):
    def __init__(self):
        super().__init__(
            "You accessed a method which requires a current device but none was set on this thread. "
            "Either pass an explicit 'device=' or set a current device via "
            "`with device:`"
        )


class ApiSequencingError(Exception):
    ...


class UnknownDTypeError(ValueError):
    def __init__(self, dtype):
        self.dtype = dtype
        super().__init__(f"Unable to map torch dtype {dtype} to Turbine")
