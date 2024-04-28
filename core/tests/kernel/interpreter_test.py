import logging
import unittest
import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.functional as tkf


class Test(unittest.TestCase):
    def testInterpreter(self):
        interpreter = tkf.Interpreter()
        interpreter.interpret("/Users/harsh/SHARK-Turbine/inspect.mlir")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
