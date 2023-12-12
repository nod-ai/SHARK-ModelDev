import logging
import unittest

from shark_turbine.kernel.compiler import (
    builder,
    dispatch_codegen,
)


class Test(unittest.TestCase):
    def testEmptyStreamExecutable(self):
        mb = builder.ModuleBuilder()
        try:
            exe = dispatch_codegen.StreamExecutable(mb)
            workgroup_builder, disp_builder = exe.define_entrypoint(
                "dispatch", 2, 3, 1, 3
            )
            workgroup_builder.terminate(workgroup_builder.workload)
            disp_builder.terminate()
        finally:
            print(mb.module_op.get_asm())
        mb.module_op.verify()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
