import argparse
import sys
from iree import runtime as ireert


class vmfbRunner:
    def __init__(self, device, vmfb_path, external_weight_path=None):
        self.config = ireert.Config(device)

        # TODO: enable multiple vmfb's
        mod = ireert.VmModule.mmap(self.config.vm_instance, vmfb_path)
        vm_modules = [
            mod,
            ireert.create_hal_module(self.config.vm_instance, self.config.device),
        ]

        # TODO: Enable multiple weight files
        if external_weight_path:
            index = ireert.ParameterIndex()
            index.load(external_weight_path)
            # TODO: extend scope
            param_module = ireert.create_io_parameters_module(
                self.config.vm_instance, index.create_provider(scope="model")
            )
            vm_modules.insert(0, param_module)

        self.ctx = ireert.SystemContext(
            vm_modules=vm_modules,
            config=self.config,
        )
