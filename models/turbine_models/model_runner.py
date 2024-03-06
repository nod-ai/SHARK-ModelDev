import argparse
import sys
from iree import runtime as ireert


class vmfbRunner:
    def __init__(self, device, vmfb_path, external_weight_path=None):
        self.config = ireert.Config(device)
        mods = []
        if not isinstance(vmfb_path, list):
            vmfb_path = [vmfb_path]
        for path in vmfb_path:
            mods.append(ireert.VmModule.mmap(self.config.vm_instance, path))
        vm_modules = [
            *mods,
            ireert.create_hal_module(self.config.vm_instance, self.config.device),
        ]

        # TODO: Enable multiple weight files
        if external_weight_path:
            index = ireert.ParameterIndex()
            if not isinstance(external_weight_path, list):
                external_weight_path = [external_weight_path]
            for i, path in enumerate(external_weight_path):
                if path in ["", None]:
                    continue
                index.load(path)
                # TODO: extend scope
                param_module = ireert.create_io_parameters_module(
                    self.config.vm_instance, index.create_provider(scope="model")
                )
                vm_modules.insert(i, param_module)

        self.ctx = ireert.SystemContext(
            vm_modules=vm_modules,
            config=self.config,
        )
