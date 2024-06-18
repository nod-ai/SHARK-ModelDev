import argparse
import sys
from iree import runtime as ireert
#from iree.runtime._binding import create_hal_driver


class vmfbRunner:
    def __init__(self, device, vmfb_path, external_weight_path=None, extra_plugin=None):
        flags = []

        # If an extra plugin is requested, add a global flag to load the plugin
        # and create the driver using the non-caching creation function, as
        # the caching creation function may ignore the flag.
        # if extra_plugin:
        #     ireert.flags.parse_flags(f"--executable_plugin={extra_plugin}")
        #     haldriver = create_hal_driver(device)

        # No plugin requested: create the driver with the caching create
        # function.
        #else:
        haldriver = ireert.get_driver(device)
        if "://" in device:
            try:
                device_idx = int(device.split("://")[-1])
                device_uri = None
            except:
                device_idx = None
                device_uri = device.split("://")[-1]
        else:
            device_idx = 0
            device_uri = None
        if device_uri:
            if not any(x in device for x in ["cpu", "task"]):
                allocators = ["caching"]
                haldevice = haldriver.create_device_by_uri(
                    device_uri, allocators=allocators
                )
            else:
                haldevice = haldriver.create_device_by_uri(device_uri)
        else:
            hal_device_id = haldriver.query_available_devices()[device_idx]["device_id"]
            if not any(x in device for x in ["cpu", "task"]):
                allocators = ["caching"]
                haldevice = haldriver.create_device(
                    hal_device_id, allocators=allocators
                )
            else:
                haldevice = haldriver.create_device(hal_device_id)

        self.config = ireert.Config(device=haldevice)
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
                del param_module
            del index

        self.ctx = ireert.SystemContext(
            vm_modules=vm_modules,
            config=self.config,
        )

    def unload(self):
        self.ctx = None
        self.config = None
