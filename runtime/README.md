# SHARK-Turbine Runtime Components

This should be considered an overlay project to the upstream IREE
`runtime/` directory. It contains extension and packaging support.

## Developer notes

The Python packaging and README are symlinked from the IREE source tree.
Since the source directory structure matches and CMake install tree
(which is project invariant) is used for actually building the wheel,
this just works. However, at some point, we may want to modularize the
upstream `setup.py` so that these can diverge without completely
being copy-paste.
