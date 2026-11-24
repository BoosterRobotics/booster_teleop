# XRoboToolkit native SDK for AArch64

This directory contains the ARM64 native client library needed by
`xrobotoolkit_sdk` on the onboard computer.

Contents:

- `lib/libPXREARobotSDK.so`: stripped AArch64 ELF shared library.
- `include/PXREARobotSDK.h`: matching public C API header.
- `LICENSE`: upstream Apache-2.0 license.

Source provenance:

- Repository: `XR-Robotics/XRoboToolkit-PC-Service`
- Branch: `orin`
- Commit: `d1b45d1f36a3b0f5edf4095e00397d9344c5bdc0`
- Native library SHA256: `d97b4afd2806a9228dac5a9932a6d4282d4756ccdf3e404988990216ce3d9a2b`
- Header SHA256: `4665952f6505d721fadff1c442fbed7fb3d0cc97637f3f18bdb75dac838f3dae`

Runtime requirement: AArch64 Linux with glibc 2.34 or newer. The onboard
environment script checks both the glibc version and unresolved shared-library
dependencies before building the Python extension.

The Python extension is built on the target board by
`bash scripts/install_env.sh` (select Onboard), using the local source in `python/` and
the pinned build dependencies. Environment sourcing does not install packages.

Python binding source and nlohmann/json header provenance:

- Repository: `XR-Robotics/XRoboToolkit-PC-Service-Pybind`
- Commit: `a7ae949849ef335f0fa7bbef1e741c5b35e2124e`
- Retained: `bindings/py_bindings.cpp`, `include/nlohmann/json.hpp`, `LICENSE`
- Packaging: local setuptools/pybind11 build, linking this directory's ARM64 SDK
- The bundled json header contains its upstream MIT license notice.

The same architecture-independent Python binding source and public header are
also reused by the Host installer. It stages them in a temporary directory with
the native library from `../xrobotoolkit_sdk_x86_64`, without changing this
directory's AArch64 library. See that directory's README for host provenance.
