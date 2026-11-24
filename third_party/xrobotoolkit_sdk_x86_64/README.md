# XRoboToolkit native SDK for Linux x86_64

This directory contains the pinned native client library used to build the Host
`xrobotoolkit_sdk` Python extension. The receiver service is installed separately
by `scripts/install_env.sh`; it remains necessary for receiving Pico data.

Contents:

- `lib/libPXREARobotSDK.so`: unmodified x86_64 ELF shared library (about 23 MiB).
- `LICENSE`: upstream Apache-2.0 license.

Source provenance:

- Repository: https://github.com/XR-Robotics/XRoboToolkit-PC-Service
- Release: `v1.0.0`
- Release commit: `6386009475369615fa984bca0a6e9902cd81eb2a`
- Package: `XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb`
- Download: https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
- Package SHA256: `61961067eb4b41f81ed7cae35f4690dbb0ddfefb329a12b24e0b90ebc46ada91`
- Original member: `opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so`
- Library SHA256: `2348e1b0bf7f05cd13d95d628f04237b6b9fc50c6008b82f9bf4b046fc9373e6`

Select Host in `bash scripts/install_env.sh`. After `uv sync --locked`, the
installer checks the library's architecture and SHA256, stages the shared
binding source and public header from `../xrobotoolkit_sdk_aarch64` together with
this library, then builds and installs `xrobotoolkit-sdk==1.0.2` into the selected
Python 3.12 environment. Build dependencies are pinned in the binding's
`pyproject.toml`; `build-essential` is installed if `g++` is missing.

The wheel includes the native library beside the Python extension, with an
`$ORIGIN` runtime search path, and preserves the MIT binding license and
Apache-2.0 native library license. Building and importing do not require a
pre-existing `/opt/apps/roboticsservice/SDK/x64` library. Rerunning the installer
rebuilds the Host binding after environment synchronization.

The final environment check verifies import and callback/polling APIs without
starting an SDK session. AArch64 installation continues to use its own bundled
native library and locked onboard dependency.
