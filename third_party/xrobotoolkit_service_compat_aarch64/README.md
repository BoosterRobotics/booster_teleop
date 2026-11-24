# XRoboToolkit RoboticsService AArch64 compatibility libraries

This directory contains the ICU 70 runtime libraries required by the
XRoboToolkit `RoboticsServiceProcess` shipped with the onboard UFO/T2 bundle.
The target board runs Ubuntu 24.04 with ICU 74, while that service binary was
built against ICU 70.

Source package:

- Ubuntu package: `libicu70_70.1-2ubuntu1_arm64.deb`
- Archive: `https://ports.ubuntu.com/ubuntu-ports/pool/main/i/icu/`
- Architecture: AArch64

Only `libicudata`, `libicui18n`, and `libicuuc` are included. The environment
wrapper prepends `lib/` to `LD_LIBRARY_PATH`; it does not install or replace
system ICU packages.
