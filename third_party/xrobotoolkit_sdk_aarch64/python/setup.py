from pathlib import Path
import shutil

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parent


class BuildExt(build_ext):
    def run(self):
        # A previous interrupted uv/setuptools build can leave a zero-byte
        # extension in build/. Always compile the native module from scratch.
        self.force = True
        super().run()
        native_library = SDK / "lib/libPXREARobotSDK.so"
        self.check_elf(native_library)
        shutil.copy2(native_library, self.build_lib)
        for output in self.get_outputs():
            self.check_elf(Path(output))

    @staticmethod
    def check_elf(path):
        with path.open("rb") as stream:
            header = stream.read(64)
        if len(header) < 64 or header[:5] != b"\x7fELF\x02":
            raise RuntimeError(f"Invalid 64-bit ELF SDK build output: {path}")

    def get_outputs(self):
        return super().get_outputs() + [str(Path(self.build_lib) / "libPXREARobotSDK.so")]


setup(
    ext_modules=[
        Pybind11Extension(
            "xrobotoolkit_sdk",
            ["bindings/py_bindings.cpp"],
            include_dirs=[str(ROOT / "include"), str(SDK / "include")],
            library_dirs=[str(SDK / "lib")],
            libraries=["PXREARobotSDK"],
            runtime_library_dirs=["$ORIGIN"],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
