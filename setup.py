from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "alphago._go",
    ["engine/src/board.cpp", "engine/src/features.cpp", "engine/src/bindings.cpp"],
    include_dirs=["engine/src"],
    cxx_std=17,
    extra_compile_args=["-O3"],
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
