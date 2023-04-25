import distutils.core
import Cython.Build
distutils.core.setup(
    ext_modules = Cython.Build.cythonize(["raytracer_premature.pyx", "raytracer_thoramature.pyx", "bvh.pyx"]))

