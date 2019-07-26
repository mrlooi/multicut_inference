from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np


extensions = [
  Extension(
    'MCInference', ['MCInference.pyx'],
    language="c++",
    include_dirs=[np.get_include(), './include'],
    extra_compile_args=['-DILOUSESTL','-DIL_STD','-std=c++11','-O3', '-DHAVE_CPP11_INITIALIZER_LISTS'],
    extra_link_args=['-std=c++11']
  )
] 

setup(
    name = 'MCInference',
    ext_modules = cythonize(extensions)
)
