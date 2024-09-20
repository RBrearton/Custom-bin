#!/usr/bin/env python3
# encoding: utf-8

from setuptools import Extension, setup, find_packages
import sysconfig

EXTRA_COMPILE_ARGS = sysconfig.get_config_var('CFLAGS').split()
EXTRA_COMPILE_ARGS += ["-std=c99"]

ext_modules=[Extension('mapper_c_utils',
                          ['src/mapper_c_utils/mapper_c_utils.c'],
                          language='c99',
                          extra_compile_args=EXTRA_COMPILE_ARGS)]

if __name__ == "__main__":
    setup(name='mapper_c_utils',
          version='0.1',
          packages=find_packages(),
          ext_modules=ext_modules,
          
        )
