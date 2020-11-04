#!/usr/bin/env python
from typeguard.importhook import install_import_hook

install_import_hook(packages=['numba'])


import runpy
import os

# ensure full tracebacks are available and no help messages appear in test mode
os.environ['NUMBA_DEVELOPER_MODE'] = '1'


if __name__ == "__main__":
    runpy.run_module('numba.runtests', run_name='__main__')
