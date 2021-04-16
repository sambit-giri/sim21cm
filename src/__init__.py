'''
sim21cm package is meant for simulating the 21cm signals from the 
Epoch of Reionization (EoR) and Cosmic Dawn (CD).
We incorpoarte its predecessor, c2raytools, into this package.
You can also get documentation for all routines directory from
the interpreter using Python's built-in help() function.
For example:
>>> import sim21cm as s2c
>>> help(s2c.calc_dt)
'''


try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

from . import *
# from .radtrans import H_He_final, Source

from . import halo_finder, RT_1D
