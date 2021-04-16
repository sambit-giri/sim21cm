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

from . import *
from .radtrans import * 

from . import halo_finder
