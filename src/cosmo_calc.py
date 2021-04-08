import numpy as np 
from astropy import units, constants

def critical_density0(cosmo=None, h=0.7, hunits=False):
	G = constants.G
	h = h if cosmo is None else cosmo.h
	H = h*100*units.km/(units.Mpc*units.s)
	rho_crit = 3*H**2/(8*np.pi*G)
	rho_crit = rho_crit.to('solMass/Mpc^3')
	return rho_crit/h**2 if hunits else rho_crit
