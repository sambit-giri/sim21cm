import numpy as np 
from scipy.interpolate import splev, splrep
from scipy.misc import derivative
from tqdm import tqdm 

from astropy import units, constants, cosmology
import camb
from camb import model, initialpower


def sigma2_fit(z=0, cosmo='Planck15'):
	if type(cosmo)==str:
		if cosmo.lower() in ['planck15']: cosmo = cosmology.Planck15
		elif cosmo.lower() in ['planck13']: cosmo = cosmology.Planck13
		elif cosmo.lower() in ['wmap9']: cosmo = cosmology.WMAP9
		elif cosmo.lower() in ['wmap7']: cosmo = cosmology.WMAP7
		elif cosmo.lower() in ['wmap5']: cosmo = cosmology.WMAP5
		else:
			print('Setting to Planck15 cosmology.')
			cosmo = cosmology.Planck15
	#Now get matter power spectra and sigma8 at redshift 0 and z
	pars = camb.CAMBparams()
	pars.set_cosmology(H0=cosmo.H0.value, ombh2=cosmo.Ob0*cosmo.h**2, omch2=cosmo.Om0*cosmo.h**2)
	pars.InitPower.set_params(ns=0.96)
	pars.set_matter_power(redshifts=[z], kmax=2.0)
	#Linear spectra
	#pars.NonLinear = model.NonLinear_none
	results = camb.get_results(pars)
	kh, zs, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
	s8 = np.array(results.get_sigma8())

	rho_mean = cosmo.Om(z)*cosmo.critical_density(z)
	Ms   = 10**np.linspace(5,17,100)*units.solMass/cosmo.h
	fM2R = lambda M: np.cbrt(3*M/(4*np.pi*rho_mean)).to('Mpc')
	sigR = np.array([results.get_sigmaR(r.to('Mpc').value, z_indices=None, hubble_units=False, return_R_z=False) for r in tqdm(fM2R(Ms))]).squeeze()

	sigM_tck = splrep(Ms.value, sigR)
	return lambda x: splev(x.to('solMass').value if type(x)==units.quantity.Quantity else x, sigM_tck)


def Watson2013_fit(z=0, cosmo='Planck15', halo_finder='FoF'):
	if type(cosmo)==str:
		if cosmo.lower() in ['planck15']: cosmo = cosmology.Planck15
		elif cosmo.lower() in ['planck13']: cosmo = cosmology.Planck13
		elif cosmo.lower() in ['wmap9']: cosmo = cosmology.WMAP9
		elif cosmo.lower() in ['wmap7']: cosmo = cosmology.WMAP7
		elif cosmo.lower() in ['wmap5']: cosmo = cosmology.WMAP5
		else:
			print('Setting to Planck15 cosmology.')
			cosmo = cosmology.Planck15
	Az  = lambda z: cosmo.Om(z)*(0.990*(1+z)**-3.216+0.074)
	alz = lambda z: cosmo.Om(z)*(5.907*(1+z)**-3.599+2.344)
	bez = lambda z: cosmo.Om(z)*(3.136*(1+z)**-3.058+2.349)
	A  = 0.282 if z==0 else Az(z)
	al = 2.163 if z==0 else alz(z)
	be = 1.406 if z==0 else bez(z)
	ga = 1.210 
	func_fsig = lambda sig: A*((be/sig)**al+1)*np.exp(-ga/sig**2)

	func_sig  = sigma2_fit(z=z, cosmo=cosmo)
	func_invsig = lambda x: np.log(1/func_sig(np.exp(x)))
	Ms   = 10**np.linspace(5,17,100)*units.solMass/cosmo.h
	lnMs = np.log(Ms.value)
	dlninvsig_dlnM = np.array([derivative(func_invsig, lnMs0, dx=1e-6) for lnMs0 in tqdm(lnMs)])

	rho_mean = cosmo.Om(z)*cosmo.critical_density(z)
	dndlnM = (rho_mean/Ms)*func_fsig(func_sig(Ms))*dlninvsig_dlnM
	dndlnM = dndlnM.to('1/Mpc^3')
	dndlnM_tck  = splrep(Ms.value, dndlnM.value)
	func_dndlnM = lambda x: splev(x.to('solMass').value if type(x)==units.quantity.Quantity else x, dndlnM_tck)*dndlnM.unit
	# func_dndM   = lambda x: func_dndlnM(x)/x

	return {'M': Ms, 'multiplicity': func_fsig, 'dndlnM': func_dndlnM}



