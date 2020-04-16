import numpy as np
import c2raytools as c2t


def Mgrid_2_Msolar(M):
	return M*(c2t.conv.M_grid*c2t.const.solar_masses_per_gram)

def Msolar_2_Mgrid(M):
	return M/(c2t.conv.M_grid*c2t.const.solar_masses_per_gram)

def put_sphere(array, centre, radius, label=1, periodic=True, verbose=False):
	assert array.ndim == 3
	nx, ny, nz = array.shape
	aw  = np.argwhere(np.isfinite(array))
	RR  = ((aw[:,0]-centre[0])**2 + (aw[:,1]-centre[1])**2 + (aw[:,2]-centre[2])**2).reshape(array.shape)
	array[RR<=radius**2] = label
	if periodic: 
		RR2 = ((aw[:,0]+nx-centre[0])**2 + (aw[:,1]+ny-centre[1])**2 + (aw[:,2]+nz-centre[2])**2).reshape(array.shape)
		array[RR2<=radius**2] = label
		if verbose: print "Periodic circle of radius", radius, "made at", centre
	else: 
		if verbose: print "Non-periodic circle of radius", radius, "made at", centre
	return array

def put_circle(array, centre, radius, label=1, periodic=True, verbose=False):
	assert array.ndim == 2
	nx, ny = array.shape
	aw  = np.argwhere(np.isfinite(array))
	RR  = ((aw[:,0]-centre[0])**2 + (aw[:,1]-centre[1])**2).reshape(array.shape)
	array[RR<=radius**2] = label
	if periodic: 
		RR2 = ((aw[:,0]+nx-centre[0])**2 + (aw[:,1]+ny-centre[1])**2).reshape(array.shape)
		array[RR2<=radius**2] = label
		if verbose: print "Periodic circle of radius", radius, "made at", centre
	else: 
		if verbose: print "Non-periodic circle of radius", radius, "made at", centre
	return array
