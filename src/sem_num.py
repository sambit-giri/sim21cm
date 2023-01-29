import numpy as np
import tools21cm as t2c
from . import usefuls

def sem_num(dens_cube, sourcelist, z, Nion=30, Nrec=0, Rmfp=10, boxsize=None):
	"""
	@Majumdar et al. (2014)
	Rmfp: Default is 10 Mpc (proper) from the observations.
	"""
	if boxsize is None: boxsize = t2c.conv.LB
	Mhalos   = usefuls.Mgrid_2_Msolar(sourcelist[:,3])
	xx,yy,zz = (sourcelist[:,:3].T-1).astype(dtype=np.int)
	N_h = np.zeros(dens_cube.shape)
	N_h[xx,yy,zz] = Nion*Mhalos*t2c.const.OmegaB/t2c.const.Omega0/t2c.const.m_p/t2c.const.solar_masses_per_gram
	nn  = dens_cube.shape[0]
	n_h = N_h*(nn/boxsize)**3
	n_H = t2c.const.Mpc**3*dens_cube/t2c.const.m_p
	G_mfp = Rmfp*(1.+z)*nn/boxsize
	Rs = np.arange(G_mfp)+1.
	xf = np.zeros(dens_cube.shape)
	for i in xrange(Rs.size):
		ra     = Rs[i]
		kernel = usefuls.put_circle(np.zeros((nn,nn)), [nn/2,nn/2], ra, label=1)
		nh_    = smooth_with_kernel_3d(n_h, kernel)
		nH_    = smooth_with_kernel_3d(n_H, kernel)
		xf[nh_>=nH_*(1+Nrec)] = 1
		print (i+1)*100/Rs.size, "% completed"
	xf[nh_<nH_*(1+Nrec)] = (nh_/nH_)[nh_<nH_*(1+Nrec)]
	return xf

def smooth_with_kernel_3d(array, kernel):
	assert array.ndim==3 and kernel.ndim==2
	out = np.zeros(array.shape)
	for i in range(array.shape[0]): out[i,:,:] = t2c.smooth_with_kernel(array[i,:,:], kernel)
	for j in range(array.shape[1]): out[:,j,:] = t2c.smooth_with_kernel(out[:,j,:], kernel)
	return out


	

