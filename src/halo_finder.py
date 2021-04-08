import numpy as np 
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import cluster
from astropy import units, constants
from .cosmo_calc import critical_density0

def z2a(z): 
	return 1/(1+z)

def a2z(a): 
	return 1/a-1

def particle_mass(box_len, n_row, cosmo=None, Om=0.315, h=0.673, rho_crit=1.27353443e+11, hunits=False):
	if cosmo is not None:
		Om, h = cosmo.Om0, cosmo.h
		rho_crit = critical_density0(cosmo=cosmo, h=h, hunits=False) # cosmo.critical_density(0).to('solMass/Mpc^3')
	if hunits: box_len = box_len/h
	if type(box_len)!=units.quantity.Quantity: box_len *= units.Mpc
	m1 = Om*rho_crit*(box_len/n_row)**3
	return (m1*h).to('solMass').value if hunits else m1.to('solMass').value


class FoF:
	'''
	Incomplete.
	'''
	def __init__(self, pos_nbody=None, b=0.2, box_len=None, n_p=None, n_jobs=None):
		self.pos_nbody = pos_nbody
		self.box_len   = box_len
		self.n_jobs    = n_jobs
		self.b   = b
		self.n_p = n_p 

		self.particle_position()

	def particle_position(self, pos_nbody=None, box_len=None, n_p=None):
		if pos_nbody is not None: self.pos_nbody = pos_nbody
		if box_len is not None: self.box_len = box_len
		if n_p is not None: self.n_p = n_p
		assert self.pos_nbody is not None and self.box_len is not None

		self.n_p = self.n_p if self.n_p is not None else int(np.cbrt(self.pos_nbody.shape[0]))
		self.D_p = self.box_len/self.n_p

	def find_clusters(self, min_samples=2):
		if self.pos_nbody is None:
			print('Provide the position of particles using particle_position attribute.')
			return None
		clustering = cluster.DBSCAN(eps=self.D_p*self.b, min_samples=min_samples, n_jobs=self.n_jobs)
		clustering.fit(self.pos_nbody)


try:
	from nbodykit.lab import BigFileCatalog
	from nbodykit.lab import FOF
	from nbodykit.lab import HaloCatalog
	from nbodykit.lab import KDDensity
except:
	print('Install nbodykit to use FoF_nbodykit.')


class FoF_nbodykit:
	'''
	Using nbodykit.
	'''
	def __init__(self, filename_nbody=None, b=0.2, box_len=None, n_p=None, n_jobs=None, cosmo=None, N_min=20, zs=None):
		from nbodykit.lab import BigFileCatalog
		from nbodykit.lab import FOF
		from nbodykit.lab import HaloCatalog
		from nbodykit.lab import KDDensity
		if cosmo is None: from nbodykit.cosmology import Planck15 as cosmo

		self.cosmo = cosmo
		self.filename_nbody = filename_nbody if type(filename_nbody) not in [str] else [filename_nbody]
		self.box_len   = box_len
		self.n_jobs = n_jobs
		self.N_min  = N_min
		self.b   = b
		self.n_p = n_p
		self.zs  = zs if type(zs) not in [int, float] else [zs]

	def find_haloes_epoch(self, filename, z, N_min=None, with_peak=True):
		if N_min is not None: self.N_min = N_min

		cat = BigFileCatalog(filename, header='Header', dataset='1/') 
		self.cosmo = self.cosmo.match(Omega0_m=cat.attrs['Om0'])
		if self.n_p is None: self.n_p = cat.attrs['nc'][0]

		cat.attrs['boxsize'] = np.ones(3) * cat.attrs['boxsize'][0]
		cat.attrs['Nmesh']   = np.ones(3) * cat.attrs['nc'][0]

		# M0 = cat.attrs['Om0'][0] * 27.75 * 1e10 * cat.attrs['boxsize'].prod() / cat.csize
		M0 = cat.attrs['Om0'][0] * critical_density0(cosmo=self.cosmo, hunits=True).value * cat.attrs['boxsize'].prod() / cat.csize
		self.M_p = M0
		cat.attrs['BoxSize'] = cat.attrs['boxsize']

		print('BoxSize', cat.attrs['boxsize'])
		print('Nmesh', cat.attrs['Nmesh'])
		print('Mass of a particle', M0/1e8, 'x1e8 solar mass.')
		print('OmegaM', cosmo.Om0)

		print('Finding haloes...')
		cat['Density'] = KDDensity(cat).density
		fof = FOF(cat, linking_length=self.b, nmin=self.N_min)
		features = fof.find_features(peakcolumn='Density') if with_peak else fof.find_features(peakcolumn=None)

		halo_catalog = fof.to_halos(M0, self.cosmo, z)
		print(halo_catalog)
		features['Mass'] = M0 * features['Length']

		print('...done')
		print('Total number of haloes found', halo_catalog.csize)
		# print('Saving columns', features.columns)
		# print('halos', halo_catalog.columns)

		return halo_catalog

	def find_haloes(self, N_min=None, with_peak=True):
		assert len(self.filename_nbody) == len(self.zs)
		self.halo_catalogs = {}
		for i in range(len(self.zs)):
			halo_cat = self.find_haloes_epoch(self.filename_nbody[i], self.zs[i], N_min=N_min, with_peak=with_peak)
			self.halo_catalogs['{:3f}'.format(self.zs[i])] = halo_cat



