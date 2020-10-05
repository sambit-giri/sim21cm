import numpy as np 
from sklearn.neighbors import KDTree, NearestNeighbors
import pandas as pd
from time import time

from joblib import Parallel, delayed  
from joblib.externals.loky import set_loky_pickler
from joblib import wrap_non_picklable_objects

from tqdm import tqdm  

class ParticleToGrid:
	"""
	For detail description see Sefusatti et al. (2016; https://arxiv.org/abs/1512.07295).
	This reference also describes the interlacing method, which can reduce the aliasing effect.
	We will include it in the future.
	"""
	def __init__(self, nGrid=100, box_len=256, position=None, scheme='NGP', leaf_size=100, metric='minkowski', periodic=True, n_jobs=1):
		assert scheme in ['NGP', 'CIC', 'TSC', 'PCS']

		self.nGrid = nGrid
		self.box_len = box_len
		self.scheme = scheme
		self.position = position
		self.leaf_size = leaf_size
		self.metric = metric
		self.periodic = periodic
		self.n_jobs = n_jobs
		self.particle_position(leaf_size=self.leaf_size, metric=self.metric, position=position)

	def particle_position(self, position=None, leaf_size=None, metric=None, position_min=None, position_max=None):
		if leaf_size is not None: self.leaf_size = leaf_size
		if metric is not None: self.metric = metric
		if position is None: 
			print('Input particle positions using particle_position().')
			return None
		else: 
			self.position = position

		if isinstance(self.position, (pd.core.frame.DataFrame)):
			X = np.vstack((self.position.x, self.position.y, self.position.z)).T
		else:
			X = self.position

		position_min = X.min() if position_min is None else position_min
		position_max = X.max() if position_max is None else position_max

		X = (X-position_min)/(position_max-position_min)

		print('Building tree...')
		self.tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
		print('Tree built with the positions.')

	def to_grid(self, nGrid=None, box_len=None, scheme=None):
		if scheme is not None: self.scheme = scheme
		if nGrid is not None: self.nGrid = nGrid
		if box_len is not None: self.box_len = box_len

		if self.scheme=='NGP': data_grid = _NGP(self.nGrid, self.tree)
		elif self.scheme=='CIC': data_grid = _CIC(self.nGrid, self.tree, periodic=self.periodic) if self.n_jobs<=1 else _CIC_njobs(self.nGrid, self.tree, periodic=self.periodic, n_jobs=self.n_jobs) 
		elif self.scheme=='TSC': data_grid = _TSC(self.nGrid, self.tree, periodic=self.periodic)
		elif self.scheme=='PCS': data_grid = _PCS(self.nGrid, self.tree, periodic=self.periodic)
		else: 
			print('Choose from the implemented interpolation schemes:')
			print('NGP', 'CIC', 'TSC', 'PCS')

		return data_grid




def _NGP(nGrid, tree):
	data_grid = np.zeros((nGrid,nGrid,nGrid))
	#Xi = np.indices((nGrid,nGrid,nGrid)).reshape(3, nGrid*nGrid*nGrid).T
	#Xq = (Xi+0.5)/nGrid

	#yidx, yr = tree.query_radius(Xq, 1/nGrid/2, return_distance=True)

	#for ii, ri in zip(Xi,yr):
	#	data_grid[ii] = len(ri)

	dGrid = 1/(nGrid+1)
	count, count_tot = 0, np.product(data_grid.shape)
	tstart = time()
	percent_past, percent = 0, 0

	for ii in tqdm(range(data_grid.shape[0])):
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid/2, return_distance=True)
				data_grid[ii,ji,ki] = len(yr[0][yr[0]<dGrid/2]) + 0.5*len(yr[0][yr[0]==dGrid/2])
				# count += 1
				# percent_past, percent = percent, 100*count/count_tot
				# if (percent_past%10)>(percent%10):
				# 	tend = time()
				# 	print('Completed {0:.1f} % in {1:.2f} minutes.'.format(percent,(tend-tstart)/60))

	return data_grid

def _CIC(nGrid, tree, periodic=True):
	data_grid = np.zeros((nGrid,nGrid,nGrid))
	X = tree.get_arrays()[0]
	dGrid = 1/nGrid

	count, count_tot = 0, np.product(data_grid.shape)+1
	tstart = time()
	percent_past, percent = 0, 0

	for ii in tqdm(range(data_grid.shape[0])):
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,ji,ki] = np.sum(1-yr[0]/dGrid)
				count += 1
				percent_past, percent = percent, 100*count/count_tot
				if (percent_past%10)>(percent%10):
					tend = time()
					print('Completed {0:.1f} % in {1:.2f} minutes.'.format(percent,(tend-tstart)/60))

	if periodic:
		print('Estimating contribution from periodicty.')
		## axis=0
		ii = -1
		for ji in tqdm(range(data_grid.shape[1])):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,ji,ki] += np.sum(1-yr[0]/dGrid)
		ii = data_grid.shape[0]
		for ji in tqdm(range(data_grid.shape[1])):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[0,ji,ki] += np.sum(1-yr[0]/dGrid)
		## axis=1
		ji = -1
		for ii in tqdm(range(data_grid.shape[0])):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,ji,ki] += np.sum(1-yr[0]/dGrid)
		ji = data_grid.shape[1]
		for ii in tqdm(range(data_grid.shape[0])):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,0,ki] += np.sum(1-yr[0]/dGrid)
		## axis=2
		ki = -1
		for ii in tqdm(range(data_grid.shape[0])):
			for ji in range(data_grid.shape[1]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,ji,ki] += np.sum(1-yr[0]/dGrid)
		ki = data_grid.shape[2]
		for ii in tqdm(range(data_grid.shape[0])):
			for ji in range(data_grid.shape[1]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,ji,0] += np.sum(1-yr[0]/dGrid)

	tend = time()
	print('Completed 100 % in {0:.2f} minutes.'.format((tend-tstart)/60))

	if not periodic: print('The grid is not periodic.')

	return data_grid

def _CIC_njobs(nGrid, tree, periodic=True, n_jobs=2):
	data_grid = np.zeros((nGrid,nGrid,nGrid))
	X = tree.get_arrays()[0]
	dGrid = 1/nGrid

	count, count_tot = 0, np.product(data_grid.shape)+1
	tstart = time()
	percent_past, percent = 0, 0

	@delayed
	@wrap_non_picklable_objects
	def loop_ijk(ii,ji,ki,tree):
		Xq = np.array([ii,ji,ki]).reshape(1,3)
		yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
		return np.sum(1-yr[0]/dGrid)

	arg_list = np.array([[ii,ji,ki] for ii in tqdm(range(data_grid.shape[0])) for ji in range(data_grid.shape[1]) for ki in range(data_grid.shape[2])])
	out_list = Parallel(n_jobs=n_jobs)(loop_ijk(ii,ji,ki,tree) for ii,ji,ki in tqdm(arg_list))
	# out_list = Parallel(n_jobs=n_jobs)(delayed(loop_ijk)(ii,ji,ki,tree) for ii,ji,ki in tqdm(arg_list))
	data_grid[arg_list[:,0],arg_list[:,1],arg_list[:,2]] = np.array(out_list)

	# for ii in range(data_grid.shape[0]):
	# 	for ji in range(data_grid.shape[1]):
	# 		for ki in range(data_grid.shape[2]):
	# 			Xq = np.array([ii,ji,ki]).reshape(1,3)
	# 			yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
	# 			data_grid[ii,ji,ki] = np.sum(1-yr[0]/dGrid)
				# count += 1
				# percent_past, percent = percent, 100*count/count_tot
				# if (percent_past%10)>(percent%10):
				# 	tend = time()
				# 	print('Completed {0:.1f} % in {1:.2f} minutes.'.format(percent,(tend-tstart)/60))

	if periodic:
		#print('The grid is periodic.')
		## axis=0
		ii = -1
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,ji,ki] += np.sum(1-yr[0]/dGrid)
		ii = data_grid.shape[0]
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[0,ji,ki] += np.sum(1-yr[0]/dGrid)
		## axis=1
		ji = -1
		for ii in range(data_grid.shape[0]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,ji,ki] += np.sum(1-yr[0]/dGrid)
		ji = data_grid.shape[1]
		for ii in range(data_grid.shape[0]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,0,ki] += np.sum(1-yr[0]/dGrid)
		## axis=2
		ki = -1
		for ii in range(data_grid.shape[0]):
			for ji in range(data_grid.shape[1]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,ji,ki] += np.sum(1-yr[0]/dGrid)
		ki = data_grid.shape[2]
		for ii in range(data_grid.shape[0]):
			for ji in range(data_grid.shape[1]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,ji,0] += np.sum(1-yr[0]/dGrid)

	tend = time()
	print('Completed 100 % in {0:.2f} minutes.'.format((tend-tstart)/60))

	if not periodic: print('The grid is not periodic.')

	return data_grid


def _TSC(nGrid, tree, periodic=False):
	#### Not tested or finalised this method.
	data_grid = np.zeros((nGrid,nGrid,nGrid))
	X = tree.get_arrays()[0]
	dGrid = 1/nGrid

	count, count_tot = 0, np.product(data_grid.shape)+1
	tstart = time()
	percent_past, percent = 0, 0

	for ii in range(data_grid.shape[0]):
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 3*dGrid/2, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<0.5]
				s2 = sr[(sr>=0.5)*(sr<1.5)]
				data_grid[ii,ji,ki] = np.sum(3/4-s1**2)+np.sum(0.5*(3/2-s2)**2)
				count += 1
				percent_past, percent = percent, 100*count/count_tot
				if (percent_past%10)>(percent%10):
					tend = time()
					print('Completed {0:.1f} % in {1:.2f} minutes.'.format(percent,(tend-tstart)/60))

	if periodic:
		#print('The grid is period.')
		## axis=0
		ii = -1
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 3*dGrid/2, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<0.5]
				s2 = sr[(sr>=0.5)*(sr<1.5)]
				data_grid[ii,ji,ki] = np.sum(3/4-s1**2)+np.sum(0.5*(3/2-s2)**2)
		ii = data_grid.shape[0]
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 3*dGrid/2, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<0.5]
				s2 = sr[(sr>=0.5)*(sr<1.5)]
				data_grid[0,ji,ki] = np.sum(3/4-s1**2)+np.sum(0.5*(3/2-s2)**2)
		## axis=1
		ji = -1
		for ii in range(data_grid.shape[0]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 3*dGrid/2, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<0.5]
				s2 = sr[(sr>=0.5)*(sr<1.5)]
				data_grid[ii,ji,ki] = np.sum(3/4-s1**2)+np.sum(0.5*(3/2-s2)**2)
		ji = data_grid.shape[1]
		for ii in range(data_grid.shape[0]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 3*dGrid/2, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<0.5]
				s2 = sr[(sr>=0.5)*(sr<1.5)]
				data_grid[ii,0,ki] = np.sum(3/4-s1**2)+np.sum(0.5*(3/2-s2)**2)
		## axis=2
		ki = -1
		for ii in range(data_grid.shape[0]):
			for ji in range(data_grid.shape[1]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 3*dGrid/2, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<0.5]
				s2 = sr[(sr>=0.5)*(sr<1.5)]
				data_grid[ii,ji,ki] = np.sum(3/4-s1**2)+np.sum(0.5*(3/2-s2)**2)
		ki = data_grid.shape[2]
		for ii in range(data_grid.shape[0]):
			for ji in range(data_grid.shape[1]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 3*dGrid/2, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<0.5]
				s2 = sr[(sr>=0.5)*(sr<1.5)]
				data_grid[ii,ji,0] = np.sum(3/4-s1**2)+np.sum(0.5*(3/2-s2)**2)

	tend = time()
	print('Completed 100 % in {0:.2f} minutes.'.format((tend-tstart)/60))

	if not periodic: print('The grid is not periodic.')

	return data_grid


def _PCS(nGrid, tree, periodic=False):
	#### Not tested or finalised this method.
	data_grid = np.zeros((nGrid,nGrid,nGrid))
	X = tree.get_arrays()[0]
	dGrid = 1/nGrid

	count, count_tot = 0, np.product(data_grid.shape)+1
	tstart = time()
	percent_past, percent = 0, 0

	for ii in range(data_grid.shape[0]):
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 2*dGrid, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<1]
				s2 = sr[(sr>=1)*(sr<2)]
				data_grid[ii,ji,ki] = np.sum((1/6)*(4-6*s1**2+3*s1**3))+np.sum((1/6)*(2-s2)**3)
				count += 1
				percent_past, percent = percent, 100*count/count_tot
				if (percent_past%10)>(percent%10):
					tend = time()
					print('Completed {0:.1f} % in {1:.2f} minutes.'.format(percent,(tend-tstart)/60))

	if periodic:
		#print('The grid is period.')
		## axis=0
		ii = -1
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 2*dGrid, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<1]
				s2 = sr[(sr>=1)*(sr<2)]
				data_grid[ii,ji,ki] = np.sum((1/6)*(4-6*s1**2+3*s1**3))+np.sum((1/6)*(2-s2)**3)
		ii = data_grid.shape[0]
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 2*dGrid, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<1]
				s2 = sr[(sr>=1)*(sr<2)]
				data_grid[0,ji,ki] = np.sum((1/6)*(4-6*s1**2+3*s1**3))+np.sum((1/6)*(2-s2)**3)
		## axis=1
		ji = -1
		for ii in range(data_grid.shape[0]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 2*dGrid, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<1]
				s2 = sr[(sr>=1)*(sr<2)]
				data_grid[ii,ji,ki] = np.sum((1/6)*(4-6*s1**2+3*s1**3))+np.sum((1/6)*(2-s2)**3)
		ji = data_grid.shape[1]
		for ii in range(data_grid.shape[0]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 2*dGrid, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<1]
				s2 = sr[(sr>=1)*(sr<2)]
				data_grid[ii,0,ki] = np.sum((1/6)*(4-6*s1**2+3*s1**3))+np.sum((1/6)*(2-s2)**3)
		## axis=2
		ki = -1
		for ii in range(data_grid.shape[0]):
			for ji in range(data_grid.shape[1]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 2*dGrid, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<1]
				s2 = sr[(sr>=1)*(sr<2)]
				data_grid[ii,ji,ki] = np.sum((1/6)*(4-6*s1**2+3*s1**3))+np.sum((1/6)*(2-s2)**3)
		ki = data_grid.shape[2]
		for ii in range(data_grid.shape[0]):
			for ji in range(data_grid.shape[1]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, 2*dGrid, return_distance=True)
				sr = yr[0]/dGrid
				s1 = sr[sr<1]
				s2 = sr[(sr>=1)*(sr<2)]
				data_grid[ii,ji,0] = np.sum((1/6)*(4-6*s1**2+3*s1**3))+np.sum((1/6)*(2-s2)**3)

	tend = time()
	print('Completed 100 % in {0:.2f} minutes.'.format((tend-tstart)/60))

	if not periodic: print('The grid is not periodic.')

	return data_grid
