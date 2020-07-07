import numpy as np 
from sklearn.neighbors import KDTree, NearestNeighbors
import pandas as pd

class ParticleToGrid:
	def __init__(self, nGrid=100, box_len=256, position=None, scheme='NGP', leaf_size=100, metric='minkowski'):
		assert scheme in ['NGP', 'CIC', 'spline']

		self.nGrid = nGrid
		self.box_len = box_len
		self.scheme = scheme
		self.position = position
		self.leaf_size = leaf_size
		self.metric = metric
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

	def to_grid(self, nGrid=None, box_len=None, scheme=None):
		if scheme is not None: self.scheme = scheme
		if nGrid is not None: self.nGrid = nGrid
		if box_len is not None: self.box_len = box_len

		if self.scheme=='NGP': data_grid = _NGP(self.nGrid, self.tree)
		elif self.scheme=='CIC': data_grid = _CIC(self.nGrid, self.tree)

		return data_grid




def _NGP(nGrid, tree):
	data_grid = np.zeros((nGrid,nGrid,nGrid))
	#Xi = np.indices((nGrid,nGrid,nGrid)).reshape(3, nGrid*nGrid*nGrid).T
	#Xq = (Xi+0.5)/nGrid

	#yidx, yr = tree.query_radius(Xq, 1/nGrid/2, return_distance=True)

	#for ii, ri in zip(Xi,yr):
	#	data_grid[ii] = len(ri)

	dGrid = 1/(nGrid+1)

	for ii in range(data_grid.shape[0]):
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid/2, return_distance=True)
				data_grid[ii,ji,ki] = len(yr[0][yr[0]<dGrid/2]) + 0.5*len(yr[0][yr[0]==dGrid/2])

	return data_grid

def _CIC(nGrid, tree, periodic=True):
	data_grid = np.zeros((nGrid,nGrid,nGrid))
	X = tree.get_arrays()[0]
	dGrid = 1/nGrid

	for ii in range(data_grid.shape[0]):
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx, yr = tree.query_radius(Xq*dGrid+dGrid/2, dGrid, return_distance=True)
				data_grid[ii,ji,ki] = np.sum(1-yr[0]/dGrid)

	if periodic:
		print('The grid is period.')
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

	return data_grid


def _TSC(nGrid, tree, periodic=False):
	#### Not tested or finalised this method.
	data_grid = np.zeros((nGrid,nGrid,nGrid))
	X = tree.get_arrays()[0]
	dGrid = 1/(nGrid+1)

	for ii in range(data_grid.shape[0]):
		for ji in range(data_grid.shape[1]):
			for ki in range(data_grid.shape[2]):
				Xq = np.array([ii,ji,ki]).reshape(1,3)
				yidx1, yr1 = tree.query_radius(Xq*dGrid+dGrid/2, dGrid/2, return_distance=True)
				yidx2, yr2 = tree.query_radius(Xq*dGrid+dGrid/2, 3*dGrid/2, return_distance=True)
				data_grid[ii,ji,ki] = np.sum(3/4-(yr1[0]/dGrid)**2)+0.5*np.sum((3/2-yr1[0][yidx2[0]!=yidx1[0]]/dGrid)**2)

	if periodic:
		print('The grid is period.')
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

	return data_grid
