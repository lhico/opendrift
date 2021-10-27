'''
Code for work with ECOM model grid.

This code sets all varaibles into the grid with land points, and after thtat remove all land points of
all variables, remaining the wet points

Select your path from the file 'model_grid_withLandPoints'
'''

import os
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset, MFDataset, num2date
import xarray as xr

def load(grid, nrows, verbose=False):

	"""grid: model_grid WITH LAND POINTS file name including path
		nrows: number of rows on model_grid file to skip (line number where II and JJ indexes are written)
		
	author: written by Carine    
	"""
	if verbose:
		print(' loading model_grid file')

	model_grid = np.loadtxt(grid, skiprows=nrows)
	line = open(grid, 'r').readlines()[nrows-1]

	# get data
	II = int(line[0:5])
	JJ = int(line[6:10])

	# initialize variables to NAN
	H1 = np.zeros((JJ-2, II-2))*np.NAN
	H2 = np.zeros((JJ-2, II-2))*np.NAN
	depgrid = np.zeros((JJ-2, II-2))*np.NAN
	ANG = np.zeros((JJ-2, II-2))*np.NAN
	Ygrid = np.zeros((JJ-2, II-2))*np.NAN
	Xgrid = np.zeros((JJ-2, II-2))*np.NAN
	fsm = np.zeros((JJ-2, II-2))
	# keep loop in case some grid points are not in the model_grid file (corners, for instance)
	for n in np.arange(0, len(model_grid)):
		I = int(model_grid[n, 0])
		J = int(model_grid[n, 1])
		H1[J-2, I-2] = model_grid[n, 2]
		H2[J-2, I-2] = model_grid[n, 3]
		depgrid[J-2, I-2] = model_grid[n, 4]
		ANG[J-2, I-2] = model_grid[n, 5]
		Ygrid[J-2, I-2] = model_grid[n, 6]
		Xgrid[J-2, I-2] = model_grid[n, 7]

	fsm[np.where(depgrid <0)] = 1

	return II,JJ,H1,H2,depgrid,ANG,Xgrid,Ygrid,fsm

############################################################################################


# read model_grid and creates a new xr.Dataset
def create_grid_from_modelgrid(fname, nrows):
	II,JJ,H1,H2,depgrid,ANG,Xgrid,Ygrid,fsm = load(fname, nrows)

	ds = xr.Dataset(coords={'y': np.arange(2, JJ, 1),
							'x': np.arange(2, II, 1)})

	ds['lon'] = (('y','x'), Xgrid)

	ds['lat'] = (('y','x'), Ygrid)

	ds['depth'] = (('y','x'),depgrid)

	#creating fsm mask based on -999 values at depgrid

	ds['FSM'] = (('y','x'),fsm)
	print("FSM created based on depth values form grid:", ds['FSM'])
	print("FSM min and max:", ds['FSM'].min(), ds['FSM'].max())
	return ds


def fix_ds(ds):
	#Apply the new coordinates to the others variables from original ecom netcdf
	cdir = os.path.dirname(__file__)
	model_grid = f"{cdir}/model_grid"
		

	ds_new = create_grid_from_modelgrid(model_grid,26)
	ds = ds.isel(y=slice(1,-1), x=slice(1,-1))

	# replacing by the model_grid domain
	ds['lon'] = ds_new['lon']
	ds['lat'] = ds_new['lat']
	ds['FSM'] = ds_new['FSM']


	return ds



def fix_ds_other(ds):
	#Apply the new coordinates to the others variables from original ecom netcdf
	cdir = os.path.dirname(__file__)
	model_grid = f"{cdir}/model_grid_withLandPoints_Cananeia_v2"

	ds_new = create_grid_from_modelgrid(model_grid,16)
	ds = ds.isel(y=slice(1,-1), x=slice(1,-1))

	# replacing by the model_grid domain
	ds['lon'] = ds_new['lon']
	ds['lat'] = ds_new['lat']
	ds['depth'] = ds_new['depth']
	ds['FSM'] = ds_new['FSM']
	print("FSM Cananeia at river", ds['FSM'][309:314,0])

	return ds



def fix_ds_other_2(ds):
	#Apply the new coordinates to the others variables from original ecom netcdf
	cdir = os.path.dirname(__file__)
	model_grid = f"{cdir}/model_grid_SSVBES_withLandPoints_999"

	ds_new = create_grid_from_modelgrid(model_grid,16)
	ds = ds.isel(y=slice(1,-1), x=slice(1,-1))

	# replacing by the model_grid domain
	ds['lon'] = ds_new['lon']
	ds['lat'] = ds_new['lat']
	ds['FSM'] = ds_new['FSM']

	return ds