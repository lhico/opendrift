import xarray as xr
import os



def fix_angle(ds):
	#Apply the new coordinates to the others variables from original ecom netcdf
	cdir = os.path.dirname(__file__)
	grid = f"{cdir}/roms01c_grd.nc"
	angle = xr.open_dataset(grid).angle

	ds['angle'] = angle	

	return ds