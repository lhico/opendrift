'''
3D reader to ECOM model at South Brazilian Bight
Based on ROMS_native_reader
Developed by Arian Dialectaquiz Santos and Danilo Silva from LHiCo - IO -USP (Brazil)
'''
from bisect import bisect_left, bisect_right
from datetime import datetime
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset, MFDataset, num2date
from scipy.ndimage import map_coordinates
from multiprocessing import Process, Manager, cpu_count

try:
	import xarray as xr
	has_xarray = True
except:
	has_xarray = False

import logging
logger = logging.getLogger(__name__)

from netCDF4 import num2date
import xarray as xr
from opendrift.readers.basereader import BaseReader, vector_pairs_xy, StructuredReader

from opendrift.readers.ECOM_dependencies import depth_ECOM, work_model_grid


###############################################################################

class Reader(BaseReader,StructuredReader):

	def __init__(self, filename=None, name=None, gridfile=None):

		if filename is None:
			raise ValueError('Need filename as argument to constructor')

		# Map ECOM variable names to CF standard_name
		self.ECOM_variable_mapping = {
			'FSM':'land_binary_mask', 
			'time': 'time',
			'sigma':'ocean_sigma_coordinate',
			'depth':'sea_floor_depth_below_sea_level',
			'elev':'sea_surface_height_above_sea_level',
			'wu': 'x_wind',
			'wv':'y_wind',
			'patm':'air_pressure_at_sea_level',
			'u': 'x_sea_water_velocity',
			'v':'y_sea_water_velocity',
			'w':'upward_sea_water_velocity',
			'salt':'sea_water_salinity',
			'temp':'sea_water_temperature',
			'xgrid':'xgrid',
			'xgrid':'ygrid',
			'DUM':'U1_direction_mask',
			'DVM':'V1_direction_mask',
			'cbc':'Bottom_Drag_Coefficient',
			'lon':'lon',
			'lat':'lat',
			'ang': 'ang'}


		filestr = str(filename)

		if name is None:
			self.name = filestr
		else:
			self.name = name

		try:

			 # Open file, check that everything is ok
			logger.info('Opening dataset: ' + filestr)
			if ('*' in filestr) or ('?' in filestr) or ('[' in filestr):
				logger.info('Opening files with MFDataset')
				def drop_non_essential_vars_pop(ds):
					dropvars = [v for v in ds.variables if v not in
								list(self.ECOM_variable_mapping.keys())]
					logger.debug('Dropping variables: %s' % dropvars)
					ds = ds.drop(dropvars)
					return ds

				if has_xarray is True:
					self.Dataset = xr.open_mfdataset(filename,
						chunks={'time': 1}, concat_dim='time',
						preprocess=drop_non_essential_vars_pop,
						data_vars='minimal', coords='minimal')
				else:
					self.Dataset = MFDataset(filename)
			else:
				#logger.info('Opening file with Dataset')
				if has_xarray is True:
					#Inserted function that work the zeros values in ECOM outputs of lat and lon
					self.Dataset_1 = xr.open_dataset(filename)
					print("dataset ==", self.Dataset_1)

					if 'xpos' in self.Dataset_1:
						remap_names_1 = {'x': 'corn_lon', 'y': 'corn_lat'}							
						self.Dataset_2 = self.Dataset_1.rename(remap_names_1)
						remap_names_2 = {'xpos': 'x', 'ypos': 'y'}
						self.Dataset_3 = self.Dataset_2.rename(remap_names_2)
						self.x = self.Dataset_3.variables['x']

						if len(self.x) < 100:
							self.Dataset = work_model_grid.fix_ds_other(self.Dataset_3.copy())
							self.zlevels = np.array([0.0000,-5.000,-10.0000,-15.0000,-20.0000])
							print("Using Cananeia grid in spite of SBB")
						elif len(self.x) > 290:
							self.Dataset = work_model_grid.fix_ds_other_2(self.Dataset_3.copy())
							self.zlevels = np.array([0.0000,-5.000,-10.0000,-15.0000,-20.0000])
							print("Using SESSVB grid in spite of SBB or Cananeia")

						else:
							self.Dataset = work_model_grid.fix_ds(self.Dataset_3.copy())
							self.zlevels = np.array([0, -5, -10, -15, -25,-30, -50, -75, -100, -150, -200,-250, -300, -400, -500, -600, -700, -800, -900, -1000, -1500,-2000])
							print("Using SBB grid")

					#if x is the zonal coordinate (new version from ECOM)	
					else:

						if len(self.x) < 100:
							self.Dataset = work_model_grid.fix_ds_other(self.Dataset_3.copy())
							self.zlevels = np.array([0.0000,-5.000,-10.0000,-15.0000,-20.0000])
							print("Using Cananeia grid in spite of SBB")
						elif len(self.x) > 290:
							self.Dataset = work_model_grid.fix_ds_other_2(self.Dataset_3.copy())
							self.zlevels = np.array([0.0000,-5.000,-10.0000,-15.0000,-20.0000])
							print("Using SESSVB grid in spite of SBB or Cananeia")

						else:
							self.Dataset = work_model_grid.fix_ds(self.Dataset_3.copy())
							self.zlevels = np.array([0, -5, -10, -15, -25,-30, -50, -75, -100, -150, -200,-250, -300, -400, -500, -600, -700, -800, -900, -1000, -1500,-2000])
							print("Using SBB grid")
					
				else:
					self.Dataset = Dataset(filename, 'r')
		except Exception as e:
			raise ValueError('e')


		if 'sigma' not in self.Dataset.variables:
			dimensions = 2
		else:
			dimensions = 3

		if dimensions == 3:
			# Read sigma-coordinate values
			try:
				self.sigma = self.Dataset.variables['sigma'][:]
				self.sigma = np.nan_to_num(self.sigma)
			except:
				num_sigma = len(self.Dataset.dimensions['sigma'])
				logger.warning(
					'sigma not available in dataset, constructing from'
					' number of layers (%s).' % num_sigma)
				self.sigma = (np.arange(num_sigma)+.5-num_sigma)/num_sigma

				# Read sigma-coordinate transform parameters
			try:
				self.Dataset.variables['sigma'].set_auto_mask(False)
			except:
				pass
			self.sigma = self.Dataset.variables['sigma'][:]
			try:
				self.depth = self.Dataset.variables['depth'][:]

			except:
				if has_xarray is True:
					self.depth = self.Dataset.variables['depth'].data  # scalar
				else:
					self.depth = self.Dataset.variables['depth'][0]

			self.num_layers = len(self.sigma)

		else:
			self.num_layers = 1
			self.ECOM_variable_mapping['u'] = 'x_sea_water_velocity'
			self.ECOM_variable_mapping['v'] = 'y_sea_water_velocity' 

			self.depth = self.Dataset.variables['depth'][:]

		if 'lat' in self.Dataset.variables:

			# Horizontal coordinates and directions
			self.lat = self.Dataset.variables['lat']
			print("LAT==",self.lat)
			print("Shape of lat==",self.lat.shape)

			self.lon = self.Dataset.variables['lon']
			print("LON==",self.lon)
			print("Shape of lon==",self.lon.shape)


		else:
			if gridfile is None:
				raise ValueError(filename + ' does not contain lon/lat '
								 'arrays, please supply a grid-file '
								 '"gridfile=<grid_file>"')

			else:
				gf = Dataset(gridfile)
				self.lat = gf.variables['lat'][:]
				self.lon = gf.variables['lon'][:]

		# Get time coverage

		ocean_time = self.Dataset.variables['time']
		if has_xarray:
			self.times = [datetime.utcfromtimestamp((OT -
						  np.datetime64('1970-01-01T00:00:00Z')
							) / np.timedelta64(1, 's'))
						  for OT in ocean_time.data]
		else:
			time_units = ocean_time.__dict__['units']
			if time_units == 'second':
				logger.info('Ocean time given as seconds relative to start '
							 'Setting artifical start time of 1 Jan 2000.')
				time_units = 'seconds since 2000-01-01 00:00:00'
			self.times = num2date(ocean_time[:], time_units)
		self.start_time = self.times[0]
		self.end_time = self.times[-1]
		if len(self.times) > 1:
			self.time_step = self.times[1] - self.times[0]
		else:
			self.time_step = None
		# x and y are rows and columns for unprojected datasets

	
		self.lon = self.lon.data  
		self.lat = self.lat.data
		self.sigma = self.sigma.data

		self.name = 'ECOM'

		self.precalculate_s2z_coefficients = True

		# Find all variables having standard_name
		self.variables = []
		for var_name in self.Dataset.variables:
			if var_name in self.ECOM_variable_mapping.keys():
				var = self.Dataset.variables[var_name]
				self.variables.append(self.ECOM_variable_mapping[var_name])
		print("variables with standard name:", self.variables)
		# Run constructor of parent Reader class


		super(Reader, self).__init__()


	def get_variables(self, requested_variables, time=None,
					  x=None, y=None, z=None):

		start_time = datetime.now()
		requested_variables, time, x, y, z, outside = self.check_arguments(
			requested_variables, time, x, y, z)

		# If one vector component is requested, but not the other
		# we must add the other for correct rotation
		for vector_pair in vector_pairs_xy:
			if (vector_pair[0] in requested_variables and
				vector_pair[1] not in requested_variables):
				requested_variables.extend([vector_pair[1]])
			if (vector_pair[1] in requested_variables and
				vector_pair[0] not in requested_variables):
				requested_variables.extend([vector_pair[0]])

		nearestTime, dummy1, dummy2, indxTime, dummy3, dummy4 = \
			self.nearest_time(time)

		variables = {}   
		
		######-----Find horizontal indices corresponding to requested x and y----#####################

		if hasattr(self, 'clipped'):
			clipped = self.clipped
		else: clipped = 0
		indx = np.floor((x-self.xmin)/self.delta_x).astype(int) + clipped 
		indy = np.floor((y-self.ymin)/self.delta_y).astype(int) + clipped 
		print ("indx_init ==", indx)
		print ("indy_init ==", indy)

		buffer = self.buffer
		indx = np.arange(np.max([0, indx.min()-buffer]),
							np.min([indx.max()+buffer, self.lon.shape[1]]))
		indy = np.arange(np.max([0, indy.min()-buffer]),
							np.min([indy.max()+buffer, self.lon.shape[0]]))

	
		print("Buffer ==",buffer)
		print ("indx ==", indx)
		print ("indy ==", indy)
		print ("indx_min ==", indx.min())
		print ("indy_min ==", indy.min())
		print ("indx_max ==", indx.max())
		print ("indy_max ==", indy.max())


	
###########----Working with z:-------##################################
		#self.depth = np.where(self.depth < 0,0,self.depth)
		#print("Depth at given point==", self.depth[indy,indx])

		def find_nearest(array, value):
			array = np.asarray(array)

			idx = (np.abs(array - value)).argmin()

			return idx

	
		if not hasattr(self, 'z') and (z is not None):  #if z is not in netcdf, but is requested: 
			print ("Z has been requested. Searching the nearest value inside z range ... ", z)
	
			dz = self.zlevels
			indz_i = np.asarray([find_nearest(dz, value) for value in z])
			print("indz_i ==", indz_i)
		
		
			if (indz_i.max() + self.verticalbuffer) == 1:
				indz = range(0,2)
			else:
				indz = range(np.maximum(0, indz_i.min()-self.verticalbuffer),
							np.minimum(self.num_layers,
									indz_i.max() + self.verticalbuffer))	
						
			print ("indz ==", indz)
			print("len de indz ==", len(indz))
		
			zi1_aux = np.maximum(0, bisect_left(-np.array(self.zlevels),-z.max()))
			zi2_aux = np.minimum(len(self.zlevels),bisect_right(-np.array(self.zlevels),-z.min()))
	
			if zi1_aux == zi2_aux:
				print("Using auxiliar zi1 and zi2")
				if zi1_aux == 0:
					zi1 = 0
					zi2 = zi2_aux +1
				else:
					zi1 = zi1_aux - 1
					zi2 = zi2_aux + 1
	
			else:
				zi1 = zi1_aux
				zi2 = zi2_aux
		
	
			print("Z1 ==", zi1)
			print("Z2 ==", zi2)
	
			variables['z'] = np.array(self.zlevels[zi1:(zi2)])
	
			if len(variables['z']) == 1:
				variables['z'] = np.array(self.zlevels[zi1:(zi2 +1)])

		else:
			print("Z has not been requested. Setting it as 0")
			indz = range(0,2)
			variables['z'] = np.array(self.zlevels[0:2])

		print("VAR_Z ==", variables['z'])

		#Defining and retrieveing angle of rotation between x and east
		self.angle_xi_east = self.Dataset.variables['ang'][:]
		print("Shape of angle from east to x:", self.angle_xi_east.shape)
		rad = self.angle_xi_east[indy,indx]
		print("Shape of rad:", rad.shape)
		
##################################################################################################################################
#########################---This must be done to work with sigma ----###############################################
##################################################################################################################################

		# Find depth levels covering all elements 
		sigma = np.asarray(self.sigma)
		H_ND = np.asarray(self.depth)

		layers = self.num_layers - 1  # surface layer
		H_shape = H_ND.shape      # Save the shape of H
		H = H_ND.ravel()        # and make H 1D for easy shape maniplation
		L_C = len(sigma)
		outshape = (L_C,) + H_shape
		S = -1.0 + (0.5+np.arange(L_C))/L_C 
		A = (S - sigma)[:, None]
		B = np.outer(sigma, H)
		self.z_rho_tot = (A + B).reshape(outshape)  

##################################################################################################################################
		mask_values = {}
		for par in requested_variables:
			varname = [name for name, cf in
					   self.ECOM_variable_mapping.items() if cf == par]
			var = self.Dataset.variables[varname[0]]

			if par == 'land_binary_mask':
				print("Using land_binary_mask from the Eulerian Model!!")

				self.land_binary_mask = np.absolute(1 - self.Dataset.variables['FSM']) #for some reason the values are inverted, so, lets change ir
				np.where(self.land_binary_mask < 1, self.land_binary_mask,0)
				print("land_binary_mask ==", self.land_binary_mask)
				print("land binary mask SHAPE:", self.land_binary_mask.shape)

				variables[par] = self.land_binary_mask[indy,indx]

				print("PAR SHAPE land_binary_mask ==", variables[par].shape)
				print("PAR NDIM land_binary_mask ==", variables[par].ndim)


			elif var.ndim == 2:
				variables[par] = var[indy, indx]

			elif var.ndim == 3:
				print("Retrieving 3D variables")
				variables[par] = var[indxTime, indy, indx]

			elif var.ndim == 4:
				print("Retrieving 4D variables")
				variables[par] = var[indxTime, indz, indy, indx]

			else:
				raise Exception('Wrong dimension of variable: ' +
								self.variable_mapping[par])

			variables[par] = np.asarray(variables[par]) 
			start = datetime.now()

			if par not in mask_values:
				indxgrid = indx
				indygrid = indy
				if par == 'x_sea_water_velocity':
					if not hasattr(self, 'DUM'):
						if 'DUM' in self.Dataset.variables:
							self.mask_u = self.Dataset.variables['DUM'][:]
						else:
							self.mask_u = self.Dataset.variables['FSM'][:]
					mask = self.mask_u[indygrid, indxgrid]
				elif par == 'y_sea_water_velocity':
					if not hasattr(self, 'DVM'):
						if 'DVM' in self.Dataset.variables:
							self.mask_v = self.Dataset.variables['DVM'][:]
						else:
							self.mask_v = self.Dataset.variables['FSM'][:]
					mask = self.mask_v[indygrid, indxgrid]
				else:
					if not hasattr(self, 'FSM'):
						self.mask_rho = self.Dataset.variables['FSM'][:]
					mask = self.mask_rho[indygrid, indxgrid]
				mask = np.asarray(mask)
				if mask.min() == 0 and par != 'land_binary_mask':
					first_mask_point = np.where(mask.ravel()==0)[0][0]
					if variables[par].ndim == 3:
						upper = variables[par][0,:,:]
					else:
						upper = variables[par]
					mask_values[par] = upper.ravel()[first_mask_point]
					variables[par][variables[par]==mask_values[par]] = np.nan

##################################################################################################################################
#########################---Converting  sigma variables to Z variables----########################################################
##################################################################################################################################
			
			if var.ndim == 4:

				variables[par] = var[indxTime, indz, indy, indx]
				# Regrid from sigma to z levels
				if len(np.atleast_1d(indz)) >= 1:
					logger.debug('sigma to z for ' + varname[0])
					if self.precalculate_s2z_coefficients is True: 

						y_depth = self.Dataset.variables['depth'].shape[0] #Positions of depth by Y
						x_depth = self.Dataset.variables['depth'].shape[1] #Positions of depth by X
						L_Y = y_depth #len(y_depth)
						L_X = x_depth #len(x_depth) 
						L_Z = len(self.z_rho_tot)

						logger.debug('Calculating sigma2z-coefficients for whole domain')
						starttime = datetime.now()
						dummyvar = np.ones((L_Z, L_Y, L_X))
						dummy, self.s2z_total = depth_ECOM.multi_zslice(dummyvar, self.z_rho_tot, self.zlevels)
						# Store arrays/coefficients
						self.s2z_A = self.s2z_total[0].reshape(len(self.zlevels), L_Y, L_X)
						self.s2z_C = self.s2z_total[1].reshape(len(self.zlevels), L_Y, L_X)
						self.s2z_I = self.s2z_total[2].reshape(L_Y, L_X)
						self.s2z_kmax = self.s2z_total[3]
						del self.s2z_total  # Free memory
						logger.info('Time: ' + str(datetime.now() - starttime))	
						if zi2 == 1:
							zle = np.arange(zi1, zi2+1)
						else:
							zle = np.arange(zi1, zi2)
						
						#print("zle==",zle)
						A = self.s2z_A.copy()  # Awkward subsetting to prevent losing one dimension
						A = A[:,:,indx]
						A = A[:,indy,:]
						A = A[zle,:,:]
						C = self.s2z_C.copy()
						C = C[:,:,indx]
						C = C[:,indy,:]
						C = C[zle,:,:]
						#print("C ==", C)
						C = C - C.max() + variables[par].shape[0] -1
						C[C<1] = 1
						A = A.reshape(len(zle), len(indx)*len(indy))
						#print("A ==",A)
						#print("A.shape==",A.shape)
						C = C.reshape(len(zle), len(indx)*len(indy))
						#print("C ==",C)
						#print("C.shape==",C.shape)
						I = np.arange(len(indx)*len(indy))		
						#print("I ==",I)
						#print("I.shape==",I.shape)					
						kmax = len(zle) 
				
					logger.debug('Applying sigma2z-coefficients')
	
					F = np.asarray(variables[par])

					#print("F ==", F)
					Fshape = F.shape
					#print("Fshape ==", Fshape)
					N = F.shape[0]
					#print("N ==", N)
					M = F.size // N
					#print("M ==", M)
					F_r = F.reshape((N, M))
					#print("F_r ==", F_r)

					R = (1-A)*F_r[(C-1, I)]+A*F_r[(C, I)]
					#print("R ==", R)
					#print("R.shape", R.shape)

					variables[par] = R.reshape((kmax,) + Fshape[1:])
					#variables[par]_s = R_s.reshape((kmax,) + F_sshape[1:])

					# Nan in input to multi_zslice gives extreme values in output
					variables[par][variables[par]>1e+9] = np.nan
					#print("Type var_par ==", type(variables[par]))
			

			if len(indz)<=2:
		
				if variables[par].ndim > 1:
					variables[par] = variables[par].diagonal()
					print("Var ndim after diag==", variables[par].ndim)
			
					
			else:
				variables[par] = variables[par]
			

			# Mask values outside domain			
			variables[par] = np.ma.array(variables[par], ndmin=2, mask=False)



##################################################################################################################################

		variables['x'] = indx 
		variables['y'] = indy 

		variables['x'] = variables['x'].astype(np.float)
		variables['y'] = variables['y'].astype(np.float)
		variables['time'] = nearestTime

		if 'land_binary_mask' in requested_variables:
			print("Using land_binary_mask from ECOM output")

			variables['land_binary_mask'] = self.land_binary_mask[indy,indx]
			
			print("Final land_binary_mask:", variables['land_binary_mask'])
			print("Type of final land_binary_mask:", type(variables['land_binary_mask']))
			print("Shape of final land_binary_mask:", variables['land_binary_mask'].shape)

		if 'x_sea_water_velocity' or 'x_wind' in variables.keys():
			print("Retrieving velocities")
  
			if 'x_sea_water_velocity' in variables.keys():


				variables['x_sea_water_velocity'] = np.nan_to_num(variables['x_sea_water_velocity'])
				variables['y_sea_water_velocity'] = np.nan_to_num(variables['y_sea_water_velocity'])

			if 'upward_sea_water_velocity' in variables.keys():
				variables['upward_sea_water_velocity'] = np.nan_to_num(variables['upward_sea_water_velocity'])

			if 'x_wind' in variables.keys():


				variables['x_wind'] = np.nan_to_num(variables['x_wind'])
				variables['x_wind'][variables['x_wind']>50] = 0
				variables['x_wind'][variables['x_wind']<50] = 0
				variables['y_wind'] = np.nan_to_num(variables['y_wind'])
				variables['y_wind'][variables['y_wind']>50] = 0
				variables['y_wind'][variables['y_wind']<50] = 0

		# Masking NaN of the others variables, considering u and v always requested
		for var in requested_variables:
			variables[var] = np.nan_to_num(variables[var])
			print("Now return the variables, please.")
		logger.debug('Time for ECOM reader: ' + str(datetime.now()-start_time))

		return variables

#This follow function is not to be used at the SBB grid.

def rotate_vectors_angle(x_sea_water_velocity, y_sea_water_velocity, radians):
	x_sea_water_velocity2 = x_sea_water_velocity*np.cos(radians) - y_sea_water_velocity*np.sin(radians)
	y_sea_water_velocity2 = x_sea_water_velocity*np.sin(radians) + y_sea_water_velocity*np.cos(radians)
	return x_sea_water_velocity2, y_sea_water_velocity2
