# -*- coding: utf-8 -*-

"""Vertical structure functions for ROMS

:func:`sdepth`
  Depth of s-levels
:func:`zslice`
  Slice a 3D field in s-coordinates to fixed depth
:func:`multi_zslice`
  Slice a 3D field to several depth levels
:func:`z_average`
  Vertical average of a 3D field
:func:`s_stretch`
  Compute vertical stretching arrays Cs_r or Cs_w

"""

# -----------------------------------
#Addapted from depth to roms by Arian Dialectaquiz - LHiCo - IO USP
# -----------------------------------

from __future__ import (absolute_import, division)

import numpy as np
import datetime as datetime

def sdepth(H, C, stagger="rho", Hc=300, Vtransform=1):
	"""Depth of s-levels

	*H* : arraylike
	  Bottom depths [meter, positive]

	*Hc* : scalar
	   Critical depth

	*cs_r* : 1D array
	   s-level stretching curve

	*stagger* : [ 'rho' | 'w' ]

	*Vtransform* : [ 1 | 2 ]
	   defines the transform used, defaults 1 = Song-Haidvogel

	Returns an array with ndim = H.ndim + 1 and
	shape = cs_r.shape + H.shape with the depths of the
	mid-points in the s-levels.

	Typical usage::

	>>> fid = Dataset(roms_file)
	>>> H = fid.variables['h'][:, :]
	>>> C = fid.variables['Cs_r'][:]
	>>> Hc = fid.variables['hc'].getValue()
	>>> z_rho = sdepth(H, Hc, C)

	"""
	H = np.asarray(H)
	Hshape = H.shape      # Save the shape of H
	H = H.ravel()         # and make H 1D for easy shape maniplation
	C = np.asarray(C)
	N = len(C)
	outshape = (N,) + Hshape       # Shape of output
	if stagger == 'rho':
		S = -1.0 + (0.5+np.arange(N))/N    # Unstretched coordinates
	elif stagger == 'w':
		S = np.linspace(-1.0, 0.0, N)
	else:
		raise ValueError("stagger must be 'rho' or 'w'")

	if Vtransform == 1:         # Default transform by Song and Haidvogel
		A = Hc * (S - C)[:, None]
		B = np.outer(C, H)
		return (A + B).reshape(outshape)

	elif Vtransform == 2:       # New transform by Shchepetkin
		N = Hc*S[:, None] + np.outer(C, H)
		D = (1.0 + Hc/H)
		return (N/D).reshape(outshape)

	else:
		raise ValueError("Unknown Vtransform")

# ------------------------------------


def sdepth_w(H, Hc, cs_w):
	"""Return depth of w-points in s-levels

	Kept for backwards compatibility
	use *sdepth(H, Hc, cs_w, stagger='w')* instead

	"""
	return sdepth(H, Hc, cs_w, stagger='w')


def multi_zslice(F, S, Z):

	"""Slice a 3D ECOM field to fixed depth

	Vertical interpolation of a field in s-coordinates to
	fixed vertical level

	*F* : array of with vertical profiles, first dimension is vertical

	*S* : array with depth of s-levels (at rho-points)
		1D (constant depth) or  S.shape = F.shape

	*Z* : single depth value, negative

	Returns : array, ``shape = F.shape[1:]`` the vertical slice

	"""

	# TODO:
	# Option to Save A, D, Dm
	#   => faster interpolate more fields to same depth

	F = np.asarray(F)
	S = np.asarray(S)
	Fshape = F.shape  # Save original shape

	# Flat all dimensions after first
	N = F.shape[0]
	M = F.size // N
	F = F.reshape((N, M))
	S = S.reshape((N, M))

	# Make z.shape = (M,)
	Z = np.asarray(Z, dtype='float')
	# Valid possibilities
	# 1) Z = single scalar (shape = ()), one constant value
	# 2) Z = 1D array, shape=(kmax), a set of constant depths
	# 3) Z = 2D or more, reshapeable to (kmax, M)

	if Z.ndim == 0:
		Z = Z + np.zeros((1, M))
		kmax = 1
	elif Z.ndim == 1:
		kmax = Z.size
		Z = Z[:, np.newaxis] + np.zeros((kmax, M))
	else:
		kmax = Z.size // M
		Z = Z.reshape((kmax, M))

	# Find C, C.shape = (kmax, M) such that
	# z_r[C[k,i]-1, i] < Z[k] <= z_r[C[k,i], i]

	# shape: kmax, N, M => kmax, M
	G = np.sum(S[np.newaxis, :, :] < Z[:, np.newaxis, :], axis=1)
	G = np.asarray(G, dtype = int)
	G = G.clip(1, N-1)

	# Horizontal index
	I = np.arange(M, dtype=int)

	# Compute interpolation weights
	A = (Z - S[(G-1, I)])/(S[(G, I)]-S[(G-1, I)])
	A = A.clip(0.0, 1.0)   # Control the extrapolation

	# Do the interpolation
	R = (1-A)*F[(G-1, I)]+A*F[(G, I)]

	# Give the result the correct shape
	R = R.reshape((kmax,) + Fshape[1:])

	return R, (A, G, I, kmax)

# ------------------------------------------------------


def sigma_to_z(var_sigma,depth,indz,z_rho_tot,zlevels):

	M = self.depth[0] #famoso y_depth
	N = self.depth[1] #famoso x_depth
	O = len(self.z_rho_tot)
	if not hasattr(self, 's2z_A'):
		#self.logger.debug('Calculating sigma2z-coefficients for whole domain')
		starttime = datetime.now()
		dummyvar = np.ones((O, M, N))
		dummy, self.s2z_total = depth_ECOM.multi_zslice(dummyvar, self.z_rho_tot, self.zlevels)
		# Store arrays/coefficients
		self.s2z_A = self.s2z_total[0].reshape(len(self.zlevels), M, N)
		self.s2z_C = self.s2z_total[1].reshape(len(self.zlevels), M, N)
		#self.s2z_I = self.s2z_total[2].reshape(M, N)
		self.s2z_kmax = self.s2z_total[3]
		del self.s2z_total  # Free memory
		self.logger.info('Time: ' + str(datetime.now() - starttime))
	if 'A' not in locals():
		self.logger.debug('Re-using sigma2z-coefficients')
		# Select relevant subset of full arrays
		zle = np.arange(zi1, zi2)  # The relevant depth levels
		A = self.s2z_A.copy()  # Awkward subsetting to prevent losing one dimension
		A = A[:,:,indx]
		A = A[:,indy,:]
		A = A[zle,:,:]
		C = self.s2z_C.copy()
		C = C[:,:,indx]
		C = C[:,indy,:]
		C = C[zle,:,:]
		C = C - C.max() + variables[par].shape[0] - 1
		C[C<1] = 1
		A = A.reshape(len(zle), len(indx)*len(indy))
		C = C.reshape(len(zle), len(indx)*len(indy))
		I = np.arange(len(indx)*len(indy))
		## Check
		#dummyvar2, s2z = depth.multi_zslice(
		#    variables[par].copy(), z_rho.copy(), variables['z'])
		#print len(zle), variables[par].shape, 'zle, varshape'
		#Ac,Cc,Ic,kmaxc = s2z
		#print C, 'C'
		#print Cc, 'Cc'
		#print C.shape, Cc.shape
		#if C.max() != Cc.max():
		#    print 'WARNING!!'
		#    import sys; sys.exit('stop')
		kmax = len(zle)  # Must be checked. Or number of sigma layers?
	if 'A' not in locals():
		self.logger.debug('Calculating new sigma2z-coefficients')
		variables[par], s2z = depth.multi_zslice(
			variables[par], z_rho, variables['z'])
		A,C,I,kmax = s2z
		# Reshaping to compare with subset of full array
		#zle = np.arange(zi1, zi2)
		#A = A.reshape(len(zle), len(indx), len(indy))
		#C = C.reshape(len(zle), len(indx), len(indy))
		#I = I.reshape(len(indx), len(indy))
	else:

		self.logger.debug('Applying sigma2z-coefficients')
		# Re-using sigma2z koefficients:
		F = np.asarray(variables[par])
		Fshape = F.shape
		N = F.shape[0]
		M = F.size // N
		F = F.reshape((N, M))
		R = (1-A)*F[(C-1, I)]+A*F[(C, I)]
		variables[par] = R.reshape((kmax,) + Fshape[1:])

	return variables[par]
