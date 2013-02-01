#!/usr/bin/python2

#
# Import the AtlasWerks modules
#
from AtlasWerks.DataTypes import *
from AtlasWerks.Base import *
from AtlasWerks.UtilitiesDataTypes import *

import AtlasWerksVis as AWVis

from AtlasWerksPythonExtra import *

import time   # for timing stuff
import sys    # for flushing stdout

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot
matplotlib.pyplot.ion()

import numpy
from matplotlib.pylab import *

# cube image dimensions
N=64

# Create random (smooth) vector field
vf = fVectorField(N,N,N)
# Create white noise vector field
vf.fromlist(randn(N,N,N,3)*0.01)
# Apply K to get a smoothed version of this
vf = applyDiffOperInv(vf,0.1,0.01,0.0005,0)

# quiver plot a slice out of the vector field
figure(1)
clf()
AWVis.QuiverSlice(vf,AWVis.DIR_X,round(N/2),scale=0)
title('Original smooth vector field')

# figure(3)
# clf()
# AWVis.Quiver3D(vf)
# title('Original smooth vector field (3D quiver)')

# Apply group exponential (time it)
h = fVectorField(N,N,N)
hf3u = HField3DUtils()
print "Computing group exponential..."
sys.stdout.flush()
start = time.clock()
hf3u.GroupExponential(vf,1000,8,h)
print "t=%.2f sec" % (time.clock()-start)
print "DONE"

# quiver plot the resulting deformation
hf3u.hToVelocity(h); # note this converts in-place
figure(2)
clf()
AWVis.QuiverSlice(h,AWVis.DIR_X,round(N/2),scale=0)
title('Flow displacement field')
