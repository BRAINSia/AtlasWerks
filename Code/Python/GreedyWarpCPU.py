#
# Import the AtlasWerks modules
#
from AtlasWerks.DataTypes import *
from AtlasWerks.Base import *
from AtlasWerks.UtilitiesDataTypes import *
from AtlasWerks.Algorithms import GreedyWarpCPU
from AtlasWerks.Algorithms import GreedyWarpParam
from AtlasWerks.Algorithms import GreedyScaleLevelParam

#
# import numpy and scipy (arrays and linear algebra goodies)
#
import numpy as np
import scipy
import scipy.linalg as linalg
#
# Import matplotlib for matlab-like plotting
#
import matplotlib
#
# I had problems getting the gtk backend to work, so tell matplotlib
# to use the Qt backend
#
matplotlib.use('WXAgg')
# pyplot contains most of the plotting functionality
import matplotlib.pyplot as plt
# tell it to use interactive mode -- see results immediately
plt.ion()

#
# Set up the parameters
#
print "Creating GreedyWarpParam"
param = GreedyWarpParam()
scaleLevel = GreedyScaleLevelParam()
scaleLevel.SetNIterations(100)
multiparam = param.ScaleLevel()
multiparam.AddParsedParam(scaleLevel)
scaleLevel.ScaleLevel().SetDownsampleFactor(2)
multiparam.AddParsedParam(scaleLevel)

#
# Read in the images
#
print "Loading images"
im0 = fImage()
im1 = fImage()
ApplicationUtils.LoadImageITK("/home/sam/Projects/AtlasWerks/Testing/Data/Input/Bullseyes/BullseyeTest00.mha", im0)
ApplicationUtils.LoadImageITK("/home/sam/Projects/AtlasWerks/Testing/Data/Input/Bullseyes/BullseyeTest01.mha", im1)

#
# Initial Image Difference
#
# origDiff = Array3DUtils.squaredDifference(im0, im1);

#
# Create the warper
#
print "Creating Warper"
warper = GreedyWarpCPU(im0, im1, param)
print "Running Warp"
#warper.RunWarp()

defIm = fImage()
#warper.GetDefImage(defIm)

#
# Final Image Difference
#
# finalDiff = Array3DUtils.squaredDifference(defIm, im1);

#
# Test the final error compared to original error
#
errPct = 1.0;
# if(origDiff*(errPct/100.0) < finalDiff):
#     imArray = np.array(defIm.tolist())
#     imSlice = np.squeeze(imArray[:,:,16])
#     plt.imshow(imSlice)
#     plt.gray()
# else:
#     print "Passed, image diff percent is {0:g}%".format(100*(finalDiff/origDiff))

