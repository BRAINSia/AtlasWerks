#
# Import the AtlasWerks modules
#
from AtlasWerks.DataTypes import *
from AtlasWerks.Base import *
from AtlasWerks.UtilitiesDataTypes import *
from AtlasWerks.Algorithms import DiffOper
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
matplotlib.use('Qt4Agg')
# pyplot contains most of the plotting functionality
import matplotlib.pyplot as plt
# tell it to use interactive mode -- see results immediately
plt.ion()

def Plot2DExample():
    """
    Just a simple 2D plotting example using numpy and matplotlib
    """
    plt.figure(1)
    x = np.linspace(-2,2,100)
    g = np.exp(-(x*x))
    plt.plot(x,g)
    #plt.show()
    plt.figure(2)
    gx = np.expand_dims(g, axis=1)
    gy = np.expand_dims(g, axis=0)
    g = gx*gy
    plt.imshow(g)
    #plt.show()

def WriteImageExample():
    """
    Create a 3D tensor-product gaussian as a numpy array, convert it
    to a fImage and save it as a MHA image
    """
    x = np.linspace(-2,2,100)
    g = np.exp(-(x*x))
    #
    # This is one way to create copies in different dimensions of a 3D
    # array
    #gx = np.expand_dims(g, axis=1)
    #gx = np.expand_dims(gx, axis=2)
    #gy = np.expand_dims(g, axis=0)
    #gy = np.expand_dims(gy, axis=2)
    #gz = np.expand_dims(g, axis=0)
    #gz = np.expand_dims(gz, axis=0)
    #
    # This is another way
    #
    gy = np.atleast_3d(g)
    gx = np.transpose(gy, [1, 0, 2])
    gz = np.transpose(gy, [0, 2, 1])
    # multiply, uses 'broadcast' rules to create tensor-product
    g = gx*gy*gz
    # ListToArray3D is part of DataTypes package
    im = fImage()
    im.fromlist(g.tolist())
    ApplicationUtils.SaveImageITK("output.mha", im)

def ReadImageExample():
    """
    Read the image created by WriteImageExample, create a numpy
    array from it, and plot a slice
    """
    im = fImage()
    ApplicationUtils.LoadImageITK("output.mha", im)
    # tolist is added to python interface
    npim = np.array(im.tolist())
    slice = np.squeeze(npim[:,:,16])
    plt.imshow(slice)
    #plt.show()

def MatInv():
    """
    Invert a diagonal matrix, just to test scipy's linalg lib. Note,
    for least-squares use linalg.lstsq(a,b) for ax=b
    """
    a = np.array([[2., 0., 0.],[0., 5., 0.], [0., 0., 3.2]])
    print a
    aInv = linalg.inv(a)
    print aInv

def GenCrossTestIm(sz=[50, 50, 50], width=10, border=5, sigma=0, asImage=False):
    """Generate a test image of a 3D 'plus' symbol"""
    if type(sz) == list:
        sz = np.array(sz)
    im = np.zeros(sz)
    center = sz/2
    im[(center[0]-width/2):(center[0]+width/2), (center[1]-width/2):(center[1]+width/2), border:(sz[2]-border)] = 1.0
    im[(center[0]-width/2):(center[0]+width/2), border:(sz[1]-border), (center[2]-width/2):(center[2]+width/2)] = 1.0
    im[border:(sz[0]-border), (center[1]-width/2):(center[1]+width/2), (center[2]-width/2):(center[2]+width/2)] = 1.0
    if sigma > 0:
        import scipy.ndimage
        im = scipy.ndimage.filters.gaussian_filter(im, sigma)

    if asImage:
        rtnImage = fImage()
        rtnImage.fromlist(im.tolist())
        im = rtnImage
    return im

def GenRandVelField(sz=[50, 50, 50], maxVLen=5, nVecs=10, asVectorField=False):
    """Generate a random 'smooth' vector field
    
    Generate a random 'smooth' vector field by seeding an empty vector
    field with nVecs randomly placed and oriented vectors, and then
    applying the K operator to create a smooth field
    """

    if type(sz) == list:
        sz = np.array(sz)

    # Create the bland vector field as a numpy array
    vfSz = np.append(sz,3)
    vf = np.zeros(vfSz)

    # Create the initial random vectors
    pos = np.random.rand(nVecs, 3)*sz
    pos = pos.astype('int')
    dir = np.random.rand(nVecs, 3)
    l = np.sqrt(np.sum(dir*dir, 1))
    l = np.expand_dims(l,1)
    dir = dir / l
    dir = dir * maxVLen
    for vIdx in range(nVecs):
        vf[pos[vIdx,0],pos[vIdx,1],pos[vIdx,2],:] = dir[vIdx,:]

    # Apply K
    vecField = fVectorField()
    vecField.fromlist(vf.tolist())
    imSz = uiVector3D()
    imSz.fromlist(sz.tolist())
    diffOp = DiffOper(imSz)
    diffOp.Initialize()
    diffOp.CopyIn(vecField)
    diffOp.ApplyInverseOperator()
    diffOp.CopyOut(vecField)
    vf = np.array(vecField.tolist())

    # Re-normalize and scale
    l = np.squeeze(np.sum(vf*vf, 3))
    l = np.sqrt(np.max(l))
    vf = vf / l
    vf = vf * maxVLen

    if asVectorField:
        vecField.fromlist(vf.tolist())
        vf = vecField

    return vf

def DiffOperExample(vecFieldFileName):
    """
    Read in a vector field, apply differential operator (L and K) to
    it, and display components of it as well as a quiver plot
    """
    # Read in the vector field
    vf = fVectorField()
    hOrigin = dVector3D()
    hSpacing = dVector3D()
    ApplicationUtils.LoadHFieldITK(vecFieldFileName, hOrigin, hSpacing, vf)
    # Test if it is an hField (deformation) or vField (offset)
    if HField3DUtils.IsHField(vf):
        # Convert to vField
        HField3DUtils.hToVelocity(vf, hSpacing)
    # Plot the x-component of a slice of the vField
    plt.figure(1)
    plt.subplot(221)
    vfArray = np.array(vf.tolist())
    plt.imshow(np.squeeze(vfArray[:,:,16,0]))
    # Apply the differential operator L
    diffOp = DiffOper(vf.getSize(), hSpacing)
    diffOp.Initialize()
    diffOp.CopyIn(vf)
    diffOp.ApplyOperator()
    diffOp.CopyOut(vf)
    # Plot the x component of a slice again
    plt.subplot(222)
    vfArray = np.array(vf.tolist())
    plt.imshow(np.squeeze(vfArray[:,:,16,0]))
    # Apply the differential operator K
    diffOp.CopyIn(vf)
    diffOp.ApplyInverseOperator()
    diffOp.CopyOut(vf)
    # Plot the x component of a slice again
    plt.subplot(223)
    vfArray = np.array(vf.tolist())
    plt.imshow(np.squeeze(vfArray[:,:,16,0]))
    # Create a 2D quiver plot of the slice
    vf_x = np.squeeze(vfArray[:,:,16,0])
    vf_y = np.squeeze(vfArray[:,:,16,1])
    plt.figure(2)
    plt.quiver(vf_x,vf_y)

#DiffOperExample("VField.mha");
