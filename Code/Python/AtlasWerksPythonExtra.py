# A bunch of extra code to wrap common code in AtlasWerks (these
# should probably be included into AtlasWerks at some point)

from AtlasWerks.DataTypes import *
from AtlasWerks.Base import *
from AtlasWerks.UtilitiesDataTypes import *

# DiffOper-related stuff
from AtlasWerks.Algorithms import GreedyWarpParam
from AtlasWerks.Algorithms import GreedyScaleLevelParam
from AtlasWerks.Algorithms import DiffOper

import AtlasWerksVis as AWVis

def applyDiffOperInv(ivf,alpha,beta,gamma,incomp):
    """
    Just apply the DiffOper operator inverse (just L^*L^-1=K).  Note
    that if you're going to do this repeatedly then you should do it
    the right way, just instantiate one DiffOper and call it
    repeatedly.
    """
    op = DiffOper(ivf.getSize())

    op.SetAlpha(alpha)
    op.SetBeta(beta)
    op.SetGamma(gamma)
    op.SetDivergenceFree(incomp)
    # default LPow is 1

    op.Initialize() # initialize (sets up FFTs)
    print "initialized.."
    op.CopyIn(ivf)
    print "copied in"
    ovf = fVectorField(ivf.getSize())
    print "allocd ovf"
    op.ApplyInverseOperator()
    print "applied K"
    op.CopyOut(ovf)
    print "copied out"

    return ovf
