/* ================================================================
 *
 * AtlasWerks Project
 *
 * Copyright (c) Sarang C. Joshi, Bradley C. Davis, J. Samuel Preston,
 * Linh K. Ha. All rights reserved.  See Copyright.txt or for details.
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notice for more information.
 *
 * ================================================================ */


#include "LDMMAtlasThreadCPU.h"

/* ################ LDMMAtlasManagerCPU ################ */

unsigned int 
LDMMAtlasManagerCPU::
CheckNumThreads(unsigned int requestedThreads, unsigned int nImages)
{
  unsigned int nThreads = requestedThreads;
  if(requestedThreads == 0){
    nThreads = nImages;
    LOGNODE(logINFO) << "Using " << nThreads << " threads.";
  }

  if(nThreads > nImages){
    LOGNODE(logWARNING) << "Number of threads (" << nThreads 
			<< ") greater than number of images (" << nImages
			<< "), using " << nImages << " threads" << std::endl;
    nThreads = nImages;
  }

  return nThreads;
}

/* ################ LDMMAtlasDefDataCPU ################ */

LDMMAtlasDefDataCPU::
LDMMAtlasDefDataCPU(const RealImage *I, 
		    const RealImage *IHat,
		    Real weight,
		    const LDMMParam &param) :
  LDMMDeformationData(IHat, I, param),
  mJacDet(new RealImage()),
  mWeight(weight)
{
  // IHat is I0, do not automatically rescale this
  this->ScaleI0(false);
}

LDMMAtlasDefDataCPU::
~LDMMAtlasDefDataCPU()
{
  delete mJacDet;
}

void 
LDMMAtlasDefDataCPU::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  // call parent function
  LDMMDeformationData::SetScaleLevel(scaleManager);

  if(scaleManager.InitialScaleLevel()){
    mJacDet->resize(mCurSize);
    mJacDet->setSpacing(mCurSpacing);
    mJacDet->setOrigin(mImOrigin);
    mJacDet->fill(1.0);
  }else{
    scaleManager.UpsampleToLevel(*mJacDet, scaleManager.CurScaleLevel());
  }

}

/* ################ LDMMAtlasThreadCPU ################ */

LDMMAtlasThreadCPU::
LDMMAtlasThreadCPU(std::vector<DeformationDataType*> defData,
		   const ParamType &param,
		   AtlasBuilderInterface &builder,
		   RealImage *globalMean,
		   unsigned int nodeId, unsigned int nNodes,
		   unsigned int threadId, unsigned int nThreads,
		   unsigned int nTotalImages)
  : AtlasThread<LDMMAtlasManagerCPU>(defData, param, builder, globalMean, 
				     nodeId, nNodes, threadId, nThreads, nTotalImages),
    mNTimeSteps(param.NTimeSteps()),
    mThreadMean(NULL),
    mJacDetSum(NULL),
    mThreadEnergy(NULL),
    mDist(NULL),
    mTMWeights(NULL)
{
  mThreadMean = new RealImage();
  mJacDetSum = new RealImage();
  mDist = new Real[mNImages];
  mTMWeights = new Real[mNImages];
  memset(mDist, 0, mNImages*sizeof(Real));
  memset(mTMWeights, 0, mNImages*sizeof(Real));
}

LDMMAtlasThreadCPU::
~LDMMAtlasThreadCPU()
{
  delete mThreadMean;
  delete mJacDetSum;
  delete [] mDist;
  delete [] mTMWeights;
}

void 
LDMMAtlasThreadCPU::
InitThread()
{
  if(mIterator) delete mIterator;
  mIterator = new IteratorType(mImSize, mImOrigin, mImSpacing, mNTimeSteps, true);
}

void 
LDMMAtlasThreadCPU::
LoadInitData()
{
  // TODO: load VFields
  if(mParam.InputVFieldFormat().size() > 0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, LDMM relaxation VField initialization not implemented yet");
  }
  
  if(mParam.InputMeanImage().size() > 0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, cannot initialize LDMM relaxation from alpha0s");
  }
  if(mParam.InputAlpha0Format().size() > 0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, cannot initialize LDMM relaxation from alpha0s");
  }
}

void
LDMMAtlasThreadCPU::
FinishThread()
{
  if(mParam.WriteAlphas() || mParam.WriteAlpha0s()){
    this->ComputeAtlasAlphas();
  }
}

void
LDMMAtlasThreadCPU::
BeginImageIteration(int iteration, int imageIndex)
{
  if(iteration == 0 && mCurScaleParam->UseAdaptiveStepSize()){
    LOGNODETHREAD(logINFO) << "Calculating step size";
    mIterator->UpdateStepSizeNextIteration();
    mIterator->Iterate(*mDeformationData[imageIndex]);
    LOGNODETHREAD(logINFO) << "Done calculating step size";
  }
}

void
LDMMAtlasThreadCPU::
FinishImageIteration(int iteration, int imageIndex)
{
  // test for increasing energy
  if(iteration > 0){
    
    EnergyHistory &hist = mDeformationData[imageIndex]->GetEnergyHistory();
    Real energyDiff = hist.LastEnergyChange();
    if(energyDiff > 0.f){
      LOGNODETHREAD(logWARNING) << "Increasing energy detected: image " << imageIndex 
				<< " (" << mDeformationData[imageIndex]->GetName()
				<< "), increase = " << energyDiff;
    }
  }
  
  // calculate distance
  if(mDeformationData[imageIndex]->HasEnergy()){
    mDist[imageIndex] = sqrt(mDeformationData[imageIndex]->LastEnergy().GetEnergy());
  }else{
    mDist[imageIndex] = 0.f;
  }
}

void
LDMMAtlasThreadCPU::
SetScaleLevel(const MultiscaleManagerType &scaleManager)
{
  AtlasThread<LDMMAtlasManagerCPU>::SetScaleLevel(scaleManager);

  for(unsigned int i=0;i<mNImages;i++){
    mDeformationData[i]->StepSize(mCurScaleParam->StepSize());
  }
  
  mThreadMean->resize(mCurSize);
  mThreadMean->setSpacing(mCurSpacing);
  mThreadMean->setOrigin(mImOrigin);
  mJacDetSum->resize(mCurSize);
  mJacDetSum->setSpacing(mCurSpacing);
  mJacDetSum->setOrigin(mImOrigin);

  if(mThreadEnergy) delete mThreadEnergy;
  mThreadEnergy = new LDMMEnergy(mNTimeSteps, mCurScaleParam->Sigma());
  
  // still need to generate the mean at this scale level...
}

void
LDMMAtlasThreadCPU::
ComputeThreadMean()
{
  if(mParam.TrimmedMeanSize() > 0){
    // sychronizes threads
    mBuilder.ComputeWeights();
  }
  
  // create the weighted average for all images on this node
  mThreadMean->fill(0.0);
  mJacDetSum->fill(0.0);
  mThreadEnergy->Clear();
  Real *averageData = mThreadMean->getDataPointer();
  Real *jacSumData = mJacDetSum->getDataPointer();
  unsigned int size = mThreadMean->getNumElements();
  RealImage defIm(mImSize, mImOrigin, mImSpacing);
  for(unsigned int imIdx=0; imIdx<mNImages; imIdx++){

    // get the weight
    Real curWeight = mDeformationData[imIdx]->Weight();
    Real imWeight = curWeight;
    if(mParam.ComputeMedian()){
      Real d = mDist[imIdx];
      // on first mean calculation energy is zero, don't reweight (just compute mean)
      if(d == 0) d = 1.f;
      d = 1.0/d;
      imWeight *= d;
      LOGNODETHREAD(logDEBUG2) << "individual InvDistSum: " << d;
    }

    if(mParam.TrimmedMeanSize()){
      imWeight = mTMWeights[imIdx];
    }

    // generate the deformed image
    mDeformationData[imIdx]->GetI1At0(defIm);
    const Real *imageData = defIm.getDataPointer();

    // get the jac det for this deformation
    mIterator->ComputeJacDet(mDeformationData[imIdx]->v(), &mDeformationData[imIdx]->JacDet());
    const Real *jacData = mDeformationData[imIdx]->JacDet().getDataPointer();
    // update weighted image and jac det sums
    if(mParam.JacobianScale()){
      for(unsigned int j = 0; j < size; j++){
	averageData[j] += imageData[j] * jacData[j] * imWeight;
	jacSumData[j] += jacData[j] * imWeight;
      }
    }else{
      for(unsigned int j = 0; j < size; j++){
	averageData[j] += imageData[j] * imWeight;
      }
    }
    // sum energies
    // get the energy
    if(mDeformationData[imIdx]->HasEnergy()){
      const LDMMEnergy &curEnergy = 
	dynamic_cast<const LDMMEnergy&>(mDeformationData[imIdx]->LastEnergy());
      
      //(*mThreadEnergy) += curEnergy * curWeight;
      (*mThreadEnergy) += curEnergy * imWeight;
    }
    if(mDeformationData[imIdx]->SaveDefToMean()){
      *mDeformationData[imIdx]->GetDefToMean() = defIm;
    }

  } // end loop over images
}

void
LDMMAtlasThreadCPU::
ComputeAtlasAlphas()
{
  LOGNODETHREAD(logDEBUG) << "Computing alphas";
  for(unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
    LOGNODETHREAD(logDEBUG) << "Computing alphas for image " << imIdx << "...";
    // compute the alphas
    mDeformationData[imIdx]->ComputeAlphas(true);
    mIterator->Iterate(*mDeformationData[imIdx], true);
  }
}
