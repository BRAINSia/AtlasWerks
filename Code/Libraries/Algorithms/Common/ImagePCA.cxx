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

#include "ImagePCA.h"
#include "log.h"

#ifdef MPI_ENABLED
#include <mpi.h>
#endif // MPI_ENABLED

ImagePCA::
ImagePCA(unsigned int nImages, 
	 SizeType imSize, 
	 unsigned int nComponents)
  : mNImages(nImages),
    mNThreads(1),
    mNPowerIters(20)
{
  SetNComponents(nComponents);
  SetImageSize(imSize);
  mXXt.setSize(mNImages, mNImages);
  mU.setSize(mNImages, mNImages);
  mS.setSize(mNImages);
}

void 
ImagePCA::
SetImageSize(SizeType imSize)
{
  if(mImSize != imSize){
    mImSize = imSize;
    mNVox = mImSize.productOfElements();
    for(unsigned int i=0;i<mNComponents;i++){
      mComponent[i]->resize(mImSize);
    }
  }
}

void 
ImagePCA::
SetNComponents(unsigned int nComponents)
{
  if(mNComponents != nComponents){
    mNComponents = nComponents;
    while(mComponent.size() < mNComponents){
      mComponent.push_back(new RealImage(mImSize));
    }
    while(mComponent.size() > mNComponents){
      delete mComponent.back();
      mComponent.pop_back();
    }
  }
}

RealImage& 
ImagePCA::
GetComponent(unsigned int i){
  if(i > mNComponents){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, requested uncomputed pca component");
  }
  return *mComponent[i];
}

void
ImagePCA::
ComputePCA(std::vector<RealImage*> &images,
	   bool computeProjections)
{
  // compute X*X^T
  for(unsigned int r=0; r<mNImages; r++){
    RealImage &rIm = *images[r];
    Real* rData = rIm.getDataPointer();
    for(unsigned int c=0; c<mNImages; c++){
      RealImage &cIm = *images[c];
      Real* cData = cIm.getDataPointer();
      double sum = 0.f;
      for(unsigned int i=0;i<mNVox;i++){
	sum += rData[i]*cData[i];
      }
      mXXt(r, c) = sum;
    }
  }

  Matrix Vt;
  mXXt.factorSVD(mU, Vt, mS);

  // what we actually get is S^2, get S from it
  for(unsigned int r=0; r<mNImages; r++){
    mS(r) = sqrt(mS(r));
  }

  // get the columns of U we want
  Matrix pc(mNImages,mNComponents);
  mU.extractMatrix(0,0,pc);

  //
  // Construct Components
  //

  // get pointer arrays to component and image data, for convenience
  Real *dataPtr[mNImages];
  for(unsigned int imIdx=0; imIdx<mNImages; imIdx++){
    dataPtr[imIdx] = images[imIdx]->getDataPointer();
  }
  Real *pcPtr[mNComponents];
  for(unsigned int pcIdx=0; pcIdx<mNComponents; pcIdx++){
    pcPtr[pcIdx] = mComponent[pcIdx]->getDataPointer();
  }
  
  // compute P=U^T*X, a scaled version of the PCA components (U is
  // only nImages-by-nComponents, ie only keeping the columns of U we
  // want)
  for(unsigned int c=0;c<mNVox;c++){
    for(unsigned int r=0; r<mNComponents; r++){
      pcPtr[r][c] = 0;
      for(unsigned int i=0; i<mNImages; i++){
	pcPtr[r][c] += pc(i,r)*dataPtr[i][c];
      }
    }
  }
  
  if(computeProjections){
    
    // compute the projections as U*P, once again U is
    // nImages-by-nComponents.
    for(unsigned int c=0;c<mNVox;c++){
      for(unsigned int r=0; r<mNImages; r++){
	dataPtr[r][c] = 0;
	for(unsigned int i=0; i<mNComponents; i++){
	  dataPtr[r][c] += pc(r,i)*pcPtr[i][c];
	}
      }
    }
    
  }
  
  // scale the principle components by 1/Sigma
  for(unsigned int pcIdx=0; pcIdx < mNComponents; ++pcIdx){
    mComponent[pcIdx]->scale(1.0/mS(pcIdx));
  }
  
}

void 
ImagePCA::
RandomizeComponents()
{
  for(unsigned int pcIdx=0; pcIdx<mNComponents; ++pcIdx){
    double s = 0.0;
    RealImage &pc = *mComponent[pcIdx];
    Real *pcData = pc.getDataPointer();
    // generate random data and normalization factor
    for(unsigned int i=0; i<mNVox; i++){
      pcData[i] = ((Real)std::rand())/((Real)RAND_MAX);
      s += pcData[i]*pcData[i];
    }
    LOGNODE(logDEBUG3) << "norm factor is " << s;
    // normalize
    pc.scale(1.f/sqrt(s));
  }
}

void
ImagePCA::
ComputePCAPower(std::vector<RealImage*> &images)
{

  // yes, this is a giant matrix 
  // I should look into getting around allocating this...
  Matrix proj(mNImages,mNVox);
  proj.setAll(0.f);

  Real *MPIBuff = NULL;

  for(unsigned int cIdx=0; cIdx < mNComponents; ++cIdx){

    // compute buff = X*d
    double buff[mNImages];
    RealImage &pc = *mComponent[cIdx];
    Real *pcData = pc.getDataPointer();
    
    for(unsigned int iter=0; iter < mNPowerIters; ++iter){
      
      LOGNODE(logDEBUG3) << "power iteration " << iter;
      
      for(unsigned int r=0; r<mNImages; r++){
	Real *imData = images[r]->getDataPointer();
	buff[r] = 0;
	for(unsigned int c=0; c<mNVox; c++){
	  buff[r] += pcData[c]*(imData[c]-proj(r,c));
	}
      }
      
      // zero pcData
      pc.fill(0.f);
    
      // compute X^T*buff
      for(unsigned int r=0; r<mNImages; r++){
	Real *imData = images[r]->getDataPointer();
	for(unsigned int c=0; c<mNVox; c++){
	  pcData[c] += buff[r]*(imData[c]-proj(r,c));
	}
      }
      
#ifdef MPI_ENABLED
      
      if(MPIBuff == NULL) MPIBuff = new Real[mNVox];
      
      // compute the sum of all per-node principle components
      MPI_Allreduce(pcData, MPIBuff, mNVox,
		    MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      memcpy(pcData, MPIBuff, mNVox*sizeof(Real));
      
#endif
      
      // normalize 
      double normFac = 0.0;
      for(unsigned int c=0; c<mNVox; c++){
	normFac += pcData[c]*pcData[c];
      }
      LOGNODE(logDEBUG3) << "norm factor is " << normFac;
      pc.scale(1.0/sqrt(normFac));
      
    } // end iteration
    
    // compute projections
    
    for(unsigned int imIdx=0; imIdx<mNImages; imIdx++){
      Real *imData = images[imIdx]->getDataPointer();
      // compute dot product
      double s = 0;
      for(unsigned int i=0; i<mNVox; i++){
	s += (imData[i]-proj(imIdx,i))*pcData[i];
      }
      LOGNODE(logDEBUG2) << "Projection dot product for image " 
			 << imIdx << ": " << s;
      for(unsigned int i=0; i<mNVox; i++){
	proj(imIdx,i) += s*pcData[i];
      }
    }
    
  } // end loop over components

  if(MPIBuff != NULL) delete [] MPIBuff;
    
  // copy projections back into image data
  for(unsigned int r=0; r<mNImages; r++){
    Real *imData = images[r]->getDataPointer();
    for(unsigned int c=0; c<mNVox; c++){
      imData[c] = proj(r,c);
    }
  }

}
