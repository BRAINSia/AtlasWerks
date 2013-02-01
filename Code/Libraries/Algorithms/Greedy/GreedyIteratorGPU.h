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


#ifndef __GREEDY_ITERATOR_GPU_H__
#define __GREEDY_ITERATOR_GPU_H__

#include "AtlasWerksTypes.h"
#include "VectorMath.h"
#include "cudaInterface.h"
#include "cudaReduce.h"
#include "KernelInterfaceGPU.h"
#include "DeformationIteratorInterface.h"
#include "DiffOperGPU.h"
#include "MultiGaussianKernelGPU.h"
#include "GreedyDeformationDataGPU.h"
#include "MultiscaleManager.h"

#include "Energy.h"
#include "GreedyIteratorParam.h"

class GreedyIteratorGPU : 
  public DeformationIteratorInterface
{

public:
  
  typedef GreedyDeformationDataGPU DeformationDataType;
  typedef GreedyIteratorParam ParamType;


  GreedyIteratorGPU(SizeType &size, 
		    OriginType &origin,
		    SpacingType &spacing,
		    bool debug=true);
  
  ~GreedyIteratorGPU();

  void SetScaleLevel(const MultiscaleManager &scaleManager,
		     const GreedyIteratorParam &param);
  
  /** Get/set whether to calculate debugging info (energy calculations, etc.) */
  void SetDebug(bool debug){mDebug = debug; }
  bool GetDebug(){return mDebug; }
  
  void Iterate(DeformationDataInterface &deformationData);

  void UpdateStepSizeNextIteration(){ mUpdateStepSizeNextIter = true; }

protected:

  void updateDeformation(GreedyDeformationDataGPU &data);
  
  /** Relative step size, between 0 and 1 */
  Real mMaxPert;

  /** Should the stepsize be updated based on MaxPert next iteration? */
  bool mUpdateStepSizeNextIter;

  /** Do we calculate energy for debugging? */
  bool mDebug;

  const GreedyIteratorParam *mParam;

  // device data

  /** Reduce plan*/
  cplReduce *mRd; 

  /** FFT Solver */
  KernelInterfaceGPU *mKernel;

  /** Update Vector Field */
  cplVector3DArray mdU;

  /** Scratch Image */
  float *mdScratchI;
  /** Scratch Vector Field */
  cplVector3DArray mdScratchV;
  
};

#endif
