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

#ifndef __GREEDY_ATLAS_BUILDER_SINGLE_GPU_THREAD_H__
#define __GREEDY_ATLAS_BUILDER_SINGLE_GPU_THREAD_H__

#include "AtlasWerksTypes.h"
#include "MultiParam.h"
#include "GreedyAtlasScaleLevelParam.h"
#include <Vector3D.h>
#include <vector>

#ifdef CUDA_ENABLED
#include "GreedyExecutionFFT.h"
#include <VectorMath.h>
#include <cudaReduce.h>
#include <cudaFFTSolver.h>
#include <cudaGaussianFilter.h>
#endif


class GreedyAtlasBuilderSingleGPUThread{
public:

  /**
   * \param node_id ID of the node this thread is running on
   * \param thread_id ID of this thread (relative to this node)
   * \param images set of images for which this thread is responsible
   * \param h_avgL the local average (mean) image, to be sent back to main 
   *        thread for computation of the global average
   * \param h_avgG the global average from the main thread
   * \param h_sqrErr location where host expects squared error from this thread
   */
  GreedyAtlasBuilderSingleGPUThread(int node_id, int thread_id,
				    std::vector<const RealImage*> images,
				    const MultiParam<GreedyAtlasScaleLevelParam> &param,
				    RealImage* h_avgL, RealImage* h_avgG, Real* h_sqrErr, int nTotalImgs);
    
  ~GreedyAtlasBuilderSingleGPUThread();
  
  void BuildAtlas();
  
  void InitDeviceData();
  void FreeDeviceData();

  void SetWeights(std::vector<float> weights){m_weights = weights;}
  
  float getDelta() { return m_delta; };
  void setDelta(float delta) { m_delta = delta;};
  
  int getThreadID() {return m_threadId; };
  int NumImages(){ return m_nImages; }
  
  void getHField(int imIdx, VectorField &vf);
  void getDefImage(int imIdx, RealImage &im);
  
private:
  
  void InitHostData();
  void FreeHostData();
  
  int  m_nodeId;
  int  m_threadId;
  unsigned int m_nTotalImgs;      // Total number of images
  unsigned int m_nVox;            // Number of voxels in images 
  unsigned int m_nImages;         // Number of images
  
  
  // image information
  MultiParam<GreedyAtlasScaleLevelParam> mParams;
  std::vector<const RealImage*> h_I0;
  // if no weights are specified, images are unscaled
  std::vector<float> m_weights; 
  
  Vector3Di m_vSize;        // Size of the image
  Vector3Df m_vOrg;         // Its origin  
  Vector3Df m_vSp;          // Spacing of the image 
  
  RealImage*  h_avgL;                 // host local average pointer
  RealImage*  h_avgG;                 // global average pointer 
  Real *h_sqrErr;                     // squared error for this node
  
  // parameters
  bool    mUseGlobalDelta;
  float   m_delta;               // Global delta value
  
  //
  // CPU variables, used to store final results on host
  //
  /** final deformed images */
  RealImage** h_I0t;
  /** final deformation fields*/
  VectorField** h_hFields;

  //
  // GPU variables
  //
  /** device array, original images */
  float** d_I0;
    
  /** device array, scaled version of original images */
  float** d_sI0;

  /** device array, current forward-deformed image  */
  float** d_I0t;

#ifdef CUDA_ENABLED
  /** the 3D vector field array */
  cplVector3DArray d_v;
    
  /** the 3D tranfomation array */
  cplVector3DArray *d_h;

  /** the average template */
  float* d_I1;

  /** scatch memory */
  float*            d_scratchI;

  /** template device memory */
  cplVector3DArray d_scratchV;

  /**  */
  cplReduce* p_Rd;
    
  /**  */
  FFTSolverPlan3D* pFFTSolver;
    
  /**  */
  GreedyExecutionFFT *d_vol;

    cplGaussianFilter* mSm;
#endif //CUDA_ENABLED
};

#endif 
