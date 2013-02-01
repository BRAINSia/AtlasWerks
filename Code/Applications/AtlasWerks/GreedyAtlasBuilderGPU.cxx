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

#include "GreedyAtlasBuilderGPU.h"
#include <cudaInterface.h>
#include <cutil_comfunc.h>
#include <pthread.h>
#include <multithreading.h>
#include <cudaDownsizeFilter3D.h>

#include "AtlasWerksException.h"

#ifdef MPI_ENABLED
#include <mpi.h>
#endif

pthread_barrier_t barr;

/**
 * One thread based on this function will be created for each device
 * on the node
 */
static CUT_THREADPROC GreedyThread(GreedyAtlasBuilderSingleGPUThread* greedy){
  // Set the device context
  cudaSetDevice(greedy->getThreadID());
 
  // Initialize the GPU data
  greedy->InitDeviceData();

  // run the atlas build
  greedy->BuildAtlas();

  // free device data
  greedy->FreeDeviceData();

}

/**
 * This will be the 'main' thread, where synchronization across
 * threads is managed
 */
static CUT_THREADPROC SumThread(GreedyAtlasBuilderGPU* main)
{
  main->computeAverageThread();
}

GreedyAtlasBuilderGPU::
GreedyAtlasBuilderGPU(
		      unsigned int nodeId,
		      unsigned int nNodes,
		      unsigned int nTotalImgs,
		      WeightedImageSet &imageSet,
		      const MultiParam<GreedyAtlasScaleLevelParam> &param,
		      unsigned int nGPUs)
  :m_nodeId(nodeId),
   m_nNodes(nNodes),
   m_nTotalImgs(nTotalImgs),
   m_imageSet(imageSet),
   mParams(param),
   m_nGPUs(nGPUs),
   m_sqrErr(NULL),
   mUseGlobalDelta(false),
   m_deltaG_i(FLT_MAX),
   m_deltaG_o(FLT_MAX)
{
  init();
}

GreedyAtlasBuilderGPU::
GreedyAtlasBuilderGPU(
		      WeightedImageSet &imageSet,
		      const MultiParam<GreedyAtlasScaleLevelParam> &param,
		      unsigned int nGPUs)
  :m_nodeId(0),
   m_nNodes(1),
   m_imageSet(imageSet),
   mParams(param),
   m_nGPUs(nGPUs),
   m_sqrErr(NULL),
   mUseGlobalDelta(false),
   m_deltaG_i(FLT_MAX),
   m_deltaG_o(FLT_MAX)
{
  m_nTotalImgs = m_imageSet.NumImages();
  init();
}

GreedyAtlasBuilderGPU::
~GreedyAtlasBuilderGPU()
{
  FreeHostData();
}

void
GreedyAtlasBuilderGPU::
checkParams()
{
  // need to test that we're not using any unsupported parameters
  if(m_imageSet.HasTransforms()){
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "Error, cannot use transforms with GPU implementation");
  }
  
  // check scale level params
  for(unsigned int s = 0; s < mParams.size(); s++){
    GreedyAtlasScaleLevelParam &curLevelParam = mParams[s];
    if(curLevelParam.UpdateAfterSubIteration()){
      throw AtlasWerksException(__FILE__, __LINE__, 
			     "Error, UpdateAfterSubIteration not supported in GPU implementation");
    }
    if(curLevelParam.DiffOper().LPow() != 1){
      throw AtlasWerksException(__FILE__, __LINE__, 
			     "Error, non-unitary powers of differential operator not supported"
			     " in GPU implementation");
    }
    if(curLevelParam.DiffOper().DivergenceFree()){
      throw AtlasWerksException(__FILE__, __LINE__, 
			     "Error, divergence-free differential operator not supported in "
			     "GPU implementation");
    }
  }
}

void 
GreedyAtlasBuilderGPU::
init()
{
  // make sure the settings are valid
  checkParams();

  m_nImages = m_imageSet.NumImages();
  
  // have to create a vector of const pointers
  for(unsigned int i=0;i<m_nImages;i++) h_I0.push_back(m_imageSet.GetImage(i));

  m_weights = m_imageSet.GetWeightVec();
  
  m_vSize = h_I0[0]->getSize();
  m_vOrg  = h_I0[0]->getOrigin();
  m_vSp   = h_I0[0]->getSpacing();
  
  m_nVox    = m_vSize.productOfElements();
  
  unsigned int systemGPUs = getNumberOfCapableCUDADevices();
  if(m_nGPUs == 0){
    m_nGPUs = systemGPUs;
  }else if(m_nGPUs > systemGPUs){
    std::cout << "Cannot use " << m_nGPUs << " GPUs, only " << systemGPUs << " available" << std::endl;
    m_nGPUs = systemGPUs;
  }

  if (m_nGPUs > m_nImages){
    std::cout << "Node " << m_nodeId << ": More GPUs are available than images, will only use " << m_nImages << " GPUs." << std::endl;
    m_nGPUs = m_nImages;
  }
  
  fprintf(stderr, "Number of GPU devices: %d \n", m_nGPUs);
  
  initHostData();
  
  // create builder pointer 
  m_builders = new GreedyAtlasBuilderSingleGPUThread* [m_nGPUs];
}

void GreedyAtlasBuilderGPU::
initHostData()
{
  // allocate local average
  h_avgL.resize(m_nGPUs);
  for (unsigned int i=0; i< m_nGPUs; ++i){
    RealImage* img = new RealImage(m_vSize, m_vOrg, m_vSp);
    h_avgL[i] = img;
  }
  // allocate aray for device errors
  m_sqrErr = new Real[m_nGPUs];
  for (unsigned int i=0; i< m_nGPUs; ++i){ m_sqrErr[i] = 0; }

  // allocate the global average of the node 
  h_avgNode_o   = h_avgL[0];
#ifdef MPI_ENABLED
  h_avgNode_i   = new RealImage(m_vSize, m_vOrg, m_vSp);
#else
  h_avgNode_i = h_avgL[0];
#endif

}

void GreedyAtlasBuilderGPU::
FreeHostData()
{
  for (unsigned int i=0; i< m_nGPUs; ++i)
    delete h_avgL[i];
#ifdef MPI_ENABLED
  delete h_avgNode_i;
#endif
}

void cpuAdd_I(float* h_a, float* h_b, int n)
{
  for (int i=0; i< n; ++i)
    h_a[i] += h_b[i];
}

void cpuMul_C(float* h_a, float c, int n)
{
  for (int i=0; i< n; ++i)
    h_a[i] *= c;
}

void 
GreedyAtlasBuilderGPU::
SumAcrossNodes(int nP)
{
#ifdef MPI_ENABLED
  // compute the total sum of all image on the cluster
  MPI_Allreduce(h_avgNode_o->getDataPointer(0), h_avgNode_i->getDataPointer(0),
		nP, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  // sum error across nodes
  // get summed error across all nodes
  Real sum;
  MPI_Reduce(&m_sqrErr[0], &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  m_sqrErr[0] = sum;
#endif
}

void 
GreedyAtlasBuilderGPU::
UpdateGlobalDelta()
{
  if (mUseGlobalDelta){
    m_deltaG_o = FLT_MAX;
    
    // synchronize the global compute of the delta 
    pthread_barrier_wait(&barr);
    for (unsigned int i=0; i< m_nGPUs; ++i){
      float deltaL = m_builders[i]->getDelta();
      m_deltaG_o = (m_deltaG_o < deltaL) ? m_deltaG_o : deltaL;
    }
    
#ifdef MPI_ENABLED
    // get minimum delta across all nodes
    MPI_Allreduce(&m_deltaG_o, &m_deltaG_i,  1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
#else
    // use local minimum delta
    m_deltaG_i = m_deltaG_o;
#endif
    
    // update global delta
    for (unsigned int i=0; i< m_nGPUs; ++i)
      m_builders[i]->setDelta(m_deltaG_i);
    
    pthread_barrier_wait(&barr);
    
    fprintf(stderr, "Delta value %f \n",m_deltaG_i);
  }
}

// Compute the avergage image using the reduction 
// result return in the h_avgL[0] array 
void 
GreedyAtlasBuilderGPU::
computeAverageThread()
{
  int3 org_size = make_int3(m_vSize.x, m_vSize.y, m_vSize.z);
  int3 isize    = org_size;

  for (unsigned int iL=0; iL < mParams.size(); ++iL){
    //compute the size of the image 
    int       f  = mParams[iL].ScaleLevel().DownsampleFactor();
    int3  factor = make_int3(f,f,f);

    //computeDownsize(isize, org_size, factor);
    isize.x = org_size.x / f;
    isize.y = org_size.y / f;
    isize.z = org_size.z / f;
    
    int nP = isize.x * isize.y * isize.z;
        
    // synchronize and compute the sum
    for (unsigned int  cnt=0; cnt< mParams[iL].NIterations(); ++cnt){
      pthread_barrier_wait(&barr);

      // compute the combined average for all GPUs on this node
      computeAverage(nP);

      // sum the error for this node in m_sqrErr[0]
      for(unsigned int gpuId=1;gpuId<m_nGPUs;gpuId++){
	m_sqrErr[0] += m_sqrErr[gpuId];
      }
            
      // compute the total sum of all image on the cluster
      SumAcrossNodes(nP);

      //             MPI_Allreduce(h_avgNode_o->getDataPointer(0), h_avgNode_i->getDataPointer(0),
      //                           nP, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

      // Write out intermediate images if requested
      if(m_nodeId == 0){
	if(cnt > 0){
	  std::cout << "RMSE for iteration " << cnt << " is " << sqrt(m_sqrErr[0]/m_nTotalImgs) << std::endl;
	}

	if(mParams[iL].OutputMeanEveryNIterations() > 0 
	   && (cnt+1)%mParams[iL].OutputMeanEveryNIterations() == 0)
	  {
	    RealImage tmp;
	    this->GetMeanImage(tmp);
	    std::string name = StringUtils::strPrintf("IntermediateMeanImageScaleLevel%02dIter%04d.mhd",iL,cnt);
	    ApplicationUtils::SaveImageITK(name.c_str(), tmp);
	  }
      }
      // end write 

      pthread_barrier_wait(&barr);
            
      if (cnt == 0 && mUseGlobalDelta){
	// This function has pthread_barrier synchronization,
	// finds minimum delta across threads/nodes
	UpdateGlobalDelta();
      }

    } // end loop over iterations
  } // end loop over scales
}

//
// Compute the sum of all GPU averages.
//
void 
GreedyAtlasBuilderGPU::
computeAverage(int nVox)
{
  // sum
  unsigned int half = nextPowerOf2(m_nGPUs) >> 1;
  for (unsigned int i=0; i< half; ++i){
    if (i + half < m_nGPUs)
      cpuAdd_I(h_avgL[i]->getDataPointer(0), h_avgL[i+half]->getDataPointer(0), nVox);
  }

  half >>= 1;
  while (half > 0){
    for (unsigned int i=0; i< half; ++i)
      cpuAdd_I(h_avgL[i]->getDataPointer(0), h_avgL[i+half]->getDataPointer(0), nVox);
    half >>=1;
  }

}

void 
GreedyAtlasBuilderGPU::
BuildAtlas()
{
  fprintf(stderr, "Create the builders ");
  for (unsigned int i=0; i< m_nGPUs; ++i)
    {
      int bid;
      int nLocalImgs;
      std::cout << "Distributing " << m_nImages << " among " << m_nGPUs << std::endl;
      ApplicationUtils::Distribute(m_nImages, m_nGPUs, i, bid, nLocalImgs);
      m_bid.push_back(bid);
      std::vector<const RealImage*> imgs(h_I0.begin() + bid, h_I0.begin() + bid + nLocalImgs);
      std::vector<Real> weights(m_weights.begin() + bid, m_weights.begin() + bid + nLocalImgs);
      m_builders[i] = new GreedyAtlasBuilderSingleGPUThread(m_nodeId, i, imgs, mParams,
							    h_avgL[i], h_avgNode_i, &m_sqrErr[i], m_nTotalImgs);
      m_builders[i]->SetWeights(weights);
    }

  fprintf(stderr, "... done \n");

  // create Barrier 
  if(pthread_barrier_init(&barr, NULL, m_nGPUs + 1)){
    printf("Could not create a barrier\n");
    return;
  }

  fprintf(stderr, "Create the threads...\n ");
  CUTThread*  threadID = new CUTThread [m_nGPUs+1];
  for(unsigned int  i = 0; i < m_nGPUs; i++)
    threadID[i] = cutStartThread((CUT_THREADROUTINE)GreedyThread, (void *)m_builders[i]);
  threadID[m_nGPUs] = cutStartThread((CUT_THREADROUTINE)SumThread, (void *)this);
  cutWaitForThreads(threadID, m_nGPUs+1);
}

void 
GreedyAtlasBuilderGPU::
GetImageLocation(int imIdx, int &gpuIdx, int &localIdx)
{
  gpuIdx = 0;
  while((uint)gpuIdx < m_nGPUs && imIdx >= (int)m_bid[gpuIdx]) gpuIdx++;
  gpuIdx--;
  localIdx = imIdx-m_bid[gpuIdx];
}

void 
GreedyAtlasBuilderGPU::
GetDeformedImage(int imIdx, RealImage &im)
{
  std::cerr << "Requesting image " << imIdx << std::endl;
  if(imIdx >= 0 && imIdx < (int)m_nImages){
    int gpuIdx=0, localIdx=0;
    GetImageLocation(imIdx, gpuIdx, localIdx);
    std::cerr << "Image " << imIdx << " being retreived from GPU " << gpuIdx << " local ID " << localIdx << std::endl;
    m_builders[gpuIdx]->getDefImage(localIdx, im);
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "Error, requesting invalid deformed image");
  }
}

void 
GreedyAtlasBuilderGPU::
GetHField(int imIdx, VectorField &vf)
{
  if(imIdx >= 0 && imIdx < (int)m_nImages){
    int gpuIdx=0, localIdx=0;
    GetImageLocation(imIdx, gpuIdx, localIdx);
    m_builders[gpuIdx]->getHField(localIdx, vf);
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "Error, requesting invalid deformation");
  }
}

void 
GreedyAtlasBuilderGPU::
GetInvHField(int imIdx, VectorField &vf)
{
  throw AtlasWerksException(__FILE__, __LINE__, 
			 "Error: GetInvHField not implemented yet!");
}

void 
GreedyAtlasBuilderGPU::
GetMeanImage(RealImage &mean)
{
  mean = *h_avgNode_i;
}
  
void 
GreedyAtlasBuilderGPU::
SetComputeInverseHFields(bool computeInverseHFields)
{
  if(computeInverseHFields){
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "Error: Computation of inverse HFields not implemented" 
			   "in GPU version of AtlasWerks!");
  }
}

bool 
GreedyAtlasBuilderGPU::
GetComputeInverseHFields()
{
  return false;
}
