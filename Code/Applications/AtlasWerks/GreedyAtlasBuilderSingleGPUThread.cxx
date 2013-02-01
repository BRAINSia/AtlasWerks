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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "GreedyAtlasBuilderSingleGPUThread.h"
#include "GreedyExecutionFFT.h"

#include <cutil.h>
#include "cutil_comfunc.h"
#include <cudaInterface.h>
#include <VectorMath.h>
#include <cudaReduce.h>
#include <cudaFFTSolver.h>
#include <cudaUpsample3D.h>
#include <cudaDownsizeFilter3D.h>
#include <cudaImage3D.h>
#include <cudaHField3DUtils.h>
#include <cudaAccumulate.h>
#include <cudaDownSample.h>

// #include <image2D.h>
// #include <gaussFilter.h>

extern pthread_barrier_t barr;

GreedyAtlasBuilderSingleGPUThread::
GreedyAtlasBuilderSingleGPUThread(int node_id, int thread_id,
                                  std::vector<const RealImage*> images,
                                  const MultiParam<GreedyAtlasScaleLevelParam> &param,
                                  RealImage* h_avgL, RealImage* h_avgG, Real* h_sqrErr, int nTotalImgs)
    :  m_nodeId(node_id),
       m_threadId(thread_id),
       m_nTotalImgs(nTotalImgs),
       mParams(param),
       h_I0(images),
       h_avgL(h_avgL),
       h_avgG(h_avgG),
       h_sqrErr(h_sqrErr),
       mUseGlobalDelta(false),
       m_delta(FLT_MAX),
       d_I0(NULL),
       d_sI0(NULL),
       d_I0t(NULL),
       d_h(NULL),
       d_I1(NULL),
       d_scratchI(NULL),
       p_Rd(NULL),
       pFFTSolver(NULL),
       d_vol(NULL)
{
    m_nImages = images.size();

    std::cout << "Node " << m_nodeId << " thread " << m_threadId << " has " 
              << m_nImages << " of " << m_nTotalImgs << " images" << std::endl;
    
    m_vSize = images[0]->getSize();
    m_vOrg  = images[0]->getOrigin();
    m_vSp   = images[0]->getSpacing();
    
    m_nVox  = m_vSize.productOfElements();

    InitHostData();
}

GreedyAtlasBuilderSingleGPUThread::
~GreedyAtlasBuilderSingleGPUThread()
{
    FreeHostData();
}

void 
GreedyAtlasBuilderSingleGPUThread::
InitHostData(){
    // allocate result arrays
    h_I0t = new RealImage*[m_nImages];
    h_hFields = new VectorField*[m_nImages];
    for(uint i=0;i<m_nImages;i++){
        h_I0t[i] = new RealImage(m_vSize, m_vOrg, m_vSp);
        h_hFields[i] = new VectorField(m_vSize);
    }
}

void 
GreedyAtlasBuilderSingleGPUThread::
FreeHostData(){
    // deallocate result arrays
    for(uint i=0;i<m_nImages;i++){
        delete h_I0t[i];
        delete h_hFields[i];
    }
    delete [] h_I0t;
    delete [] h_hFields;
}

void 
GreedyAtlasBuilderSingleGPUThread::
InitDeviceData(){
    int size = m_nVox * sizeof(float);

    // allocate device memorym
    d_I0  = new float*[m_nImages];
    d_sI0 = new float*[m_nImages];
    d_I0t = new float*[m_nImages];
    d_h   = new cplVector3DArray[m_nImages];

    std::cout << "Node " << m_nodeId << " thread " << m_threadId << " allocating device data" << std::endl;

    for (unsigned int i=0; i< m_nImages; ++i){
        allocateDeviceArray((void**)&d_I0[i], size);
        allocateDeviceArray((void**)&d_I0t[i], size);

        // allocate the scaled image 
        allocateDeviceArray((void**)&d_sI0[i], size / 8);
                
        // allocate the deformation field
        allocateDeviceVector3DArray(d_h[i], m_nVox);

        // copy data to the device
        copyArrayToDevice(d_I0[i],h_I0[i]->getDataPointer(0), m_nVox);
        
        // normalize the data to the range [0,1]
        //cplVectorOpers::AddCMulC_I(d_I0[i], -mDataInfo->v_min, 1.f / (mDataInfo->v_max - mDataInfo->v_min), m_nVox);
    }

    // Allocate common vector field
    allocateDeviceVector3DArray(d_v, m_nVox);

    // Allocate the average
    allocateDeviceArray((void**)&d_I1, size);

    // allocate template memory
    allocateDeviceVector3DArray(d_scratchV, m_nVox);
    allocateDeviceArray((void**)&d_scratchI, size);

    p_Rd       = new cplReduce();
    pFFTSolver = new FFTSolverPlan3D();
    mSm        = new cplGaussianFilter();
    
    // Initialize volume execution
    d_vol = new GreedyExecutionFFT[m_nImages];
    for (unsigned int i=0; i< m_nImages; ++i){
        d_vol[i].d_I0t      = d_I0t[i];
        d_vol[i].d_v        = d_v;
        d_vol[i].d_h        = d_h[i];
        d_vol[i].mDelta     = 0.f;
        d_vol[i].pFFTSolver = pFFTSolver;
        d_vol[i].mSx        = m_vSp.x;
        d_vol[i].mSy        = m_vSp.y;
        d_vol[i].mSz        = m_vSp.z;
    }
    checkCUDAError("GreedyAtlasBuilderSingleGPUThread:: InitDeviceData");
}

void 
GreedyAtlasBuilderSingleGPUThread::
FreeDeviceData(){
    fprintf(stderr, "Free device memory for node %d thread %d", m_nodeId, m_threadId);
  
    // deallocate device memory
    for (unsigned int i=0; i< m_nImages; ++i){
        freeDeviceArray((void*)d_I0[i]); d_I0[i] = NULL;
        freeDeviceArray((void*)d_I0t[i]); d_I0t[i] = NULL;
        freeDeviceArray((void*)d_sI0[i]); d_sI0[i] = NULL;
        freeDeviceVector3DArray(d_h[i]); 
    }
  
    delete d_I0; d_I0=NULL;
    delete d_sI0; d_sI0=NULL;
    delete d_I0t;d_I0t=NULL;
    delete d_h;
  
    freeDeviceVector3DArray(d_v);
    freeDeviceArray((void*)d_I1);
    freeDeviceVector3DArray(d_scratchV);
    freeDeviceArray((void*)d_scratchI);
  
    delete p_Rd;
    delete pFFTSolver;

    checkCUDAError("Free Device Data");
}

void 
GreedyAtlasBuilderSingleGPUThread::
BuildAtlas()
{

    std::cout << "Node " << m_nodeId << " thread " << m_threadId << " Building Atlas" << std::endl;    

    Vector3Di org_size(m_vSize.x, m_vSize.y, m_vSize.z);
    Vector3Di isize    = org_size;
    Vector3Di psize(1,1,1);

    // loop over scale levels
    for (unsigned int iL=0; iL < mParams.size(); ++iL){
        int       f   = mParams[iL].ScaleLevel().DownsampleFactor();
        int3 factor   = make_int3(f,f,f);
        float sigma   = sqrt(((float)f)/2.0);
        float3 sigmaVec = make_float3(sigma, sigma, sigma);
        int kernelSize = 2*static_cast<int>(std::ceil(sigma));
        int3 kSizeVec = make_int3(kernelSize, kernelSize, kernelSize);
        fprintf(stderr, "Level %d Scale %d \n", iL, f);
            
        //1. Down sampling the input volume
        fprintf(stderr, "Downsampling the input volume \n");
        if (f == 1)
            fprintf(stderr,"Using the actual image \n");
        else
            fprintf(stderr,"Sigma = %f \n",sigma);

        psize = isize;
        if (f == 1){
            // determine the starting input
            for (unsigned int i=0; i< m_nImages; ++i){
                d_vol[i].d_I0 = d_I0[i];
            }
            isize = org_size;
        }
        else {
            float *d_scratchI1 = d_I1;
            isize.x = org_size.x / f;
            isize.y = org_size.y / f;
            isize.z = org_size.z / f;
            fprintf(stderr,"Scaled image size %d %d %d\n", isize.x, isize.y, isize.z);

            Vector3Df sigma;
            Vector3Di kRadius;
            computeDSFilterParams(sigma, kRadius, f);
            mSm->init(org_size, sigma, kRadius);

            for (unsigned int i=0; i < m_nImages; ++i){
                // Downsampling the input
                cplDownSample(d_sI0[i], d_I0[i], isize, org_size,
                              mSm, d_scratchI, d_scratchI1);
                
                // cudaGaussianDownsample(d_sI0[i], d_I0[i],
                //                        d_scratchI, d_scratchI1,
                //                        isize, org_size,
                //                        factor, sigmaVec, kSizeVec, true);
                // determine the starting input
                d_vol[i].d_I0 = d_sI0[i];
            }
        }
        int nP = isize.x * isize.y * isize.z;
            
        //2. Create the hfield
        if (iL == 0){
            // start with identity
            fprintf(stderr, "Level 0 : set the HField to identity \n");
            for (unsigned int i=0; i< m_nImages; ++i){
                cudaHField3DUtils::setToIdentity(d_h[i], isize);
            }
        }else {
            // up sampling HField
            fprintf(stderr, "HField up sampling [%d, %d, %d], [%d, %d, %d] \n", psize.x, psize.y, psize.z, isize.x, isize.y, isize.z);
      
//       // upsample the vector fields
            SizeType inSize(psize.x, psize.y, psize.z);
            SizeType outSize(isize.x, isize.y, isize.z);
      
            for (unsigned int i=0; i< m_nImages; ++i){
                cudaHField3DUtils::resample(d_scratchV, d_h[i],
                                            outSize, inSize,
                                            BACKGROUND_STRATEGY_CLAMP, true);

                copyArrayDeviceToDevice(d_h[i], d_scratchV, nP);
            }
        }
        
        //3. Compute the atlas at this level
        //   Compute the local average
        if (iL == 0){
            // since the HField is identity initially
            // compute the average directly on the data
            if(f == 1){
                // this only happens if there is only one scale level -- iL == 0 and f == 1
                for (unsigned int i=0; i< m_nImages; ++i)
                    cudaMemcpy(d_vol[i].d_I0t, d_I0[i], nP * sizeof(float), cudaMemcpyDeviceToDevice);
            }else{
                for (unsigned int i=0; i< m_nImages; ++i)
                    cudaMemcpy(d_vol[i].d_I0t, d_sI0[i], nP * sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }
        else {
            // deform the image based on the current Hfield 
            for (unsigned int i=0; i< m_nImages; ++ i){
                d_vol[i].updateImage(isize.x, isize.y, isize.z);
            }
        }
        // initialize the solver with the size of scaled image
        pFFTSolver->setSize(Vector3Di(isize.x, isize.y, isize.z) , m_vSp);
        for (unsigned int i=0; i< m_nImages; ++i){
            d_vol[i].mStepL = mParams[iL].MaxPert();
        }
            
        // if use th individual each volume need to update the delta 
        for (unsigned int i=0; i< m_nImages; ++i)
            d_vol[i].mDelta = 0.f;

        fprintf(stderr, "GPU  %d : Run infomation %f %f %f \n",
                m_threadId,
                mParams[iL].DiffOper().Alpha(), mParams[iL].DiffOper().Beta(), mParams[iL].DiffOper().Gamma());

        // run for NIteration iterations
        for (unsigned int cnt=0; cnt< mParams[iL].NIterations(); ++cnt){
	
            // if we have weights, scale the images
            if(m_weights.size() == m_nImages){
                cplVectorOpers::SetMem(d_I1, 0.0f, nP);
                for(unsigned int i=0;i<m_nImages;i++){
                    cplVectorOpers::Add_MulC_I(d_I1, d_I0t[i], m_weights[i], nP);
                    //cplVectorOpers::AddScaledArray(d_I1, m_weights[i], d_I0t[i], nP);
                }
            }else{
                //Compute the sum 
                cplAccumulate(d_I1, d_I0t, m_nImages, nP, d_scratchI);
            }
                
            // Copy result to the host
            copyArrayFromDevice(h_avgL->getDataPointer(0), d_I1, nP);

            // Compute the total average at host
            pthread_barrier_wait(&barr);

            // Computation will occur here 

            // Synchromize the result
            pthread_barrier_wait(&barr);

            // copy back the global average
            copyArrayToDevice(d_I1, h_avgG->getDataPointer(0), nP);

            //                 if ( m_threadId == 0){
            //                     float sum = p_Rd->Sum_32f_C1(d_I1, nP);
            //                     fprintf(stderr, "Check sum of average %f \n", sum);
            //                 }
                
                
            if (mUseGlobalDelta){
                // Compute the delta value for the first iteration
                if (cnt==0){
                    for (unsigned int i=0; i< m_nImages; ++i){
                        d_vol[i].computeVectorField(d_I1, d_scratchI, d_scratchV,
                                                    p_Rd,
                                                    mParams[iL].DiffOper().Alpha(),
                                                    mParams[iL].DiffOper().Beta(),
                                                    mParams[iL].DiffOper().Gamma(),
                                                    isize.x, isize.y, isize.z, 0);

                        float localDelta = d_vol[i].computeDelta(d_v, p_Rd, d_scratchI, nP);
                        m_delta = (m_delta < localDelta) ? m_delta : localDelta;
                    }
                    pthread_barrier_wait(&barr);
                    // update the delta
                    pthread_barrier_wait(&barr);
                }
                // for each volume , perform the step
                for (unsigned int i=0; i< m_nImages; ++i){
                    float mse = d_vol[i].step(d_I1, d_scratchI, d_scratchV, p_Rd, m_delta,
                                              mParams[iL].DiffOper().Alpha(),
                                              mParams[iL].DiffOper().Beta(),
                                              mParams[iL].DiffOper().Gamma(),
                                              isize.x, isize.y, isize.z, cnt);
                    if (i ==0 && m_threadId == 0)
                        fprintf(stderr, "MSE of volume 0: %f \n", mse);
                }
                
            }
            else {
                // run the version that use the individual delta value
                float mse = 0.f;
                for (unsigned int i=0; i< m_nImages; ++i){
                    mse += d_vol[i].step(d_I1, d_scratchI, d_scratchV, p_Rd,
                                         mParams[iL].DiffOper().Alpha(),
                                         mParams[iL].DiffOper().Beta(),
                                         mParams[iL].DiffOper().Gamma(),
                                         isize.x, isize.y, isize.z, cnt);
                }
                *h_sqrErr = mse;
                mse /= static_cast<float>(m_nImages);
                //fprintf(stderr, "RMSE of volume 0: %f \n", sqrt(mse));
            }
        } // end iterate
    } // end 'for each scale level'

    // Copy out results
    for(uint imIdx=0;imIdx<m_nImages;imIdx++){
        std::cerr << "Node " << m_nodeId << " thread " << m_threadId << " moving to host image " 
                  << imIdx << " of " << m_nImages << std::endl;
        copyArrayFromDevice(h_I0t[imIdx]->getDataPointer(0), d_I0t[imIdx], m_nVox);
    }
    float *tmp = new float[m_nVox];
    for(uint imIdx=0;imIdx<m_nImages;imIdx++){
        std::cerr << "Node " << m_nodeId << " thread " << m_threadId << " moving to host deformation " 
                  << imIdx << " of " << m_nImages << std::endl;
        for(uint dim=0;dim<3;dim++){
            copyArrayFromDevice<float>(tmp, d_h[imIdx].elementArray(dim), m_nVox);
            Vector3D<float>*vecData=h_hFields[imIdx]->getDataPointer(0);
            for(uint i=0;i<m_nVox;i++){
                vecData[i][dim] = tmp[i];
            }
        }
    }
    delete tmp;
}

void 
GreedyAtlasBuilderSingleGPUThread::
getHField(int imIdx, VectorField &vf)
{
    if(imIdx >= 0 && imIdx < (int)m_nImages){
        vf = *h_hFields[imIdx];
    }else{
        throw AtlasWerksException(__FILE__, __LINE__, 
                               "Error, requesting invalid hField");
    }
}

void 
GreedyAtlasBuilderSingleGPUThread::
getDefImage(int imIdx, RealImage &im)
{
    std::cerr << "Node " << m_nodeId << " thread " << m_threadId << " retrieving deformed image " 
              << imIdx << " of " << m_nImages << std::endl;
    if(imIdx >= 0 && imIdx < (int)m_nImages){
        im = *h_I0t[imIdx];
    }else{
        throw AtlasWerksException(__FILE__, __LINE__, 
                               "Error, requesting invalid image");
    }
    std::cerr << "Node " << m_nodeId << " thread " << m_threadId << " done retrieving deformed image." << std::endl;
}
    
