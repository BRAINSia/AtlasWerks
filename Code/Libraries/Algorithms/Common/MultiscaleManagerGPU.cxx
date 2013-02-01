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

#include "MultiscaleManagerGPU.h"
#include "CUDAUtilities.h"
//#include "gaussFilter.h"
#include "cudaGaussianFilter.h"
#include "cudaImage3D.h"

#include <cudaDownSample.h>
MultiscaleManagerGPU::
MultiscaleManagerGPU(const Vector3D<unsigned int> &origSize, 
                     const Vector3D<Real> &origSpacing,
                     const Vector3D<Real> &origin) :
    MultiscaleManager(origSize, origSpacing, origin),
    mTempVarsInitialized(false), mCreatHelper(false), mCurFilterScaleLevel(-1)
{
    
}

MultiscaleManagerGPU::
MultiscaleManagerGPU(const Vector3D<unsigned int> &origSize, 
                     const Vector3D<Real> &origSpacing,
                     const Vector3D<Real> &origin,
                     const MultiscaleSettingsParam &param) :
    MultiscaleManager(origSize, origSpacing, origin, param),
    mTempVarsInitialized(false), mCreatHelper(false), mCurFilterScaleLevel(-1)
{
}

MultiscaleManagerGPU::
MultiscaleManagerGPU(const Vector3D<unsigned int> &origSize, 
                     const Vector3D<Real> &origSpacing,
                     const Vector3D<Real> &origin,
                     const MultiscaleParamInterface &param) :
    MultiscaleManager(origSize, origSpacing, origin, param),
    mTempVarsInitialized(false), mCreatHelper(false), mCurFilterScaleLevel(-1)
{
    
}

MultiscaleManagerGPU::
~MultiscaleManagerGPU()
{
    for(unsigned int i=0;i<mImagesFromOrigGPU.size();i++){
        freeDeviceArray(mImagesFromOrigGPU[i]);
    }

    for(unsigned int i=0;i<mFieldsFromOrigGPU.size();i++){
        freeDeviceVector3DArray(*mFieldsFromOrigGPU[i]);
    }

    for(unsigned int i=0;i<mImagesFromMinScaleGPU.size();i++){
        freeDeviceArray(mImagesFromOrigGPU[i]);
    }
    for(unsigned int i=0;i<mFieldsFromMinScaleGPU.size();i++){
        freeDeviceVector3DArray(*mFieldsFromMinScaleGPU[i]);
    }
    
    if(mTempVarsInitialized){
        freeDeviceVector3DArray(d_scratchV);
        freeDeviceArray(d_scratchI1);
        freeDeviceArray(d_scratchI2);
        freeDeviceArray(d_scratchI3);
    }
}

void 
MultiscaleManagerGPU::
InitTempVars()
{
    if(mTempVarsInitialized) return;

    int nVox = mScaleLevels[this->NumberOfScaleLevels()-1].Size.productOfElements();
    allocateDeviceArray((void**)&d_scratchI1, nVox*sizeof(Real));
    allocateDeviceArray((void**)&d_scratchI2, nVox*sizeof(Real));
    allocateDeviceArray((void**)&d_scratchI3, nVox*sizeof(Real));
    allocateDeviceVector3DArray(d_scratchV, nVox);
    mTempVarsInitialized = true;

    creatHelper();
}

void
MultiscaleManagerGPU::creatHelper()
{
    if (mCreatHelper)
        return;

    mSm = new cplGaussianFilter();
    mCreatHelper = true;
}
float*
MultiscaleManagerGPU::
GenerateBaseLevelImageGPU()
{
    // we assume this image will be upsampled to the maximum resolution,
    // so we allocate the maximum size needed -- we'll only use part of
    // it for the lower scale levels

    // number of voxels at the max resolution
    int nVox = mScaleLevels[this->NumberOfScaleLevels()-1].Size.productOfElements();
    int memSize = sizeof(Real)*nVox;
    float *d_image;
    allocateDeviceArray((void**)&d_image, memSize);
    cplVectorOpers::SetMem<float>(d_image, 0, nVox);
    mImagesFromMinScaleGPU.push_back(d_image);
    return d_image;
}

float*
MultiscaleManagerGPU::
GenerateBaseLevelImageGPU(const RealImage *origImage)
{
    mOrigImages.push_back(origImage);
  
    // Allocate memory for whole image.  At lower scale levels not all
    // of this will be used.
    int nVox = origImage->getSize().productOfElements();
    int memSize = sizeof(Real)*nVox;
    float *d_image;
    allocateDeviceArray((void**)&d_image, memSize);

    // downsample to lowest level
    DownsampleToLevel(origImage, NumberOfScaleLevels()-1, d_image);
    mImagesFromOrigGPU.push_back(d_image);
    return d_image;
}

cplVector3DArray*
MultiscaleManagerGPU::
GenerateBaseLevelVectorFieldGPU()
{
    // we assume this vector field will be upsampled to the maximum
    // resolution, so we allocate the maximum size needed -- we'll only
    // use part of it for the lower scale levels

    // number of voxels at the max resolution
    int nVox = mScaleLevels[this->NumberOfScaleLevels()-1].Size.productOfElements();
    cplVector3DArray *d_vf = new cplVector3DArray();
    allocateDeviceVector3DArray(*d_vf, nVox);
    cplVector3DOpers::SetMem(*d_vf, 0, nVox);
    mFieldsFromMinScaleGPU.push_back(d_vf);
    return d_vf;
}

cplVector3DArray*
MultiscaleManagerGPU::
GenerateBaseLevelVectorFieldGPU(const VectorField *origVF)
{

    mOrigFields.push_back(origVF);
  
    // Allocate space for the original image on the device
    int nVox = origVF->getSize().productOfElements();
    cplVector3DArray *d_vf = new cplVector3DArray();
    allocateDeviceVector3DArray(*d_vf, nVox);

    // downsample
    //InitTempVars();
    //CUDAUtilities::CopyVectorFieldToDevice(*origVF, d_scratchV, h_scratchI);
    DownsampleToLevel(*origVF, NumberOfScaleLevels()-1, *d_vf);
    mFieldsFromOrigGPU.push_back(d_vf);

    return d_vf;
}

#include <cutil_inline.h>
void MultiscaleManagerGPU::updateSmoothFilter(const SizeType& origSize, int scaleLevel){
    if (mCurFilterScaleLevel == scaleLevel) return;
    mCurFilterScaleLevel = scaleLevel;
    Vector3Df sigma;
    Vector3Di kRadius;
    computeDSFilterParams(sigma, kRadius, mScaleLevels[mCurFilterScaleLevel].DownsampleFactor);
    mSm->init(origSize, sigma, kRadius);
    cutilCheckMsg("updateSmoothFilter");
}

void
MultiscaleManagerGPU::
SetScaleLevel(unsigned int scaleLevel){
    // don't have to do anything if we're already at this scale level
    if((int)scaleLevel == mCurScaleLevel) return;

    unsigned int lastScaleLevel = mCurScaleLevel;
    
    
    // call superclass version
    MultiscaleManager::SetScaleLevel(scaleLevel);

    for(unsigned int i=0;i<mImagesFromOrigGPU.size();i++){
        DownsampleToLevel(mOrigImages[i],mCurScaleLevel,mImagesFromOrigGPU[i]);
    }
    for(unsigned int i=0;i<mFieldsFromOrigGPU.size();i++){
        DownsampleToLevel(*mOrigFields[i],mCurScaleLevel,*mFieldsFromOrigGPU[i]);
    }
    for(unsigned int i=0;i<mImagesFromMinScaleGPU.size();i++){
        UpsampleToLevel(mImagesFromMinScaleGPU[i], lastScaleLevel, mCurScaleLevel);
    }
    for(unsigned int i=0;i<mFieldsFromMinScaleGPU.size();i++){
        UpsampleToLevel(*mFieldsFromMinScaleGPU[i], lastScaleLevel, mCurScaleLevel);
    }

}

void 
MultiscaleManagerGPU::
UpsampleToLevel(float *image,
                unsigned int prevScaleLevel,
                unsigned int nextScaleLevel)
{

    if(nextScaleLevel <= 0 || nextScaleLevel >= NumberOfScaleLevels()){
        throw MultiscaleException(__FILE__, __LINE__, 
                                  "Error, invalid scale level to upsample to:" + nextScaleLevel );
    }
    SizeType origSize = mScaleLevels[prevScaleLevel].Size;
    SizeType newSize = mScaleLevels[nextScaleLevel].Size;
    unsigned int nVox = newSize.productOfElements();

    std::cerr << "Upsampling image to " << newSize << std::endl;
    if(mUseSincImageUpsample){
        throw MultiscaleException(__FILE__, __LINE__, 
                                  "Sinc upsampling not supported on GPU");
    }else{
        InitTempVars();
        cplUpsample(d_scratchI1, image,
                    newSize, origSize, BACKGROUND_STRATEGY_CLAMP, true);
        // resample(d_scratchI1, image,
        //          newSize.x, newSize.y, newSize.z,
        //          origSize.x, origSize.y, origSize.z,
        //          BACKGROUND_STRATEGY_CLAMP,
        //          true);
        copyArrayDeviceToDevice(image, d_scratchI1, nVox);
    }
}

void
MultiscaleManagerGPU::
UpsampleToLevel(cplVector3DArray &d_vf, 
                unsigned int prevScaleLevel,
                unsigned int nextScaleLevel)
{
    if(nextScaleLevel <= 0 || nextScaleLevel >= NumberOfScaleLevels()){
        throw MultiscaleException(__FILE__, __LINE__, 
                                  "Error, invalid scale level to upsample to:" + nextScaleLevel );
    }
    SizeType origSize = mScaleLevels[prevScaleLevel].Size;
    SizeType newSize = mScaleLevels[nextScaleLevel].Size;
    int nVox = newSize.productOfElements();
    std::cerr << "Upsampling hField to " << newSize << std::endl;
    // make sure d_scratchV is initialized
    InitTempVars();
    // Up sampling current vector field to the next level
    cudaHField3DUtils::resample(d_scratchV, d_vf,
                                newSize, origSize,
                                BACKGROUND_STRATEGY_CLAMP, 
				mScaleVectorFields);
    copyArrayDeviceToDevice(d_vf, d_scratchV, nVox);
}

void 
MultiscaleManagerGPU::
DownsampleToLevel(const RealImage *orig, 
                  unsigned int scaleLevel,
                  float *downsampled)
{
    if(scaleLevel < 0 || scaleLevel >= NumberOfScaleLevels()){
        std::cerr << "Error, invalid scale level to downsample to:" << scaleLevel << std::endl;
        return;
    }

    SizeType origSize = orig->getSize();
    SizeType newSize  = mScaleLevels[scaleLevel].Size;
    int nVox          = newSize.productOfElements();
    int f             = mScaleLevels[scaleLevel].DownsampleFactor;
    if(f == 1){
        std::cerr << "Using original image, size " << newSize << std::endl;
        copyArrayToDevice(downsampled, orig->getDataPointer(), nVox);
    }else{
        std::cerr << "Downsampling image to " << newSize << std::endl;
        InitTempVars();
        copyArrayToDevice(d_scratchI1, orig->getDataPointer(), origSize.productOfElements());
#if 1
        this->updateSmoothFilter(origSize, scaleLevel);
        cplDownSample(downsampled, d_scratchI1,
                      newSize, origSize,
                      mSm, d_scratchI2, d_scratchI3);
#else
        // Old implementation 
        int3 osize = make_int3(origSize.x, origSize.y, origSize.z);
        int3 csize = make_int3(newSize.x, newSize.y, newSize.z);
        int3  factor = make_int3(f,f,f);
        float3 sigma = make_float3(static_cast<float>(sqrt(factor.x/2.0)),
                                   static_cast<float>(sqrt(factor.y/2.0)),
                                   static_cast<float>(sqrt(factor.z/2.0)));
        int3 kRadius = make_int3(2*static_cast<int>(std::ceil(sigma.x)),
                                 2*static_cast<int>(std::ceil(sigma.y)),
                                 2*static_cast<int>(std::ceil(sigma.z)));
        cudaGaussianDownsample(downsampled, d_scratchI1, d_scratchI2,  d_scratchI3,
                               csize, osize, factor, sigma, kRadius, true);
#endif
    }
}

// UNIMPLEMENTED  
void 
MultiscaleManagerGPU::
DownsampleToLevel(const VectorField &orig, 
                  unsigned int scaleLevel,
                  cplVector3DArray &downsampled)
{
    throw(MultiscaleException(__FILE__, __LINE__, "DownsampleToLevel unimplemented for VectorFields"));
}

