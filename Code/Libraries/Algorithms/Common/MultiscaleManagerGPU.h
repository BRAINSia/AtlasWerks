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

#ifndef __MULTISCALE_MANAGER_GPU_H__
#define __MULTISCALE_MANAGER_GPU_H__

#include "MultiscaleManager.h"
#include "AtlasWerksTypes.h"
#include "cudaInterface.h"
#include "cudaVector3DArray.h"
#include "cudaHField3DUtils.h"
#include "VectorMath.h"
#include "CUDAUtilities.h"

#include <cudaGaussianFilter.h>
#include <cudaUpsample3D.h>
#include <cudaDownsizeFilter3D.h>

class MultiscaleManagerGPU : public MultiscaleManager 
{
  
public:

    /**
     * Create a new MultiscaleManagerGPU for images/fields with the given
     * native size and spacing.
     */
    MultiscaleManagerGPU(const Vector3D<unsigned int> &origSize, 
                         const Vector3D<Real> &origSpacing,
                         const Vector3D<Real> &origin = Vector3D<Real>(0,0,0));

    /**
     * Create a new MultiscaleManagerGPU for images/fields with the given
     * native size and spacing. Settings are initialized from
     * MultiscaleSettingsParam
     */
    MultiscaleManagerGPU(const Vector3D<unsigned int> &origSize, 
                         const Vector3D<Real> &origSpacing,
                         const Vector3D<Real> &origin,
                         const MultiscaleSettingsParam &param);
  
    /**
     * Create a new MultiscaleManagerGPU for images/fields with the given
     * native size and spacing. Settings are initialized from
     * MultiscaleParamInterface, including scale levels.
     */
    MultiscaleManagerGPU(const Vector3D<unsigned int> &origSize, 
                         const Vector3D<Real> &origSpacing,
                         const Vector3D<Real> &origin,
                         const MultiscaleParamInterface &param);
  
    ~MultiscaleManagerGPU();

    /** generate an empty image at the base level size, which will be
        upsampled to higher scale levels */
    float *GenerateBaseLevelImageGPU();
    /** generate a downsampled version of the image given.  When the
        scale level is changed, the image will again be downsampled to
        the new level from the original image. A pointer to the original
        image is kept -- do not delete this image while
        MultiscaleManager is still in use.*/
    float *GenerateBaseLevelImageGPU(const RealImage *origImage);
    /** generate an empty vector field at the base level size, which
        will be upsampled to higher scale levels */
    cplVector3DArray* GenerateBaseLevelVectorFieldGPU();
    /** UNIMPLEMENTED */
    cplVector3DArray* GenerateBaseLevelVectorFieldGPU(const VectorField *origVectorField);
  
    /** Get the number of voxels in the current scale level image/vector field */
    SizeType CurScaleNVox(){ 
        return CurScaleSize().productOfElements(); 
    }

    /**
     * Resample all images/vector fields to the given scale level.
     */
    virtual void SetScaleLevel(unsigned int scaleLevel);

    void UpsampleToLevel(float *image, 
                         unsigned int prevScaleLevel,
                         unsigned int nextScaleLevel);

    void UpsampleToLevel(cplVector3DArray &vf, 
                         unsigned int prevScaleLevel,
                         unsigned int nextScaleLevel);

    void DownsampleToLevel(const RealImage *orig, 
                           unsigned int scaleLevel,
                           float *downsampled);
    // UNIMPLEMENTED
    void DownsampleToLevel(const VectorField &orig, 
                           unsigned int scaleLevel,
                           cplVector3DArray &downsampled);

protected:
    void creatHelper();
        
    void InitTempVars();

    std::vector<const RealImage*> mOrigImagesGPU;

    std::vector<float*> mImagesFromOrigGPU;
    std::vector<float*> mImagesFromMinScaleGPU;

    std::vector<const VectorField*> mOrigFieldsGPU;
    std::vector<cplVector3DArray*> mFieldsFromOrigGPU;
    std::vector<cplVector3DArray*> mFieldsFromMinScaleGPU;
  
    // temp vars
    bool mTempVarsInitialized;
    cplVector3DArray d_scratchV;
    float *d_scratchI1;
    float *d_scratchI2;
    float *d_scratchI3;

    bool mCreatHelper;
    
    cplGaussianFilter  * mSm;  // Gaussian smooth Filter

    int mCurFilterScaleLevel;
    void updateSmoothFilter(const SizeType& origSize, int scaleLevel);
};

#endif
