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

// File: EstimateAffine.h (Quasi-Newton Step Method)
//
// Mark Foskey
// Peter Lorenzen
#ifndef _ESTIMATE_AFFINE_H_
#define _ESTIMATE_AFFINE_H_

#include <vector>

#include <Image.h>
#include <ROI.h>
#include <Array2D.h>

#include "AffineTransform3D.h"
#include "ImageUtils.h"
#include "Matrix.h"

class EstimateAffine {
public:
    typedef float                     VoxelType;
    typedef Image<VoxelType>          ImageType;
    typedef ImageType*                ImagePointer;
    typedef ROI<int, unsigned int>    ImageRegionType;
    typedef Vector3D<unsigned int>    ImageSizeType;
    typedef Vector3D<double>          ImageIndexType;
    typedef Array2D<double>           ScheduleType;
    typedef AffineTransform3D<double> TransformType;

    enum TransformGroupType { TRANSLATE, RIGID, AFFINE };

    enum OutputMode {SILENT, CONCISE, VERBOSE};

    // All the state associated with a given level of
    // both the translation and affine estimation
    struct EstimateAffineLevelStruct {

        // Images at the given scale
        const ImageType* fixedImage;
        const ImageType* movingImage;

        // Components of the gradient in geometric coords
        ImageType* movingImageGradientX;
        ImageType* movingImageGradientY;
        ImageType* movingImageGradientZ;

        // Factors by which the image was reduced.  
        Vector3D<double> scaleFactors;

        // Region of interest in voxels
        ImageRegionType roi;

        TransformType transformEstimate;

        // Location in geometric coordinates of the origin used for
        // applying transformations to the image during the
        // optimization.
        Vector3D<double> movingImageTempOrigin;
        Vector3D<double> fixedImageTempOrigin;

        EstimateAffineLevelStruct();
        ~EstimateAffineLevelStruct();
    };

    // Methods
    EstimateAffine( const ImageType* movingImage, 
                    const ImageType* fixedImage,
                    bool _useMinMax = false,
                    OutputMode _outputMode = CONCISE );

    ~EstimateAffine();

    void Print();
    void SetShrinkSchedule( ScheduleType shrinkSchedule );
    void SetROI( ImageRegionType roi );
    void SetInitialTransformEstimate( const TransformType& transform );
    void RunEstimateTranslation();
    void RunEstimateRotation();
    void RunEstimateAffine();

    const TransformType& GetTransform() const
    { return _finalTransformEstimate; }

    // The following convenience functions assume that the images are
    // already approximately registered, in terms of world
    // coordinates.  
    static TransformType Register( 
        TransformGroupType kindOfTransform,
        const ImageType* fixedImage,
        const ImageType* movingImage,
        float intensityWindowMax = 0,
        float intensityWindowMin = 0,
        const ImageRegionType& roi = ImageRegionType(),
        const TransformType& initialTransform = TransformType(),
        OutputMode outputMode = CONCISE );

    static TransformType RegisterAffine( 
        const ImageType* fixedImage,
        const ImageType* movingImage,
        float intensityWindowMax = 0,
        float intensityWindowMin = 0,
        const ImageRegionType& roi = ImageRegionType(),
        OutputMode outputMode = CONCISE  );

    static TransformType RegisterRigid( 
        const ImageType* fixedImage,
        const ImageType* movingImage,
        float intensityWindowMax = 0,
        float intensityWindowMin = 0,
        const ImageRegionType& roi = ImageRegionType(),
        OutputMode outputMode = CONCISE  );

    static TransformType RegisterTranslate( 
        const ImageType* fixedImage,
        const ImageType* movingImage,
        float intensityWindowMax = 0,
        float intensityWindowMin = 0,
        const ImageRegionType& roi = ImageRegionType(),
        OutputMode outputMode = CONCISE  );

    static void 
    ApplyTransformation(const ImageType* fixedImage,
                        const ImageType* movingImage,
                        ImageType* newImage,
                        const AffineTransform3D<double>& transform,
                        const float& backgroundValue = 0);

    static ScheduleType GenerateScheduleFromImage( const ImageType* image );

    void CreateRegisteredImage( ImagePointer newImage,
				const bool setBackgroundToFirstVoxel = false );

    void SetMinMax( float minFixedImage, float maxFixedImage, 
                    float minMovingImage, float maxMovingImage)
    {
        _minFixedImage = minFixedImage;
        _maxFixedImage = maxFixedImage;
        _minMovingImage = minMovingImage;
        _maxMovingImage = maxMovingImage;
    }

private:

    // Levels
    std::vector< EstimateAffineLevelStruct > _levels;
    int _InitializeLevels();
    void _InitializeScaleFactors();
    void _InitializeROIs();
    void _InitializeImages();
    void _InitializeMovingImageGradients();
    void _InitializeMovingImageTempOrigins();

    // Estimates
    TransformType _initialTransformEstimate;
    TransformType _previousTransformEstimate;
    TransformType _finalTransformEstimate;

    // Region of interest
    ImageRegionType _baseROI;

    // Image data
    const ImageType* _movingImage; // Input image
    const ImageType* _fixedImage;  // Input image

    float _minFixedImage,_minMovingImage,_maxFixedImage,_maxMovingImage;

    // Supporting
    float _StepEstimateTranslation( unsigned int level );
    float _StepEstimateRotation( unsigned int level, float epsilon );
    float _StepEstimateAffine( unsigned int level );

    void _RunEstimateTransformLevel( unsigned int level, 
                                     unsigned int numberOfLevels,
                                     TransformGroupType transformType );

    void _SetEstimateTranslationLevel( unsigned int level );
    void _SetEstimateAffineLevel( unsigned int level );

    void _RunEstimateTransform( TransformGroupType transformType );

    Vector3D<double> _ComputeImageBoxTempOrigin( const ImageType* image );

    void _ComputeInitialMatrixEstimate();
    Matrix _ComputeMatrixSquareRoot( Matrix input );
    void _ConvertGeneralLinearToRotation( Matrix3D<double>& matrix );
    void _ComputeGradientComponent( const ImageType* image, int direction,
                                    ImagePointer gradient );
    void _ComputeShrinkImage(
        const ImageType* image,       // image to be shrunk
        ImageSizeType scaleFactors,   // Multiply dims by 1/factor
        ImagePointer shrinkImage );   // Must allocate image yourself
    void _ComputeShrinkImageWithAveraging( const ImageType* image,
                                           ImageSizeType scaleFactors,
                                           ImagePointer shrinkImage );
    void _ComputeShrinkImageWithGaussian( ImageType* image,
                                          ImageSizeType scaleFactors,
                                          ImagePointer shrinkImage );
    void _ComputeShrinkImageWithoutAveraging( const ImageType* image,
                                              ImageSizeType scaleFactors,
                                              ImagePointer shrinkImage );

    static TransformType _TempOriginCoordinatesToWorldCoordinates(
        const ImageType* fixedImage, const ImageType* movingImage,
        const Vector3D<double>& fixedImageTempOrigin,
        const Vector3D<double>& movingImageTempOrigin,
        const TransformType& transform );

    static TransformType _WorldCoordinatesToTempOriginCoordinates(
        const ImageType* fixedImage, const ImageType* movingImage,
        const Vector3D<double>& fixedImageTempOrigin,
        const Vector3D<double>& movingImageTempOrigin,
        const TransformType& transform );

    void _ConvertIndexToTempOriginCoordinates( 
        const Vector3D<double>& spacing, const Vector3D<double>& center,
        const size_t& xIndex, const size_t& yIndex, const size_t& zIndex,
        double& xPhysical, double& yPhysical, double& zPhysical);

    //
    // Parameters
    //
    bool _useMinMax;
    OutputMode _outputMode;
    ScheduleType _shrinkSchedule;

    // Misc.
    float _initialMeanSquareError;
}; // class EstimateAffine

#endif // #ifndef _ESTIMATE_AFFINE_H_
