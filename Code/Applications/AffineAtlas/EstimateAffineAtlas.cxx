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

// File EstimateAffineiAtlas.cxx
//
// A quasi-Newton step method for estimating the translation between to two
// images as well as the affine between the images.  There is an option to
// restrict the affine matrix estimated to be a rotation.
//
// Notes:
//   - The coarsest level in the multi-resolution schedule has the
//     lowest number (0).
//
// Mark Foskey, Peter Lorenzen

#ifdef WIN32
#pragma warning (disable: 4786) // truncated debug info
#endif

#include <cmath>
#include <iostream>

#include "Timer.h"
#include "EstimateAffineAtlas.h"
#include "DownsampleFilter3D.h"
#include "Matrix3D.h"

#include <Array3DUtils.h>
#include <Matrix3DUtils.h>

//#include <Debugging.h>

std::ostream& operator<<(std::ostream& os, 
                         const EstimateAffineAtlas::EstimateAffineAtlasLevelStruct& eals)
{
    os << "\nMoving image dimensions: " << eals.movingImage->getSize()
       << "\nFixed image dimensions: " << eals.fixedImage->getSize()
       << "\nScale factors: " << eals.scaleFactors
       << "\nROI start: " << eals.roi.getStart()
       << "\nROI stop: " << eals.roi.getStop()
       << "\nTransform estimate: " << eals.transformEstimate
       << "\nMoving temp origin: " << eals.movingImageTempOrigin
       << "\nFixed temp origin: " << eals.fixedImageTempOrigin;
    return os;
}

void EstimateAffineAtlas::
Print()
{
    for (unsigned int i = 0; i < _levels.size(); ++i) {
        std::cout << "\n\nLevel " << i << ":" << _levels[i] << "\n";
    }
}

//-------------------------------------------------------------------------
// {Con,De}structor for EstimateAffine

// EstimateAffine::EstimateAffine
EstimateAffineAtlas::EstimateAffineAtlas(const ImageType* fixedImage,
                               const ImageType* movingImage,
                               bool useMinMax, OutputMode outputMode)
{
    // Data
    _movingImage = movingImage;
    _fixedImage = fixedImage;
    _minFixedImage = 0;
    _minMovingImage = 0;
    _maxFixedImage = 4096;
    _maxMovingImage = 4096;
    _useMinMax = useMinMax;
    _outputMode = outputMode;

    // Initialize ROI
    ImageSizeType fixedImageSize =
        (Vector3D<double>)(_fixedImage->getSize()-1);

    //start initialize at 0
    _baseROI.setSize(fixedImageSize);

    _initialMeanSquareError = -1;

} // EstimateAffine::EstimateAffine


// EstimateAffine::~EstimateAffine
EstimateAffineAtlas::~EstimateAffineAtlas()
{
} // EstimateAffine::~EstimateAffine

//-------------------------------------------------------------------------
// {Con,De}structor for level struct

EstimateAffineAtlas::EstimateAffineAtlasLevelStruct::
EstimateAffineAtlasLevelStruct()
    : fixedImage(NULL), 
      movingImage(NULL), 
      movingImageGradientX(NULL),
      movingImageGradientY(NULL),
      movingImageGradientZ(NULL)
{}

EstimateAffineAtlas::EstimateAffineAtlasLevelStruct::
~EstimateAffineAtlasLevelStruct()
{
    delete movingImage; 
    delete fixedImage; 
    delete movingImageGradientX; 
    delete movingImageGradientY; 
    delete movingImageGradientZ; 
}

//--------------------------------------------------------------

// Return the affine transform that best matches the moving image to
// the fixed image, by minimizing the root-mean-square of 
//
//    fixedImage(voxel) - movingImage(T(voxel))
//
// where T is the transform returned.  Note that the moving image is
// actually transformed by T^-1.

EstimateAffineAtlas::TransformType EstimateAffineAtlas::
Register( TransformGroupType kindOfTransform, 
          const ImageType* fixedImage,
          const ImageType* movingImage,
          float intensityWindowMax,
          float intensityWindowMin,
          const ImageRegionType& roi,
          const TransformType& initialTransform,
          OutputMode outputMode )
{
    bool useIntensityWindowing = (intensityWindowMax > intensityWindowMin);
    EstimateAffineAtlas estimateAffine( fixedImage, movingImage,
                                   useIntensityWindowing, outputMode );
    estimateAffine.SetShrinkSchedule( 
        EstimateAffineAtlas::GenerateScheduleFromImage( fixedImage ));
    if (roi.getSize() != Vector3D<unsigned int>(0,0,0)) {
        estimateAffine.SetROI( roi );
    }
    if (useIntensityWindowing) {
        estimateAffine.SetMinMax( intensityWindowMin, intensityWindowMax,
                                  intensityWindowMin, intensityWindowMax );
    }
    estimateAffine.SetInitialTransformEstimate( initialTransform );

    // Where the action happens
    estimateAffine._RunEstimateTransform( kindOfTransform );

    return estimateAffine.GetTransform();
}

EstimateAffineAtlas::TransformType EstimateAffineAtlas::
RegisterAffine( const ImageType* fixedImage, const ImageType* movingImage,
                float intensityWindowMax, float intensityWindowMin,
                const ImageRegionType& roi, OutputMode outputMode )
{
    return Register( AFFINE, fixedImage, movingImage, intensityWindowMax, 
                     intensityWindowMin, roi, TransformType(), outputMode );
}

EstimateAffineAtlas::TransformType EstimateAffineAtlas::
RegisterRigid( const ImageType* fixedImage, const ImageType* movingImage,
               float intensityWindowMax, float intensityWindowMin,
               const ImageRegionType& roi, OutputMode outputMode )
{
    return Register( RIGID, fixedImage, movingImage, intensityWindowMax, 
                     intensityWindowMin, roi, TransformType(), outputMode );
}

EstimateAffineAtlas::TransformType EstimateAffineAtlas::
RegisterTranslate( const ImageType* fixedImage, const ImageType* movingImage,
                   float intensityWindowMax, float intensityWindowMin,
                   const ImageRegionType& roi, OutputMode outputMode )
{
    return Register( TRANSLATE, fixedImage, movingImage, intensityWindowMax, 
                     intensityWindowMin, roi, TransformType(), outputMode );
}

//--------------------------------------------------------------

void EstimateAffineAtlas::
ApplyTransformation(const ImageType* fixedImage,
                    const ImageType* movingImage,
                    ImageType* newImage,
                    const AffineTransform3D<double>& transform,
                    const float& backgroundValue)
{
    newImage->resize( fixedImage->getSize() );
    newImage->setSpacing( fixedImage->getSpacing() );
    newImage->setOrigin( fixedImage->getOrigin() );
    ImageUtils::applyAffine( *movingImage, *newImage, 
                             transform, backgroundValue );
}

void EstimateAffineAtlas::
CreateRegisteredImage( ImagePointer newImage, 
		       const bool setBackgroundToFirstVoxel )
{
    Timer timer;
    if (_outputMode == VERBOSE) {
        std::cout << "CreateRegisteredImage()" << std::endl;
        timer.start();
    }

    
    float backgroundValue = 0;
    if (setBackgroundToFirstVoxel) {
      backgroundValue = (*_fixedImage)(0, 0, 0);
    }
    ApplyTransformation( _fixedImage, _movingImage, newImage, 
			 _finalTransformEstimate, backgroundValue );

    if (_outputMode == VERBOSE) {
        timer.stop();
        std::cout << "CreateRegisteredImage() took "
                  << float(timer.getMilliseconds()) / 1000.0f 
                  << " seconds." << std::endl;
    }
}
    

EstimateAffineAtlas::ScheduleType EstimateAffineAtlas::
GenerateScheduleFromImage( const ImageType* image )
{
    ImageSizeType size = image->getSize();
    ImageSizeType factors(1, 1, 1);
    std::vector< ImageSizeType > scheduleVec;
    scheduleVec.push_back(factors);
    while (true) {
        bool factorsChanged = false;
        unsigned int newFactor;
        for (unsigned int i = 0; i < 3; ++i) {
            newFactor = factors[i] * 2;
            if (size[i] / newFactor > 32) {
                factors[i] = newFactor;
                factorsChanged = true;
            }
        }

        if (!factorsChanged) break;

        scheduleVec.push_back(factors);
    }
    ScheduleType schedule( scheduleVec.size(), 3 );
    for (unsigned int i = 0; i < schedule.getSizeX(); ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            schedule(i, j) = scheduleVec[scheduleVec.size()-i-1][j];
        }
    }
    return schedule;
}

// EstimateAffine::SetShrinkSchedule
void
EstimateAffineAtlas::SetShrinkSchedule( ScheduleType shrinkSchedule )
{
    _shrinkSchedule = shrinkSchedule;
}


void
EstimateAffineAtlas::SetROI( ImageRegionType roi)
{
    _baseROI.setStart(roi.getStart());
    _baseROI.setSize(roi.getSize());
    ImageRegionType::ROIIndexType stop = _baseROI.getStop();
    const ImageSizeType& imageSize = _fixedImage->getSize();
    for (unsigned int i = 0; i < 3; ++i) {
        if (stop[i] >= (int) imageSize[i]) {
            stop[i] = (int) imageSize[i] - 1;
        }
    }
    _baseROI.setStop( stop );
}

//==============================================================
// Main driver routines

void
EstimateAffineAtlas::RunEstimateTranslation()
{ _RunEstimateTransform( TRANSLATE ); }

void
EstimateAffineAtlas::RunEstimateRotation()
{ _RunEstimateTransform( RIGID ); }

void
EstimateAffineAtlas::RunEstimateAffine()
{ _RunEstimateTransform( AFFINE ); }

void
EstimateAffineAtlas::_RunEstimateTransform( TransformGroupType transformType )
{

    // Set up diagnostic output
    Timer timer;
    if (_outputMode == VERBOSE) {
        //std::cout << "_RunEstimateTransform type " 
                 // << transformType << std::endl;
        timer.start();
    }
    if (_outputMode != SILENT) {
        //std::cout << "Lev It  RMS error   % change" << std::endl;
    }

    // Do the work
    unsigned int numberOfLevels = _InitializeLevels();
    _RunEstimateTransformLevel( 0, numberOfLevels, TRANSLATE );
    if (transformType != TRANSLATE) {
        _RunEstimateTransformLevel( 0, numberOfLevels, transformType );
    }

    // Clear _levels data structure.
    _levels = std::vector< EstimateAffineAtlasLevelStruct >();

    // Concluding diagnostic output
    if (_outputMode == VERBOSE) {
        timer.stop();
        //std::cout << "_RunEstimateTransform took "
                 // << float(timer.getMilliseconds()) / 1000.0f 
                 // << " seconds." << std::endl;
    }
}

//==============================================================

// Perform a change of coordinates so that a transform expressed in
// terms of the specified image centers is re-expressed in terms of
// the image origins.
EstimateAffineAtlas::TransformType EstimateAffineAtlas::
_TempOriginCoordinatesToWorldCoordinates(
    const ImageType* fixedImage, const ImageType* movingImage,
    const Vector3D<double>& fixedImageTempOrigin,
    const Vector3D<double>& movingImageTempOrigin,
    const TransformType& transform ) 
{
    Vector3D<double> C1 = movingImageTempOrigin - movingImage->getOrigin();
    Vector3D<double> C2 = fixedImageTempOrigin - fixedImage->getOrigin();
    TransformType fixedTransform = transform;
    fixedTransform.vector = transform.vector - C1 + transform.matrix * C2;
    return fixedTransform;
}

EstimateAffineAtlas::TransformType EstimateAffineAtlas::
_WorldCoordinatesToTempOriginCoordinates(
    const ImageType* fixedImage, const ImageType* movingImage,
    const Vector3D<double>& fixedImageTempOrigin,
    const Vector3D<double>& movingImageTempOrigin,
    const TransformType& transform ) 
{
    Vector3D<double> C1 = -movingImageTempOrigin + movingImage->getOrigin();
    Vector3D<double> C2 = -fixedImageTempOrigin + fixedImage->getOrigin();
    TransformType fixedTransform = transform;
    fixedTransform.vector = transform.vector - C1 + transform.matrix * C2;
    return fixedTransform;
}

// Like Image::imageIndexToWorldCoordinates() except that the image
// spacing and origin are replaced by the 'spacing' and 'tempOrigin'
// parameters.  The main use is to change the origin temporarily, not
// the spacing.
void EstimateAffineAtlas::
_ConvertIndexToTempOriginCoordinates( 
    const Vector3D<double>& spacing, const Vector3D<double>& tempOrigin,
    const size_t& xIndex, const size_t& yIndex, const size_t& zIndex,
    double& xPhysical, double& yPhysical, double& zPhysical)
{
    xPhysical = (double)xIndex * spacing[0] + tempOrigin[0];
    yPhysical = (double)yIndex * spacing[1] + tempOrigin[1];
    zPhysical = (double)zIndex * spacing[2] + tempOrigin[2];
}

// Return a vector in world units pointing from the center of the
// image to the center of the (0,0,0) voxel.  This vector would be the
// image origin if the world coordinate system were centered in the
// middle of the image volume.
Vector3D<double> EstimateAffineAtlas::
_ComputeImageBoxTempOrigin( const ImageType* image )
{
    const ImageType::SizeType& size = image->getSize();
    const ImageType::SpacingType& spacing = image->getSpacing();
    return -spacing * size / 2.0;
}

int
EstimateAffineAtlas::_InitializeLevels()
{
    unsigned int numberOfLevels = _shrinkSchedule.getSizeX();
    _levels.clear();
    _levels.insert( _levels.end(), numberOfLevels, EstimateAffineAtlasLevelStruct() );
    _InitializeScaleFactors();
    _InitializeROIs();
    _InitializeImages();
    _InitializeMovingImageGradients();
    _InitializeMovingImageTempOrigins();
    _levels[0].transformEstimate = 
        _WorldCoordinatesToTempOriginCoordinates( 
            _levels[0].fixedImage,
            _levels[0].movingImage,
            _levels[0].fixedImageTempOrigin,
            _levels[0].movingImageTempOrigin,
            _initialTransformEstimate );
    return numberOfLevels;
}


void EstimateAffineAtlas::
_InitializeScaleFactors()
{
    for( unsigned int level = 0; level < _levels.size(); level++ ) {
        for (unsigned int dim = 0; dim < 3; ++dim) {
            _levels[level].scaleFactors[dim] = _shrinkSchedule(level,dim);
        }
    }
}


// EstimateAffine::_InitializeROIs
void
EstimateAffineAtlas::_InitializeROIs()
{
    // For each level
    for( unsigned int level = 0; level < _levels.size(); level++ ) {

        Vector3D<double> StartScaled;
        Vector3D<double> StopScaled;

        for( unsigned int dim = 0; dim < 3; dim++ ) {
            StartScaled[dim] = (int)((_baseROI.getStart())[dim] / 
                                     (int)_levels[level].scaleFactors[dim]);
            StopScaled[dim] = (int)(((_baseROI.getStop())[dim] + 1) / 
                                    (int)_levels[level].scaleFactors[dim]) - 1;
        }
        (_levels[level].roi).setStart(StartScaled);
        (_levels[level].roi).setStop(StopScaled);
    }
} // EstimateAffine::_InitializeROIs


void
EstimateAffineAtlas::_InitializeImages()
{
    Timer timer;

    // Make copies of moving and fixed images.
    if (_outputMode == VERBOSE) {
        std::cout << "Copying images..." << std::flush;
        timer.start();
    }

    ImageType* movingImageCopy = new ImageType(*_movingImage);
    ImageType* fixedImageCopy = new ImageType(*_fixedImage);

    if (_outputMode == VERBOSE) {
        timer.stop();
        std::cout << timer.getMilliseconds() << " msec." << std::endl;
    }

    timer.start();
    if (_useMinMax) {

        if (_outputMode == VERBOSE) {
            std::cout << "Rescaling images..." << std::flush;
        }

        Array3DUtils::rescaleElements( *fixedImageCopy, _minFixedImage, 
                                       _maxFixedImage, 0.0f, 1.0f );
        Array3DUtils::rescaleElements( *movingImageCopy, _minMovingImage, 
                                       _maxMovingImage, 0.0f, 1.0f );
        _minMovingImage = _minFixedImage = 0.0f;
        _maxMovingImage = _maxFixedImage = 1.0f;
    } else {

        if (_outputMode == VERBOSE) {
            std::cout << "Finding max, min..." << std::flush;
        }

        Array3DUtils::getMinMax(*fixedImageCopy, _minFixedImage, _maxFixedImage);
        Array3DUtils::getMinMax(*movingImageCopy, _minMovingImage, 
                                _maxMovingImage);
    }
    timer.stop();
    if (_outputMode == VERBOSE) {
        std::cout << timer.getMilliseconds() << " msec." << std::endl;
    }

    if (_outputMode == VERBOSE) {
        std::cout << "Downsampling images..." << std::flush;
    }
    timer.start();
    for( unsigned int level = 0; level < _levels.size(); level++ ) {
        Vector3D<double>& currentScaleFactors = _levels[level].scaleFactors;
        if (currentScaleFactors == Vector3D<double>(1, 1, 1)) {
            _levels[level].movingImage = movingImageCopy; 
            _levels[level].fixedImage = fixedImageCopy; 
            // No point in multiple levels at full resolution.
            _levels.erase(_levels.begin() + level + 1, _levels.end());
        } else {
            ImageType* movingImage = new ImageType;
            _ComputeShrinkImage( movingImageCopy, currentScaleFactors,
                                 movingImage );
            _levels[level].movingImage = movingImage;
            ImageType* fixedImage = new ImageType;
            _ComputeShrinkImage( fixedImageCopy, currentScaleFactors,
                                 fixedImage );
            _levels[level].fixedImage = fixedImage;
        }
    }
    timer.stop();
    if (_outputMode == VERBOSE) {
        std::cout << timer.getMilliseconds() << " msec." << std::endl;
    }
}


void
EstimateAffineAtlas::_InitializeMovingImageGradients()
{
    Timer timer;
    if (_outputMode == VERBOSE) {
        std::cout << "_InitializeMovingImageGradients..." << std::flush;
    }
    timer.start();

    for( unsigned int level = 0; level < _levels.size(); level++ ) {

        _levels[level].movingImageGradientX = new ImageType;
        _ComputeGradientComponent( _levels[level].movingImage, 0,
                                   _levels[level].movingImageGradientX );

        _levels[level].movingImageGradientY = new ImageType;
        _ComputeGradientComponent( _levels[level].movingImage, 1,
                                   _levels[level].movingImageGradientY );

        _levels[level].movingImageGradientZ = new ImageType;
        _ComputeGradientComponent( _levels[level].movingImage, 2,
                                   _levels[level].movingImageGradientZ );

    }

    timer.stop();
    if (_outputMode == VERBOSE) {
        std::cout << timer.getMilliseconds() << " msec." << std::endl;
    }
}


void
EstimateAffineAtlas::_InitializeMovingImageTempOrigins()
{
    if (_outputMode == VERBOSE) {
        //std::cout << "Computing moving image temp origins..." << std::flush;
    }

    Timer timer;
    timer.start();

    // For each level
    for( unsigned int level = 0; level < _levels.size(); level++ ) {
        _levels[level].fixedImageTempOrigin = 
            _ComputeImageBoxTempOrigin( _levels[level].fixedImage );
        _levels[level].movingImageTempOrigin = 
            _ComputeImageBoxTempOrigin( _levels[level].movingImage );
    }

    timer.stop();
    if (_outputMode == VERBOSE) {
        //std::cout << timer.getMilliseconds() << " msec." << std::endl;
    }
} // EstimateAffine::_InitializeMovingImageTempOrigins


// Compute an image consisting of one component of the gradient of the
// parameter 'image'.  'direction' indicates which component is being
// computed, with 0 indicating X, 1 indicating Y, and 2 indicating Z.
// The reasult is stored in 'gradient', which is resized
// automatically.
void
EstimateAffineAtlas::_ComputeGradientComponent( const ImageType* image,
                                           int direction,
                                           ImagePointer gradient )
{
    // Create a new image

    gradient->resize( image->getSize() );
    Vector3D<double> imageSpacing = image->getSpacing();
    gradient->setSpacing( imageSpacing );
    gradient->setOrigin( image->getOrigin() );

    // Data pointers
    const VoxelType* imageDataPtr = image->getDataPointer();
    VoxelType* gradientDataPtr = gradient->getDataPointer();
    const VoxelType* temp1;
    const VoxelType* temp2;

    unsigned int xSize = gradient->getSizeX();
    unsigned int ySize = gradient->getSizeY();
    unsigned int zSize = gradient->getSizeZ();

    double xSpacing = (image->getSpacing()).x;
    double ySpacing = (image->getSpacing()).y;
    double zSpacing = (image->getSpacing()).z;

    switch( direction ) {
    case 0:
    {
        // X direction

        for( unsigned int zPos = 0; zPos < zSize; zPos++ ) {
            for( unsigned int yPos = 0; yPos < ySize; yPos++ ) {
                //
                // End Voxel
                //
                *gradientDataPtr =
                    (*(imageDataPtr+1) - *imageDataPtr) / xSpacing;

                gradientDataPtr++;
                imageDataPtr++;

                temp1 = imageDataPtr - 1;
                temp2 = imageDataPtr + 1;
                for( unsigned int xPos = 1; xPos < xSize - 1; xPos++ ) {
                    *gradientDataPtr =
                        (*temp2++ - *temp1++)/(float)(2 * xSpacing);

                    gradientDataPtr++;
                    imageDataPtr++;
                }

                //
                // Ending voxel
                //
                *gradientDataPtr =
                    (*imageDataPtr - *(imageDataPtr-1)) / xSpacing;

                gradientDataPtr++;
                imageDataPtr++;
            }
        }
        break;
    }
    case 1:
    {
        //
        // Y direction
        //
        for( unsigned int zPosDir = 0; zPosDir < zSize; zPosDir++ ) {
            temp1 = imageDataPtr;
            temp2 = imageDataPtr + xSize;

            //
            // First strip
            //
            for( unsigned int xPosFirst = 0; xPosFirst < xSize; xPosFirst++ ) {
                *gradientDataPtr =
                    (*temp2++ - *imageDataPtr) / ySpacing;

                gradientDataPtr++;
                imageDataPtr++;
            }

            //
            // Mid section
            //
            for( unsigned int yPosMid = 1; yPosMid < ySize - 1; yPosMid++ ) {
                for( unsigned int xPosMid = 0; xPosMid < xSize; xPosMid++ ) {
                    *gradientDataPtr =
                        (*temp2++ - *temp1++)/(float)(2 * ySpacing);

                    gradientDataPtr++;
                    imageDataPtr++;
                }
            }
            //
            // Last strip
            //
            for( unsigned int xPosLast = 0; xPosLast < xSize; xPosLast++ ) {
                *gradientDataPtr =
                    (*imageDataPtr - *temp1++) / ySpacing;

                gradientDataPtr++;
                imageDataPtr++;
            }
        }
        break;
    }
    case 2:
    {
        //
        // Z direction
        //
        temp1 = imageDataPtr;
        temp2 = imageDataPtr + xSize * ySize;

        //
        // First slice
        //
        for( unsigned int yPosSlice = 0; yPosSlice < ySize; yPosSlice++ ) {
            for( unsigned int xPosSlice = 0; xPosSlice < xSize; xPosSlice++ ) {
                *gradientDataPtr =
                    (*temp2++ - *imageDataPtr) / zSpacing;

                gradientDataPtr++;
                imageDataPtr++;
            }
        }

        //
        // Mid section
        //
        for( unsigned int zPosSec = 1; zPosSec < zSize - 1; zPosSec++ ) {
            for( unsigned int yPosSec = 0; yPosSec < ySize; yPosSec++ ) {
                for( unsigned int xPosSec = 0; xPosSec < xSize; xPosSec++ ) {
                    *gradientDataPtr =
                        (*temp2++ - *temp1++)/(float)(2 * zSpacing);

                    gradientDataPtr++;
                    imageDataPtr++;
                }
            }
        }

        //
        // Last slice
        //
        for( unsigned int yPosLastSlice = 0; yPosLastSlice < ySize; 
             yPosLastSlice++ ) {
            for( unsigned int xPosLastSlice = 0; xPosLastSlice < xSize;
                 xPosLastSlice++ ) {
                *gradientDataPtr =
                    (*imageDataPtr - *temp1++) / zSpacing;

                gradientDataPtr++;
                imageDataPtr++;
            }
        }
        break;
    }
    default:
        std::cerr << "[EstimateAffine::_ComputeGradientComponent] "
                  << "error: unknown direction " 
                  << direction << std::endl;
        break;
    }
} // EstimateAffine::_ComputeGradientComponent


//-------------------------------------------------------------------------
// Methods of shrinking the image.

// Wrapper function.
void EstimateAffineAtlas::
_ComputeShrinkImage(
    const ImageType* image,      // image to be shrunk
    ImageSizeType scaleFactors,  // Multiply dims by 1/factor
    ImagePointer shrinkImage )   // Must allocate image yourself
{
    _ComputeShrinkImageWithoutAveraging(image, scaleFactors, shrinkImage);
    // _ComputeShrinkImageWithGaussian(image, scaleFactors, shrinkImage);
}

// EstimateAffine:: _ComputeShrinkImageWithGaussian
void EstimateAffineAtlas::
_ComputeShrinkImageWithGaussian( 
    ImageType* image,      // image to be shrunk
    ImageSizeType scaleFactors,  // Multiply dims by 1/factor
    ImagePointer shrinkImage )   // Must allocate image yourself
{
    /** Downsample the image */
    DownsampleFilter3D filter;
    filter.SetInput(*image);
    filter.SetFactor(scaleFactors.x, scaleFactors.y, scaleFactors.z);
    filter.SetSigma(1*scaleFactors.x, 1*scaleFactors.y, 1*scaleFactors.z);
    filter.SetSize(2*scaleFactors.x, 2*scaleFactors.y, 2*scaleFactors.z);
    filter.Update();

    /** Create new downsampled image */
    ImageIndexType spacing = image->getSpacing();
    shrinkImage->resize(filter.GetNewSize());
    shrinkImage->setData(filter.GetOutput());
    //    ImageSizeType imagesize(shrinkImage->getSizeX(),
    //                        shrinkImage->getSizeY(),
    //                        shrinkImage->getSizeZ());
    spacing.scale(scaleFactors.x, scaleFactors.y, scaleFactors.z);
    shrinkImage->setSpacing(spacing);

    // The origin is the location, in geometric coordinates, of the
    // voxel indexed (0,0,0).  That location shouldn't change when the
    // image is downsampled.
    shrinkImage->setOrigin( image->getOrigin() );

}

// EstimateAffine:: _ComputeShrinkImageWithGaussian
void EstimateAffineAtlas::
_ComputeShrinkImageWithoutAveraging( 
    const ImageType* image,      // image to be shrunk
    ImageSizeType scaleFactors,  // Multiply dims by 1/factor
    ImagePointer shrinkImage )   // Must allocate image yourself
{
    Array3DUtils::downsampleByInts(*image, *shrinkImage, scaleFactors);

    ImageIndexType spacing = image->getSpacing();
    spacing.scale(scaleFactors.x, scaleFactors.y, scaleFactors.z);
    shrinkImage->setSpacing(spacing);

    // The origin is the location, in geometric coordinates, of the
    // voxel indexed (0,0,0).  That location shouldn't change when the
    // image is downsampled.
    shrinkImage->setOrigin( image->getOrigin() );

}

//-------------------------------------------------------------------------

void
EstimateAffineAtlas::_RunEstimateTransformLevel( 
    unsigned int level,
    unsigned int numberOfLevels,
    EstimateAffineAtlas::TransformGroupType transformType )
{

    double intensityRange = std::max(_maxFixedImage, _maxMovingImage) -
                            std::min(_minFixedImage, _minMovingImage);

    int iteration = 0;

    double meanSquareError = 1;
    double previousRmsError = 1; // Maximum possible
    while( true ) {

        // Note that meanSquaredError is always the value for the
        // previous estimate.
        switch (transformType) {
        case TRANSLATE:
            meanSquareError = _StepEstimateTranslation( level );
            break;
        case RIGID:
            _StepEstimateAffine( level );
            meanSquareError = _StepEstimateTranslation( level );
            _ConvertGeneralLinearToRotation(
                _levels[level].transformEstimate.matrix );
            // This does affect the meanSquareError value since the
            // MSE for iteration n is computed during iteration n+1.
            break;
        case AFFINE:
            _StepEstimateAffine( level );
            meanSquareError = _StepEstimateTranslation( level );
            break;
        }

        // Error is normalized in case it is decided to use the actual
        // mean squared error as a stopping criterion.  RMS error is
        // used instead of mean squared error for similar reason --
        // that way the stopping criterion has intuitive meaning.  The
        // speed penalty for taking sqrt is negligible in this outer
        // loop.
        double rmsError = sqrt( meanSquareError ) / intensityRange;

        double changeInError = (previousRmsError - rmsError) /
                               previousRmsError;

        if (_outputMode != SILENT) {
            //std::cout << level << "   " << iteration << "   " << rmsError
             //         << "   " << changeInError * 100.0 << std::endl;
        }

        // We test rmsError against 0 to avoid the possibility of zero
        // divides when computing changeInError.  No need for an
        // epsilon here.
        if( changeInError < 1e-3 || iteration == 10 || rmsError == 0 ) {
            // Roll back and go to the next level
            _levels[level].transformEstimate.vector = 
                _previousTransformEstimate.vector;
            break;
        }
       // break;//only 1 iteration
        ++iteration;

        _previousTransformEstimate = _levels[level].transformEstimate;
        previousRmsError = rmsError;
//break;
    }
/*
    if( level == numberOfLevels - 1 ) { 
        // Done
*/
        _finalTransformEstimate = 
            _TempOriginCoordinatesToWorldCoordinates(
                _levels[level].fixedImage, 
                _levels[level].movingImage, 
                _levels[level].fixedImageTempOrigin, 
                _levels[level].movingImageTempOrigin, 
                _levels[level].transformEstimate );
/*
    } else {
        // Go to next level
        _levels[level+1].transformEstimate = _levels[level].transformEstimate;
        _RunEstimateTransformLevel( level + 1, numberOfLevels,
                                         transformType );
    }*/ // for only 1 level
} // EstimateAffine::_RunEstimateTransformLevel


// Sets the estimated matrix and translation vector based on the
// previous level.
void EstimateAffineAtlas::
SetInitialTransformEstimate( const TransformType& transform )
{
    _initialTransformEstimate = transform;
}


void
EstimateAffineAtlas::_ConvertGeneralLinearToRotation( Matrix3D<double>& matrix )
{
    Matrix3D<double> U, Vt;
    double sv[3];
    Matrix3D<double>::computeSVD( matrix.a, U.a, Vt.a, sv );
    matrix = U * Vt;
} 


float
EstimateAffineAtlas::_StepEstimateTranslation( unsigned level )
{

    // Get references to everything in level struct.
    const Vector3D<unsigned int>& roiStart = (_levels[level].roi).getStart();
    const Vector3D<unsigned int>& roiStop = (_levels[level].roi).getStop();
    const ImageType* movingImage = _levels[level].movingImage;
    const ImageType* fixedImage = _levels[level].fixedImage;
    const ImageType* movingImageGradientX = _levels[level].movingImageGradientX;
    const ImageType* movingImageGradientY = _levels[level].movingImageGradientY;
    const ImageType* movingImageGradientZ = _levels[level].movingImageGradientZ;

    // const Vector3D<double>& scaleFactors = _levels[level].scaleFactors;
    TransformType transformEstimate = _levels[level].transformEstimate;
    const Vector3D<double>& movingImageTempOrigin =
        _levels[level].movingImageTempOrigin;
    const Vector3D<double>& fixedImageTempOrigin =
        _levels[level].fixedImageTempOrigin;

    double volume = ( 
        (roiStop[2] - roiStart[2] + 1) *
        (roiStop[1] - roiStart[1] + 1) *
        (roiStop[0] - roiStart[0] + 1));

    float meanSquareError = 0;

    // Here we use Vector rather than Vector3D to maintain cosistency
    // with _StepEstimateAffine, which needs 9D Vectors and Matrices.

    Vector V( 3 );
    V.setAll( 0 );

    Vector numerator( 3 );
    numerator.setAll( 0 );

    Matrix denominator( 3, 3 );
    denominator.setAll( 0 );

    TransformType transform =
        ImageUtils::transformInIndexCoordinates( fixedImageTempOrigin,
                                                 fixedImage->getSpacing(), 
                                                 movingImageTempOrigin,
                                                 movingImage->getSpacing(),
                                                 transformEstimate );

    double x, y, z;
    for( unsigned int zPos = roiStart[2]; zPos <= roiStop[2]; zPos++ ) {
        for( unsigned int yPos = roiStart[1]; yPos <= roiStop[1]; yPos++ ) {
            for( unsigned int xPos = roiStart[0]; xPos <= roiStop[0]; xPos++ ) {

                // We must perform an affine transformation to account
                // for scaling.
                transform.transformCoordinates( xPos, yPos, zPos, x, y, z );

                size_t i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0;
                double a0=0., a1=0., a2=0., a3=0., a4=0., a5=0., a6=0., a7=0.;
                if( Array3DUtils::
                    computeTrilerpCoeffs( *movingImage, x, y, z,
                                          i0, i1, i2, i3, i4, i5, i6, i7,
                                          a0, a1, a2, a3, a4, a5, a6, a7 )) {

                    float fixedImageValue = (*fixedImage)(xPos, yPos, zPos);

                    // Interpolate moving image
                    float movingImageValue = 
                        Array3DUtils::
                        weightedSumOfArrayValues( 
                            *movingImage,
                            i0, i1, i2, i3, i4, i5, i6, i7,
                            a0, a1, a2, a3, a4, a5, a6, a7 );

                    // Interpolate gradient images

                    V(0) = Array3DUtils::
                           weightedSumOfArrayValues(
                               *movingImageGradientX,
                               i0, i1, i2, i3, i4, i5, i6, i7,
                               a0, a1, a2, a3, a4, a5, a6, a7 );

                    V(1) = Array3DUtils::
                           weightedSumOfArrayValues(
                               *movingImageGradientY,
                               i0, i1, i2, i3, i4, i5, i6, i7,
                               a0, a1, a2, a3, a4, a5, a6, a7 );

                    V(2) = Array3DUtils::
                           weightedSumOfArrayValues(
                               *movingImageGradientZ,
                               i0, i1, i2, i3, i4, i5, i6, i7,
                               a0, a1, a2, a3, a4, a5, a6, a7 );

                    float pixelError = fixedImageValue - movingImageValue;

                    // Add to the cumulative square error
                    meanSquareError += pixelError * pixelError;

                    // Add to the numerator
                    //
                    // Numerator = Int [I_tem(X~x) - I_sub(x)]V(x) dx
                    //           = Int [movingImageValue -
                    //                  fixedImageValue]V(x) dx
                    for( unsigned int lcv = 0; lcv < 3; lcv++ ) {
                        numerator(lcv) -= pixelError * V(lcv);
                    }

                    // Add to the denominator
                    //
                    // Denominator = Int V(x)[V(x)]^T dx
                    for( unsigned int row = 0; row < 3; row++ ) {
                        for( unsigned int col = 0; col < 3; col++ ) {
                            denominator(row,col) += V(row) * V(col);
                        }
                    }
                }
            }
        }
    }

    // Normalize error by volume of image used
    meanSquareError /= volume;

    if( _initialMeanSquareError < 0 ) {
        _initialMeanSquareError = meanSquareError;
    }

    Matrix denominatorInverse( 3, 3 );
    denominator.inverse( denominatorInverse );

    Vector delta = denominatorInverse * numerator;
    _levels[level].transformEstimate.vector -= castVectorToVector3D( delta );

    return( meanSquareError );

} // EstimateAffine::_StepEstimateTranslation

// Perform an iteration improving the offsetEstimate and
// matrixEstimate.  The two estimates are expressed in physical
// coordinates and in terms of the specified centers for the two
// images; that is to say, the movingImageTempOrigin and fixedImageTempOrigin
// are used as the origins of the two spaces being transformed.  The
// resulting transform must be converted to use the actual origins for
// the two images.
float
EstimateAffineAtlas::_StepEstimateAffine( unsigned level )
{

    // Get references to everything in level struct.
    const Vector3D<unsigned int>& roiStart = (_levels[level].roi).getStart();
    const Vector3D<unsigned int>& roiStop = (_levels[level].roi).getStop();
    const ImageType* movingImage = _levels[level].movingImage;
    const ImageType* fixedImage = _levels[level].fixedImage;
    const ImageType* movingImageGradientX = _levels[level].movingImageGradientX;
    const ImageType* movingImageGradientY = _levels[level].movingImageGradientY;
    const ImageType* movingImageGradientZ = _levels[level].movingImageGradientZ;

    // const Vector3D<double>& scaleFactors = _levels[level].scaleFactors;
    TransformType& transformEstimate = _levels[level].transformEstimate;
    const Vector3D<double>& movingImageTempOrigin =
        _levels[level].movingImageTempOrigin;
    const Vector3D<double>& fixedImageTempOrigin =
        _levels[level].fixedImageTempOrigin;

    double volume = ( 
        (roiStop[2] - roiStart[2] + 1) *
        (roiStop[1] - roiStart[1] + 1) *
        (roiStop[0] - roiStart[0] + 1));

    float meanSquareError = 0;

    // V will equal (transposed image gradient) * X, where X = 
    //
    //      xc  yc  zc  0   0   0   0   0   0
    //      0   0   0   xc  yc  zc  0   0   0
    //      0   0   0   0   0   0   xc  yc  zc
    //
    Vector V( 9 );
    V.setAll( 0 );

    // The components of the next matrix estimate will be given by
    // denominator^-1 * numerator.  Both numerator and denominator are
    // computed using V.
    Vector numerator( 9 );
    numerator.setAll( 0 );

    Matrix denominator( 9, 9 );
    denominator.setAll( 0 );

    TransformType transform =
        ImageUtils::transformInIndexCoordinates( fixedImageTempOrigin,
                                                 fixedImage->getSpacing(), 
                                                 movingImageTempOrigin,
                                                 movingImage->getSpacing(),
                                                 transformEstimate );

    double x, y, z;
    for( unsigned int zPos = roiStart[2]; zPos <= roiStop[2]; zPos++ ) {
        for( unsigned int yPos = roiStart[1]; yPos <= roiStop[1]; yPos++ ) {
            for( unsigned int xPos = roiStart[0]; xPos <= roiStop[0]; xPos++ ) {

                transform.transformCoordinates( xPos, yPos, zPos, x, y, z );

                size_t i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0;
                double a0=0., a1=0., a2=0., a3=0., a4=0., a5=0., a6=0., a7=0.;
                if( Array3DUtils::
                    computeTrilerpCoeffs( *movingImage, x, y, z,
                                          i0, i1, i2, i3, i4, i5, i6, i7,
                                          a0, a1, a2, a3, a4, a5, a6, a7 )) {

                    float fixedImageValue = (*fixedImage)(xPos, yPos, zPos);

                    // Interpolate moving image
                    float movingImageValue = 
                        Array3DUtils::
                        weightedSumOfArrayValues( 
                            *movingImage,
                            i0, i1, i2, i3, i4, i5, i6, i7,
                            a0, a1, a2, a3, a4, a5, a6, a7 );

                    float interpolatedGradientValue[3];

                    interpolatedGradientValue[0] =
                        Array3DUtils::
                        weightedSumOfArrayValues(
                            *movingImageGradientX,
                            i0, i1, i2, i3, i4, i5, i6, i7,
                            a0, a1, a2, a3, a4, a5, a6, a7 );

                    interpolatedGradientValue[1] =
                        Array3DUtils::
                        weightedSumOfArrayValues(
                            *movingImageGradientY,
                            i0, i1, i2, i3, i4, i5, i6, i7,
                            a0, a1, a2, a3, a4, a5, a6, a7 );

                    interpolatedGradientValue[2] =
                        Array3DUtils::
                        weightedSumOfArrayValues(
                            *movingImageGradientZ,
                            i0, i1, i2, i3, i4, i5, i6, i7,
                            a0, a1, a2, a3, a4, a5, a6, a7 );

                    float pixelError = fixedImageValue - movingImageValue;

                    // Add to the cumulative square error
                    meanSquareError += pixelError * pixelError;

                    // Compute V(x) where x = (xc, yc, zc) (mm)

                    double xc, yc, zc;
                    _ConvertIndexToTempOriginCoordinates(
                        fixedImage->getSpacing(), fixedImageTempOrigin,
                        xPos, yPos, zPos, xc, yc, zc );

                    V(0) = interpolatedGradientValue[0] * xc;
                    V(1) = interpolatedGradientValue[0] * yc;
                    V(2) = interpolatedGradientValue[0] * zc;

                    V(3) = interpolatedGradientValue[1] * xc;
                    V(4) = interpolatedGradientValue[1] * yc;
                    V(5) = interpolatedGradientValue[1] * zc;

                    V(6) = interpolatedGradientValue[2] * xc;
                    V(7) = interpolatedGradientValue[2] * yc;
                    V(8) = interpolatedGradientValue[2] * zc;

                    // Add to the numerator
                    //
                    // Numerator = Int [I_tem(X~x) - I_sub(x)]V(x) dx
                    for( unsigned int lcv = 0; lcv < 9; lcv++ ) {
                        numerator(lcv) -= pixelError * V(lcv);
                    }

                    // Add to the denominator
                    //
                    // Denominator = Int V(x)[V(x)]^T dx
                    //
                    for( unsigned int row = 0; row < 9; row++ ) {
                        for( unsigned int col = 0; col < 9; col++ ) {
                            denominator(row,col) += V(row) * V(col);
                        }
                    }
                }
            }
        }
    }

    // Normalize error by volume of image used
    meanSquareError /= volume;

    if( _initialMeanSquareError < 0 ) {
        _initialMeanSquareError = meanSquareError;
    }

    Matrix denominatorInverse( 9, 9 );
    denominator.inverse( denominatorInverse );

    // Compute the delta step
    //
    // delta = (denominator)^-1 * numerator
    Vector delta( 9 );
    delta.setAll( 0 );
    for( unsigned int rowDelta = 0; rowDelta < 9; rowDelta++ ) {
        for( unsigned int col = 0; col < 9; col++ ) {
            delta( rowDelta ) += denominatorInverse( rowDelta, col ) *
                                 numerator( col );
        }
    }

    // Update
    for( unsigned int row = 0; row < 3; row++ ) {
        for( unsigned int col = 0; col < 3; col++ ) {
            transformEstimate.matrix(row,col) -= delta( row*3 + col );
        }
    }

    return( meanSquareError );
}

