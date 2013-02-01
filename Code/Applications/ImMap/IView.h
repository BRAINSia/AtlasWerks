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

#ifndef IView_h
#define IView_h

#include <FL/gl.h>
#include <FL/Fl.H>
#include <FL/fl_draw.H>
#include <FL/Fl_Gl_Window.H>

//#include <GL/glu.h>

#include <algorithm>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <list>

#include <Array2D.h>
#include <Array3DUtils.h>
#include <ROI.h>
#include <StringUtils.h>
#include "Image.h"
#include "Timer.h"
#include "gen.h"
#include "Anastruct.h"

enum IViewOrientationType { IVIEW_SAGITTAL, IVIEW_CORONAL, IVIEW_AXIAL };

class IViewROI
{
public:
  int start[3];
  int stop[3];
};

template <class T>
class RGBpixel 
{
public :
  T r;
  T g;
  T b;
};

class IViewAnastructGroup
{
public:
  Anastruct  anastruct;
  bool       isVisible;
  double     r;
  double     g;
  double     b;
  
  
  IViewAnastructGroup()
    : anastruct(),
    isVisible(false),
    r(0),
    g(0),
    b(0)
  {}
  
  IViewAnastructGroup(const IViewAnastructGroup& rhs)
    : anastruct(rhs.anastruct),
    isVisible(rhs.isVisible),
    r(rhs.r),
    g(rhs.g),
    b(rhs.b)
  {}
  
  IViewAnastructGroup(const Anastruct& anastruct,
    bool isVisible,
    const double& r,
    const double& g,
    const double& b)
    : anastruct(anastruct),
    isVisible(isVisible),
    r(r),
    g(g),
    b(b)
  {}
  
  IViewAnastructGroup& operator=(const IViewAnastructGroup& rhs)
  {
    anastruct = rhs.anastruct;
    isVisible = rhs.isVisible;
    r = rhs.r;
    g = rhs.g;
    b = rhs.b;
    return *this;
  }
  
  void setColor(const double& r,
    const double& g,
    const double& b)
    
  {
    this->r = r;
    this->g = g;
    this->b = b;
  }

   void setVisibility(const bool& visible)
    
  {
    this->isVisible = visible;
  }
  
};

template <class ImagePixelType, class OverlayPixelType>
class IView
  : public Fl_Gl_Window
{
public:
  typedef Image<ImagePixelType>            ImageType;
  typedef ImageType*		           ImagePointer;
  typedef ROI<int, unsigned int>           ImageRegionType;
  typedef Vector3D<double>                 ImageSizeType;
  typedef Vector3D<double>                 ImageIndexType;
  
  
  typedef Image<OverlayPixelType>          OverlayType;
  typedef OverlayType*			   OverlayPointer;
  typedef ROI<int, unsigned int>           OverlayRegionType;
  typedef Vector3D<double>		   OverlaySizeType;
  typedef Vector3D<double>                 OverlayIndexType;
  
  typedef RGBpixel<unsigned char>          WindowImagePixelType;
  typedef Array2D<WindowImagePixelType>    WindowImageType;
  typedef Array3D<float>                   MaskType;
protected:
  // true if this instance should output various info to std::cerr
  bool            _verbose;
  
  // window dimensions
  int             _windowWidth;
  int             _windowHeight;
  
  // window position
  int             _windowX;
  int             _windowY;
  
  // image data
  ImagePointer    _imageData;
  OverlayPointer  _overlayData;
  int             _imageDim[3];
  int             _maxImageDim;
  int             _overlayDim[3];
  double          _imageSpacing[3];
  double          _overlaySpacing[3];
  double          _imageOffset[3];
  double          _overlayOffset[3];
  std::string     _imageName;
  
  double          _imageDataMax;
  double          _imageDataMin;
  double          _overlayDataMax;
  double          _overlayDataMin;
  
  int             _windowCenter[3];
  float           _windowCenterf[3];
  float           _axisSign[3];
  int             _order[3];
  int             _bufferedImageSliceIndex;
  
  IViewOrientationType _orientation;
  
  std::vector<IViewAnastructGroup> _imageAnastructs;
  std::vector<IViewAnastructGroup> _overlayAnastructs;
  
  // true if image data is valid
  bool            _haveValidImageData;
  bool            _haveValidOverlayData;
  
  // the buffer being displayed by gl (a slice of the image)
  unsigned char  *_imagePixelBuffer;
  unsigned char  *_overlayPixelBuffer;
  unsigned int    _imagePixelBufferWidth;
  unsigned int    _imagePixelBufferHeight;
  unsigned int    _overlayPixelBufferWidth;
  unsigned int    _overlayPixelBufferHeight;
  
  // determindes opacity of overlay [0,1]
  double          _overlayOpacity;
  
  // intensity window limits
  double          _imageIWMax;
  double          _imageIWMin;
  double          _overlayIWMax;
  double          _overlayIWMin;
  
  // fraction of intensity level each channel should receive
  double          _imageColorR;
  double          _imageColorG;
  double          _imageColorB;
  double          _overlayColorR;
  double          _overlayColorG;
  double          _overlayColorB;
  
  double          _windowZoom;
  
  // what to view
  bool            _viewImage;
  bool            _viewOverlay;
  bool            _viewCrosshairs;
  bool            _viewROI;
  bool            _viewMask;
  bool            _viewImageContours;
  bool            _viewOverlayContours;
  bool            _viewImageInfo;
  
  bool            _ROILocked;
  
  double          _crosshairOpacity;
  double          _roiOpacity;
  
  // keep track of update requirements
  bool            _needImageUpdate;
  bool            _needOverlayUpdate;
  
  // region of interest
  IViewROI        _roi;
  bool            _roiDraggingXMin;
  bool            _roiDraggingXMax;
  bool            _roiDraggingYMin;
  bool            _roiDraggingYMax;
  bool            _roiDraggingZMin;
  bool            _roiDraggingZMax;
  
  //mask  
  MaskType               _mask;

  //line width of the contour
  float _lineWidth;
  
  
  // callbacks
  void (*_clickCallback)(int button, float xIndex, float yIndex, float zIndex, 
                         void *arg);  
  void *_clickCallbackArg;
  void (*_roiChangedCallback)(IViewOrientationType orientation, void *arg);  
  void *_roiChangedCallbackArg;
  
public:

  // fltk constructor (a la fltk)
  IView(int x, int y, int w, int h, const char *l);    
  ~IView();    
  
  // toggle verbose output
  void setVerbose(bool verbose);
  
  // set image and overlay
  void setImage(ImagePointer image, 
    const ImagePixelType& imageDataMin,
    const ImagePixelType& imageDataMax);
  void setOverlay(OverlayPointer overlay, 
    const OverlayPixelType& overlayDataMin,
    const OverlayPixelType& overlayDataMax);
  
  // these compute data min/max for you
  void setImage(ImagePointer image);
  void setOverlay(OverlayPointer overlay);
  
  void setImageName(const std::string& imageName);
  
  // anastruct interface
  void addImageAnastruct(const Anastruct& anastruct,
    bool isVisible = true,
    const double& r = 1,
    const double& g = 0,
    const double& b = 0);
  void setImageAnastructColor(int anastructIndex,
    const double& r,const double& g, const double& b);
  void clearImageAnastructs();
  
  void addOverlayAnastruct(const Anastruct& anastruct,
    bool isVisible = true,
    const double& r = 0,
    const double& g = 1,
    const double& b = 0);
  void setOverlayAnastructColor(int anastructIndex,
    const double& r,const double& g, const double& b);
  void clearOverlayAnastructs();
  
  void setImageContourVisibility(int imageContourIndex, const bool& visible)
  {
    _imageAnastructs[imageContourIndex].setVisibility(visible);
  };

  void setOverlayContourVisibility(int overlayContourIndex, const bool& visible)
  {
    _overlayAnastructs[overlayContourIndex].setVisibility(visible);
  };
    
    // true if there is a valid image(overlay) loaded
    bool haveValidImage() const;
    bool haveValidOverlay() const;
    
    // get image size
    int getImageDimX() const;
    int getImageDimY() const;
    int getImageDimZ() const;
    
    // toggle display
    bool getViewImage() const;
    void setViewImage(bool view);
    bool getViewOverlay() const;
    void setViewOverlay(bool view);
    bool getViewCrosshairs() const;
    void setViewCrosshairs(bool view);
    void setViewImageInfo(bool view);
    bool getViewROI() const;
    void setViewROI(bool view);
    void setROILocked(bool ROILocked){_ROILocked = ROILocked;};
    
    bool getViewMask() const;
    void setViewMask(bool view);

    void setCrosshairsOpacity(const double& alpha);
    void setROIOpacity(const double& alpha);
    
    // How to view the image:
    // IVIEW_SAGITTAL - along x axis
    // IVIEW_CORONAL - along y axis
    // IVIEW_AXIAL - along z axis
    // Terminology assumes patient is lying on a table in the XZ
    // plane, aligned along the Z axis.
    IViewOrientationType getOrientation() const;
    void setOrientation(IViewOrientationType orientation);
    
    // where in the data to view
    int getNumSlices() const;
    int getSliceIndex() const;
    Vector3D<unsigned int> getWindowCenter() const;
    void setSliceIndex(int index);
    void setDataCenter(float xIndex, float yIndex, float zIndex);
    
    // overlay opacity [0,1]
    double getOverlayOpacity() const;
    void setOverlayOpacity(const double& opacity);

    // line width
    void setLineWidth(const double& lineWidth);
    
    // color (e.g. red channel gets intensity * r) [0,1]
    double getImageColorR() const;
    double getImageColorG() const;
    double getImageColorB() const;
    double getOverlayColorR() const;
    double getOverlayColorG() const;
    double getOverlayColorB() const;
    void setImageColorRGB(const double& r, const double& g, const double& b);
    void setOverlayColorRGB(const double& r, const double& g, const double& b);
    
    // data range
    double getImageDataMax() const;    
    double getImageDataMin() const;    
    double getOverlayDataMax() const;    
    double getOverlayDataMin() const;    
    
    // intensity windowing
    double getImageIWMax() const;
    void setImageIWMax(const double& max);
    double getImageIWMin() const;
    void setImageIWMin(const double& min);
    double getOverlayIWMax() const;
    void setOverlayIWMax(const double& max);
    double getOverlayIWMin() const;
    void setOverlayIWMin(const double& min);
    
    // zoom [1,...)
    double getWindowZoom() const;
    void setWindowZoom(const double& zoom);
    
    // change window size (a la Fl_Gl_Window)
    void size(int w, int h);
    void resize(int x, int y, int w, int h);
    
    // handle events (a la fltk)
    int handle(int event);
    
    // draw this window
    void draw();
    
    // update pixel buffers as necessary
    void update();
    
    // set callbacks
    void setClickCallback(void (*callbackFcn)(int button,
      float xIndex,
      float yIndex,
      float zIndex,
      void *arg),
      void *callbackArg);
    
    void setROIChangedCallback(
      void (*callbackFcn)(IViewOrientationType orientation, void *arg),
      void *callbackArg);
    
    // roi interface
    void setROI(int startX, int startY, int startZ,
      int stopX, int stopY, int stopZ);
    void getROI(int& startX, int& startY, int& startZ,
      int& stopX, int& stopY, int& stopZ) const;

    // mask
    void setMask(const MaskType& mask);
    
    bool isImageCompatible(const ImagePointer& image) const;
    bool isOverlayCompatible(const ImagePointer& overlay) const;
    
    // save window as an image
    void saveWindowAsImage(const char *filename);
    
private:
  // read the current slice into a char buffer for display
  void _updateImagePixelBuffer();
  void _updateOverlayPixelBuffer();
  
  // compute data range
  static void _computeImageDataMaxMin(const ImagePointer& image, 
    double& max,
    double& min);
  
  // make sure overlay and image can be used together
  bool _checkImageCompatibility(ImagePointer image, OverlayPointer overlay);
  
  void _setImageSize(int xDim, 
    int yDim, 
    int zDim,
    const double& xSpacing, 
    const double& ySpacing, 
    const double& zSpacing,
    const double& xOffset,
    const double& yOffset,
    const double& zOffset);
  
  void _setOverlaySize(int xDim, 
    int yDim, 
    int zDim,
    const double& xSpacing, 
    const double& ySpacing, 
    const double& zSpacing,
    const double& xOffset,
    const double& yOffset,
    const double& zOffset);
  
  void _setNewImageCenter();
  
  // compute pixel zoom
  double _imagePixelZoomX() const;
  double _imagePixelZoomY() const;
  double _overlayPixelZoomX() const;
  double _overlayPixelZoomY() const;
  
  // change from window to image index coordinates and back
  void _windowToImageIndexCoordinates(float windowX,
    float windowY,
    float& imageX,
    float& imageY,
    float& imageZ) const;
  void _imageIndexToWindowCoordinates(float imageX,
    float imageY,
    float imageZ,
    float& windowX,
    float& windowY) const;
  
  void _setupGLDrawPixelsImage();
  void _setupGLDrawPixelsOverlay();
  
  void _drawImageContours();
  void _drawOverlayContours();
  void _drawImageInfo();
  void _drawCrosshairs();
  void _drawROI();
  void _drawMask();
  void _drawString(const std::string& text, const float& x, const float& y);
  
  double _getOverlaySliceIndex(const int& imageSliceIndex);
  
  void _checkForGLError(const char *message);
  
  void _updateROIDraggingFlags(bool isPress, int mouseX, int mouseY);
  void _dragROI(int mouseDX, int mouseDY);
};

//
// constructor (a la Fl_Gl_Window)
//
template <class ImagePixelType, class OverlayPixelType>
IView<ImagePixelType, OverlayPixelType>
::IView(int x, int y, int w, int h, const char *l)
: Fl_Gl_Window(x, y, w, h, l)
{
  //mode(FL_RGB|FL_DOUBLE|FL_ALPHA);
  // _verbose = true;
  _verbose = false;
  
  _windowWidth = w;
  _windowHeight = h;

  _windowX = x;
  _windowY = y;

  _maxImageDim = 0;
  
  _lineWidth = 2.0;
  
  _imageDim[0] = 0;
  _imageDim[1] = 0;
  _imageDim[2] = 0;
  _overlayDim[0] = 0;
  _overlayDim[1] = 0;
  _overlayDim[2] = 0;
  _imageSpacing[0] = 1;
  _imageSpacing[1] = 1;
  _imageSpacing[2] = 1;
  _overlaySpacing[0] = 1;
  _overlaySpacing[1] = 1;
  _overlaySpacing[2] = 1;
  _imageOffset[0] = 1;
  _imageOffset[1] = 1;
  _imageOffset[2] = 1;
  _overlayOffset[0] = 1;
  _overlayOffset[1] = 1;
  _overlayOffset[2] = 1;
  _windowCenter[0] = 0;
  _windowCenter[1] = 0;
  _windowCenter[2] = 0;
  
  _windowCenterf[0] = 0.0;
  _windowCenterf[1] = 0.0;
  _windowCenterf[2] = 0.0;
  
  _orientation = IVIEW_CORONAL;

  // By default, flip image in Z direction so head is at top.
  _axisSign[0] = 1.0f;
  _axisSign[1] = 1.0f;
  _axisSign[2] = -1.0f;

  _order[0] = 0;
  _order[1] = 1;
  _order[2] = 2;
  _bufferedImageSliceIndex = -1;
  
  _haveValidImageData = false;
  _haveValidOverlayData = false;
  
  _imageIWMin = 0;
  _imageIWMax = 0;
  _overlayIWMin = 0;
  _overlayIWMax = 0;
  
  _overlayOpacity = 0;
  _imageColorR = 1;
  _imageColorG = 1;
  _imageColorB = 1;
  _overlayColorR = 1;
  _overlayColorG = 1;
  _overlayColorB = 1;
  
  _windowZoom = 1;
  
  _viewImage      = true;
  _viewOverlay    = true;
  _viewCrosshairs = true;
  _viewROI        = true;
  _viewMask       = true;
  _ROILocked		= false;
  _viewImageContours = true;
  _viewOverlayContours = true;
  _viewImageInfo       = true;
  
  _crosshairOpacity = 0.75;
  _roiOpacity       = 0.25;
  _needImageUpdate = true;
  _needOverlayUpdate = true;
  
  _clickCallback = 0;
  _clickCallbackArg = 0;
  
  _roiChangedCallback = 0;
  _roiChangedCallbackArg = 0;
  
  _imagePixelBuffer = 0;
  _overlayPixelBuffer = 0;
  
  _roiDraggingXMin = false;
  _roiDraggingXMax = false;
  _roiDraggingYMin = false;
  _roiDraggingYMax = false;
  _roiDraggingZMin = false;
  _roiDraggingZMax = false;
  
}

//
// destructor
//
template <class ImagePixelType, class OverlayPixelType>
IView<ImagePixelType, OverlayPixelType>
::~IView()
{
  if (_verbose)
  {
    std::cerr << "[IView::~IView]" << std::endl;
  }
  if (_imagePixelBuffer != 0) 
  {
    delete [] _imagePixelBuffer;
  }
  _imagePixelBuffer = 0;
  
  if (_overlayPixelBuffer != 0)
  {
    delete [] _overlayPixelBuffer;
  }
  _overlayPixelBuffer = 0;
}

//
// setVerbose
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setVerbose(bool verbose)
{
  _verbose = verbose;
}

//
// setImage
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setImage(ImagePointer image, 
           const ImagePixelType& imageDataMin,
           const ImagePixelType& imageDataMax)
{
  _needImageUpdate = true;
  
  if (image == 0) 
  {
    _haveValidImageData = false;
    return;
  }
  
  // get whole image as a region, bail if there are no pixels there
  ImageRegionType imageRegion;
  imageRegion.setSize(Vector3D<double>(image->getSize()));
  if(image->isEmpty() != 0) 
  {
    throw std::invalid_argument(
        "IView::setImage: image region contains zero pixels."); 
  }
  
  // unused var// ImageSizeType imageSize = imageRegion.GetSize();
  int sizeX = imageRegion.getSize()[0];
  int sizeY = imageRegion.getSize()[1];
  int sizeZ = imageRegion.getSize()[2];
  double spacingX = image->getSpacing()[0];
  double spacingY = image->getSpacing()[1];
  double spacingZ = image->getSpacing()[2];
  double originX = image->getOrigin()[0];
  double originY = image->getOrigin()[1];
  double originZ = image->getOrigin()[2];
  //   _setImageSize(imageRegion.GetSize()[0],
  // 		imageRegion.GetSize()[1],
  // 		imageRegion.GetSize()[2],
  // 		image->getSpacing()[0],
  // 		image->getSpacing()[1],
  // 		image->getSpacing()[2],
  // 		image->getOrigin()[0],
  // 		image->getOrigin()[1],
  // 		image->getOrigin()[2]);
  _setImageSize(sizeX,
    sizeY,
    sizeZ,
    spacingX,
    spacingY,
    spacingZ,
    originX,
    originY,
    originZ);
  
  if ((!_haveValidImageData) &&
    (!_haveValidOverlayData))
  {
    // set reasonable viewing center
    _setNewImageCenter();
  }
  
  // set data min/max
  _imageDataMin = imageDataMin;
  _imageDataMax = imageDataMax;
  
  // set image pointer
  _imageData = image;
  _haveValidImageData = true;
}

//
// setImage
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setImage(ImagePointer image)
{
  double max = 0;
  double min = 0;
  if (image != 0)
  {
    _computeImageDataMaxMin(image, max, min);
  }
  setImage(image, min, max);
}

//
// setOverlay
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setOverlay(OverlayPointer overlay,
             const OverlayPixelType& overlayDataMin,
             const OverlayPixelType& overlayDataMax)
{
  _needOverlayUpdate = true;
  
  if (overlay == 0) 
  {
    _haveValidOverlayData = false;
    return;
  }
  
  // get whole overlay as a region, bail if there are no pixels there
  OverlayRegionType overlayRegion ;
  overlayRegion.setSize(Vector3D<double>(overlay->getSize()));
  if(overlay->isEmpty() != 0) 
  {
    throw std::invalid_argument("IView::setOverlay: overlay region contains zero pixels."); 
  }
  
  // unused var //OverlaySizeType overlaySize = overlayRegion.GetSize();
  _setOverlaySize(overlayRegion.getSize()[0], 
		  overlayRegion.getSize()[1], 
      overlayRegion.getSize()[2],
      overlay->getSpacing()[0],
      overlay->getSpacing()[1],
      overlay->getSpacing()[2],
      overlay->getOrigin()[0],
      overlay->getOrigin()[1],
      overlay->getOrigin()[2]);
  
  if ((!_haveValidImageData) &&
    (!_haveValidOverlayData))
  {
    // image size must be set,
    // set it to the size of the overlay
    _setImageSize(overlayRegion.getSize()[0], 
		    overlayRegion.getSize()[1], 
        overlayRegion.getSize()[2],
        overlay->getSpacing()[0],
        overlay->getSpacing()[1],
        overlay->getSpacing()[2],
        overlay->getOrigin()[0],
        overlay->getOrigin()[1],
        overlay->getOrigin()[2]);
    
    // set a reasonable image center
    _setNewImageCenter();
  }
  
  // set data min/max
  _overlayDataMin = overlayDataMin;
  _overlayDataMax = overlayDataMax;
  
  // set overlay pointer
  _overlayData = overlay;
  _haveValidOverlayData = true;
}

//
// setOverlay
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setOverlay(OverlayPointer overlay)
{
  double max = 0;
  double min = 0;
  if (overlay != 0)
  {
    _computeImageDataMaxMin(overlay, max, min);
  }
  setOverlay(overlay, min, max);
}

template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setImageName(const std::string& imageName)
{
  _imageName = imageName;
}

//
// addImageAnastruct
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::addImageAnastruct(const Anastruct& anastruct,
                    bool isVisible,
                    const double& r,
                    const double& g,
                    const double& b)
{
  _imageAnastructs.push_back(IViewAnastructGroup(anastruct,
    isVisible,
    r, g, b));
}

//
// setImageAnastructColor
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setImageAnastructColor(int anastructIndex,
                         
                         const double& r,const double& g, const double& b)
{
  _imageAnastructs[anastructIndex].setColor(r,g,b);
  
}


//
// clearImageAnastructs
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::clearImageAnastructs()
{
  _imageAnastructs.clear();
}

//
// addOverlayAnastruct
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::addOverlayAnastruct(const Anastruct& anastruct,
                      bool isVisible,
                      const double& r,
                      const double& g,
                      const double& b)
{
  _overlayAnastructs.push_back(IViewAnastructGroup(anastruct,
    isVisible,
    r, g, b));
}

//
// setOverlayAnastructColor
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setOverlayAnastructColor(int anastructIndex,
                           const double& r,const double& g, const double& b)
{
   _overlayAnastructs[anastructIndex].setColor(r,g,b);
}
//
// clearOverlayAnastructs
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::clearOverlayAnastructs()
{
  _overlayAnastructs.clear();
}

//
// haveValidImage
//
template <class ImagePixelType, class OverlayPixelType>
bool
IView<ImagePixelType, OverlayPixelType>
::haveValidImage() const
{
  return _haveValidImageData;
}

//
// haveValidOverlay
//
template <class ImagePixelType, class OverlayPixelType>
bool
IView<ImagePixelType, OverlayPixelType>
::haveValidOverlay() const
{
  return _haveValidOverlayData;
}

//
// getImageDimX
//
template <class ImagePixelType, class OverlayPixelType>
int
IView<ImagePixelType, OverlayPixelType>
::getImageDimX() const
{
  return _imageDim[0];
}

//
// getImageDimY
//
template <class ImagePixelType, class OverlayPixelType>
int
IView<ImagePixelType, OverlayPixelType>
::getImageDimY() const
{
  return _imageDim[1];
}

//
// getImageDimZ
//
template <class ImagePixelType, class OverlayPixelType>
int
IView<ImagePixelType, OverlayPixelType>
::getImageDimZ() const
{
  return _imageDim[2];
}

//
// getViewImage
//
template <class ImagePixelType, class OverlayPixelType>
bool
IView<ImagePixelType, OverlayPixelType>
::getViewImage() const
{
  return _viewImage;
}

//
// setViewImage
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setViewImage(bool view)
{
  _viewImage = view;
}

//
// getViewOverlay
//
template <class ImagePixelType, class OverlayPixelType>
bool
IView<ImagePixelType, OverlayPixelType>
::getViewOverlay() const
{
  return _viewOverlay;
}

//
// setViewOverlay
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setViewOverlay(bool view)
{
  _viewOverlay = view;
}

//
// getViewCrosshairs
//
template <class ImagePixelType, class OverlayPixelType>
bool
IView<ImagePixelType, OverlayPixelType>
::getViewCrosshairs() const
{
  return _viewCrosshairs;
}

//
// setViewCrosshairs
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setViewCrosshairs(bool view)
{
  _viewCrosshairs = view;
}

template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setViewImageInfo(bool view)
{
  _viewImageInfo = view;
}

//
// getViewROI
//
template <class ImagePixelType, class OverlayPixelType>
bool
IView<ImagePixelType, OverlayPixelType>
::getViewROI() const
{
  return _viewROI;
}

//
// setViewROI
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setViewROI(bool view)
{
  _viewROI = view;
}


//
// getViewMask
//
template <class ImagePixelType, class OverlayPixelType>
bool
IView<ImagePixelType, OverlayPixelType>
::getViewMask() const
{
  return _viewMask;
}

//
// setViewMask
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setViewMask(bool view)
{
  _viewMask = view;
}
//
// setCrosshairsOpacity
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setCrosshairsOpacity(const double& alpha)
{
  if (alpha < 0 || alpha > 1)
  {
    throw std::invalid_argument("invalid opacity");
  }
  _crosshairOpacity = alpha;
}

//
// setROIOpacity
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setROIOpacity(const double& alpha)
{
  if (alpha < 0 || alpha > 1)
  {
    throw std::invalid_argument("invalid opacity");
  }
  _roiOpacity = alpha;
}

//
// getOrientation
//
template <class ImagePixelType, class OverlayPixelType>
IViewOrientationType
IView<ImagePixelType, OverlayPixelType>
::getOrientation() const
{
  return _orientation;
}

//
// setOrientation
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setOrientation(IViewOrientationType orientation)
{
  if (orientation != IVIEW_SAGITTAL &&
    orientation != IVIEW_CORONAL &&
    orientation != IVIEW_AXIAL)
  {
    throw std::invalid_argument("invalid orientation");
  }
  
  _orientation = orientation;
  switch (orientation)
  {
  case IVIEW_SAGITTAL:
    _order[0] = 1;
    _order[1] = 2;
    _order[2] = 0;
    break;
  case IVIEW_CORONAL:
    _order[0] = 0;
    _order[1] = 2;
    _order[2] = 1;
    break;
  case IVIEW_AXIAL:
  default:
    _order[0] = 0;
    _order[1] = 1;
    _order[2] = 2;
    break;
  }
  
  _needImageUpdate = true;
  _needOverlayUpdate = true;
}

//
// getNumSlices
// 
template <class ImagePixelType, class OverlayPixelType>
int 
IView<ImagePixelType, OverlayPixelType>
::getNumSlices() const
{
  return _imageDim[_order[2]];
}

//
// getSliceIndex
//
template <class ImagePixelType, class OverlayPixelType>
int 
IView<ImagePixelType, OverlayPixelType>
::getSliceIndex() const
{
  return _windowCenter[_order[2]];
}

//
// setSliceIndex
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setSliceIndex(int index)
{
  _windowCenter[_order[2]] = index;
  
  if (_bufferedImageSliceIndex != index) 
  {
    _needImageUpdate = true;
    _needOverlayUpdate = true;
  }
}

template <class ImagePixelType, class OverlayPixelType>
Vector3D<unsigned int>
IView<ImagePixelType, OverlayPixelType>
::getWindowCenter() const
{
  return Vector3D<unsigned int>(
    (unsigned int)(_windowCenter[0]),
    (unsigned int)(_windowCenter[1]),
    (unsigned int)(_windowCenter[2]));
}

//
// setDataCenter
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setDataCenter(float xIndex, float yIndex, float zIndex)
{
  _windowCenter[0] = static_cast<int>(xIndex);//round 
  _windowCenter[1] = static_cast<int>(yIndex);//round 
  _windowCenter[2] = static_cast<int>(zIndex);//round 
  
  // when outside of the image in the negative value, round the
  // center to the closest negative value example : -0.25 round to -1
  // and not 0
  if (xIndex<0)
  {
    _windowCenter[0] = static_cast<int>(xIndex-1);//round 
  }
  if (yIndex<0)
  {
    _windowCenter[1] = static_cast<int>(yIndex-1);//round 
  }
  if (zIndex<0)
  {
    _windowCenter[2] = static_cast<int>(zIndex-1);//round 
  }
  _windowCenterf[0] = xIndex;
  _windowCenterf[1] = yIndex;
  _windowCenterf[2] = zIndex;
  
  if (_windowCenter[_order[2]] != _bufferedImageSliceIndex)
  {
    _needImageUpdate = true;
    _needOverlayUpdate = true;
  }
}

//
// getOverlayOpacity
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getOverlayOpacity() const
{
  return _overlayOpacity;
}

//
// setOverlayOpacity
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setOverlayOpacity(const double& opacity)
{
  if (opacity < 0 || opacity > 1)
  {
    throw std::invalid_argument("invalid opacity");
  }
  _overlayOpacity = opacity;
  
  if (_haveValidOverlayData)
  {
    int bufferSize = _overlayPixelBufferWidth * _overlayPixelBufferHeight * 2;
    unsigned char alpha = static_cast<unsigned char>(_overlayOpacity * 255.0);
    for (int i = 1; i < bufferSize; i+=2)
    {
      _overlayPixelBuffer[i] = alpha;
    }
  }
}

//
// setLineWidth
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setLineWidth(const double& lineWidth)
{
 _lineWidth = lineWidth;
}

//
// getImageColorR
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getImageColorR() const
{
  return _imageColorR;
}

//
// getImageColorG
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getImageColorG() const
{
  return _imageColorG;
}

//
// getImageColorB
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getImageColorB() const
{
  return _imageColorB;
}

//
// getOverlayColorR
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getOverlayColorR() const
{
  return _overlayColorR;
}

//
// getOverlayColorG
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getOverlayColorG() const
{
  return _overlayColorG;
}

//
// getOverlayColorB
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getOverlayColorB() const
{
  return _overlayColorB;
}

//
// setImageColorRGB
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setImageColorRGB(const double& r, const double& g, const double& b)
{
  if (r < 0 || r > 1 ||
    g < 0 || g > 1 ||
    b < 0 || b > 1)
  {
    throw std::invalid_argument("invalid color");
  }
  _imageColorR = r;
  _imageColorG = g;
  _imageColorB = b;
}

//
// setOverlayColorRGB
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setOverlayColorRGB(const double& r, const double& g, const double& b)
{
  if (r < 0 || r > 1 ||
    g < 0 || g > 1 ||
    b < 0 || b > 1)
  {
    throw std::invalid_argument("invalid color");
  }
  _overlayColorR = r;
  _overlayColorG = g;
  _overlayColorB = b;
}

//
// getImageDataMax
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getImageDataMax() const
{
  if (!_haveValidImageData)
  {
    throw std::runtime_error("IView::getImageDataMax: no image loaded.");
  }
  return _imageDataMax;
}

//
// getImageDataMin
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getImageDataMin() const
{
  if (!_haveValidImageData)
  {
    throw std::runtime_error("IView::getImageDataMin: no image loaded.");
  }
  return _imageDataMin;
}

//
// getOverlayDataMax
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getOverlayDataMax() const
{
  if (!_haveValidOverlayData)
  {
    throw std::runtime_error("IView::getOverlayDataMax: no overlay loaded.");
  }
  return _overlayDataMax;
}

//
// getOverlayDataMin
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getOverlayDataMin() const
{
  if (!_haveValidOverlayData)
  {
    throw std::runtime_error("IView::getOverlayDataMin: no overlay loaded.");
  }
  return _overlayDataMin;
}

//
// getImageIWMax
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getImageIWMax() const
{
  return _imageIWMax;
}

//
// setImageIWMax
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setImageIWMax(const double& max)
{
  if (_imageIWMax != max)
  {
    _imageIWMax = max;
    _needImageUpdate = true;
  }
}

//
// getImageIWMin
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getImageIWMin() const
{
  return _imageIWMin;
}

//
// setImageIWMin
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setImageIWMin(const double& min)
{
  if (_imageIWMin != min)
  {
    _imageIWMin = min;
    _needImageUpdate = true;
  }
}

//
// getOverlayIWMax
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getOverlayIWMax() const
{
  return _overlayIWMax;
}

//
// setOverlayIWMax
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setOverlayIWMax(const double& max)
{
  if (_overlayIWMax != max)
  {
    _overlayIWMax = max;
    _needOverlayUpdate = true;
  }
}

//
// getOverlayIWMin
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getOverlayIWMin() const
{
  return _overlayIWMin;
}

//
// setOverlayIWMin
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setOverlayIWMin(const double& min)
{
  if (_overlayIWMin != min)
  {
    _overlayIWMin = min;
    _needOverlayUpdate = true;
  }
}

//
// getWindowZoom
//
template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::getWindowZoom() const
{
  return _windowZoom;
}

//
// setWindowZoom
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::setWindowZoom(const double& zoom)
{
  if (zoom <= 0) 
  {
    throw std::invalid_argument("zoom <= 0");
  }
  _windowZoom = zoom;
}

//
// size (a la Fl_Gl_Window)
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::size(int w, int h)
{
  _windowWidth = w;
  _windowHeight = h;
  Fl_Gl_Window::size(w, h);
  
  this->redraw();
}

//
// resize (a la Fl_Gl_Window)
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::resize(int x, int y, int w, int h)
{
  _windowWidth = w;
  _windowHeight = h;

  _windowX = x;
  _windowY = y;

  Fl_Gl_Window::resize(x, y, w, h);
  
  this->redraw();
}

//
// handle (a la fltk)
//
template <class ImagePixelType, class OverlayPixelType>
int
IView<ImagePixelType, OverlayPixelType>
::handle(int event)
{
  switch (event)
  {
  case (FL_PUSH):
    {
      int button = Fl::event_button();
      int mouseX = Fl::event_x();
      int mouseY = Fl::event_y();
      float imgXf, imgYf, imgZf;
      _windowToImageIndexCoordinates(mouseX, mouseY, imgXf, imgYf, imgZf);
      /*int imgX = static_cast<int>(imgXf);
      int imgY = static_cast<int>(imgYf);
      int imgZ = static_cast<int>(imgZf);*/
      
      // Apparently, as of FLTK 1.3, FLTK does not capture the standard
      // MacOS X second and third mouse events.  We, therefore, simulate them
      // with the left shift key (third mouse button) and the left Apple key
      // (middle mouse).  FLTK interprets the left Apple key as the left
      // control (which is an entirely different key)
#if defined(__APPLE__)
      if (Fl::event_key(FL_Shift_L)) {
        button = 3;
      }
      if (Fl::event_key(FL_Control_L)) {
        button = 2;
      }
#endif
      if (_clickCallback != 0)
      {
        _clickCallback(button, imgXf, imgYf, imgZf, _clickCallbackArg);
      }
      
      switch (button)
      {
      case 1:
        float worldX, worldY, worldZ;
        _imageData->imageIndexToWorldCoordinates(imgXf, imgYf, imgZf,
                                                 worldX, worldY, worldZ);
        // output click coordinate info
        std::cerr << "Window Coords: [" 
          << mouseX << ", " << mouseY << "]" << std::endl;
        std::cerr << "Image Coords: [" 
          << imgXf << ", " << imgYf << ", " << imgZf << "]" << std::endl;
        std::cerr << "World Coords: [" << worldX << ", " << worldY << ", "
                  << worldZ <<"]" << std::endl; 
        break;
      case 2:
        // currently handled via callback
        //setDataCenter(imgX, imgY, imgZ);
        //redraw();
        break;
      case 3:
        // roi
        if (_ROILocked == false)
        {
          _updateROIDraggingFlags(true, mouseX, mouseY);
        }
        break;
      default:
        break;
      }
      return 1;
      break;
    }
  case (FL_DRAG):
    {
      int button = Fl::event_button();
      int mouseX = Fl::event_x();
      int mouseY = Fl::event_y();
      if (button == 3)
      {
        //roi
        if (_ROILocked == false)
        {
          _dragROI(mouseX, mouseY);
        }
      }
      return 1;
      break;
    }
    
  case (FL_RELEASE):
    {
      int button = Fl::event_button();
      int mouseX = Fl::event_x();
      int mouseY = Fl::event_y();
      if (button == 3)
      {
        // roi
        if (_ROILocked == false)
        {
          _updateROIDraggingFlags(false, mouseX, mouseY);
        }
      }
      return 1;
    }
    break;

  default:
    if (_verbose)
    {
      //std::cerr << "IView event: " << event << std::endl;
    }
    return 0;
    break;
  }
}

//
// draw (a la Fl_Gl_Window)
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::draw(void)
{
  gl_font(FL_COURIER, 13); 
  if (!valid())
  {
    // this must be present or the images will distort
    // in certain cases (DONT REMOVE)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    
    // set up orientation
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    // ortho is a Fl_Gl_Window call that makes origin lower left    
    //ortho();    
    GLint v[2];
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, v);
    glLoadIdentity();
    glViewport(_windowWidth-v[0]/2, _windowHeight-v[1]/2, v[0]*3/2, v[1]*3/2);
    glOrtho(_windowWidth-v[0]/2, 
            _windowWidth+v[0]/2, 
            _windowHeight-v[1]/2, 
            _windowHeight+v[1]/2, 
            -1, 1);
  }
  //Trying to fix ATI bug
  //glDisable(GL_DEPTH_TEST);
  
  // clear window
  glClearColor((float)0.0, (float)0.0, (float)0.0, (float)0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  // draw the images
  if ((_haveValidImageData && _viewImage) ||
      (_haveValidOverlayData && _viewOverlay))
  {
    // update pixel buffers as necessary
    update();
    
    GLfloat oldPixelTransferR, oldPixelTransferG, oldPixelTransferB;
    glGetFloatv(GL_RED_SCALE,   &oldPixelTransferR);
    glGetFloatv(GL_GREEN_SCALE, &oldPixelTransferG);
    glGetFloatv(GL_BLUE_SCALE,  &oldPixelTransferB);

    // draw the image
    if (_haveValidImageData && _viewImage)
    {
      // set zoom and translation
      _setupGLDrawPixelsImage();
      glPixelTransferf(GL_RED_SCALE,   _imageColorR);
      glPixelTransferf(GL_GREEN_SCALE, _imageColorG);
      glPixelTransferf(GL_BLUE_SCALE,  _imageColorB);
      glDrawPixels(_imagePixelBufferWidth, _imagePixelBufferHeight,
                   GL_LUMINANCE, GL_UNSIGNED_BYTE,
                   _imagePixelBuffer);
    }    
    
    // draw the overlay
    if (_haveValidOverlayData && _viewOverlay)
    {
      // set zoom and translation
      _setupGLDrawPixelsOverlay();
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glPixelTransferf(GL_RED_SCALE,   _overlayColorR);
      glPixelTransferf(GL_GREEN_SCALE, _overlayColorG);
      glPixelTransferf(GL_BLUE_SCALE,  _overlayColorB);
      //glPixelTransferf(GL_ALPHA_SCALE, _overlayOpacity);
      glDrawPixels(_overlayPixelBufferWidth, _overlayPixelBufferHeight, 
                   GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, 
                   _overlayPixelBuffer);
      glDisable(GL_BLEND);
    }    

    glPixelTransferf(GL_RED_SCALE,   oldPixelTransferR);
    glPixelTransferf(GL_GREEN_SCALE, oldPixelTransferG);
    glPixelTransferf(GL_BLUE_SCALE,  oldPixelTransferB);
  }
  
  // draw roi
  if (_viewROI)
  {
    _drawROI();
  }

  // draw mask
  if (_viewMask)
  {
    _drawMask();
  }
  
  // draw overlay contours
  if (_viewOverlayContours)
  {
    _drawOverlayContours();
  }
  
  // draw image contours
  if (_viewImageContours)
  {
    _drawImageContours();
  }
  
  // draw crosshairs (always at center of window)
  if (_viewCrosshairs)
  {
    _drawCrosshairs();
  }
  
  // draw image info
  if (_viewImageInfo && _haveValidImageData)
  {
    _drawImageInfo();
  }
  
  //glEnable(GL_DEPTH_TEST);
  
  // check for gl errors
  _checkForGLError("[IView::draw]: ");
}

template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_drawImageContours()
{
  switch (_orientation)
  {
  case(IVIEW_SAGITTAL): // (y , z)
  {
    for (std::vector<IViewAnastructGroup>::iterator anaGroupIter =
           _imageAnastructs.begin();
         anaGroupIter != _imageAnastructs.end();
         ++anaGroupIter)
    {
      Anastruct *currAnastruct = &anaGroupIter->anastruct;
      if (anaGroupIter->isVisible)
      {
        for (unsigned int contourIndex = 0; 
             contourIndex < currAnastruct->contours.size();
             ++contourIndex)
        {
          // draw this contour
          glColor3f(anaGroupIter->r, anaGroupIter->g, anaGroupIter->b);
          glLineWidth(_lineWidth);
          glBegin(GL_LINES);
            
          for (unsigned int vertexIndex = 0; 
               vertexIndex < (currAnastruct->
                              contours[contourIndex].vertices.size()-1);
               vertexIndex++)
          {
            double VertexX = currAnastruct->
                             contours[contourIndex].vertices[vertexIndex].x;
            double nextVertexX = currAnastruct->
                                 contours[contourIndex].vertices[vertexIndex + 1].x;
            double VertexY = currAnastruct->
                             contours[contourIndex].vertices[vertexIndex].y;
            double nextVertexY = currAnastruct->
                             contours[contourIndex].vertices[vertexIndex + 1].y;
            // test to know if we intersect a vertex
            float alpha = (_windowCenter[0] - VertexX)/(nextVertexX - VertexX);
            if ((alpha>=0)&&(alpha<=1))
            {
              float IntersectX = _windowCenter[0];
              float IntersectY = static_cast<unsigned int>(alpha*nextVertexY +
                                                           (1-alpha)*VertexY);
              //because of the capping the slice number can be negative
              int IntersectZ = currAnastruct->contours[contourIndex].sliceNumber;
                
                
              float windowX, windowY;
              _imageIndexToWindowCoordinates(IntersectX,
                                             IntersectY,
                                             IntersectZ,
                                             windowX, windowY);
              glVertex2f(windowX, windowY);
              _imageIndexToWindowCoordinates(IntersectX,
                                             IntersectY,
                                             IntersectZ+1,
                                             windowX, windowY);
              glVertex2f(windowX, windowY);
                
            }
          }
          glEnd(); 
        }
      }
    }
    break;
  }
  case(IVIEW_CORONAL): // (x , z)
  {
    for (std::vector<IViewAnastructGroup>::iterator anaGroupIter = _imageAnastructs.begin();
         anaGroupIter != _imageAnastructs.end();
         ++anaGroupIter)
    {
      Anastruct *currAnastruct = &anaGroupIter->anastruct;
      if (anaGroupIter->isVisible)
      {
        for (unsigned int contourIndex = 0; 
             contourIndex < currAnastruct->contours.size();
             ++contourIndex)
        {
            
          // draw this contour
          glColor3f(anaGroupIter->r, anaGroupIter->g, anaGroupIter->b);
          glLineWidth(_lineWidth);
          glBegin(GL_LINES);
          for (unsigned int vertexIndex = 0; 
               vertexIndex < (currAnastruct->contours[contourIndex].vertices.size()-1);
               vertexIndex++)
          {
            double VertexX = currAnastruct->contours[contourIndex].vertices[vertexIndex].x;
            double nextVertexX = currAnastruct->contours[contourIndex].vertices[vertexIndex + 1].x;
            double VertexY = currAnastruct->contours[contourIndex].vertices[vertexIndex].y;
            double nextVertexY = currAnastruct->contours[contourIndex].vertices[vertexIndex + 1].y;
            // test to know if we intersect a vertex
            float alpha = (_windowCenter[1] - VertexY)/(nextVertexY - VertexY);
            if ((alpha>=0)&&(alpha<=1))
            {
              float IntersectX = static_cast<unsigned int>(alpha*nextVertexX + (1-alpha)*VertexX);
              float IntersectY = _windowCenter[1];
              //because of the caping the slice number can be negative
              float IntersectZ = currAnastruct->contours[contourIndex].sliceNumber;
                
                
              float windowX, windowY;
              _imageIndexToWindowCoordinates(IntersectX,
                                             IntersectY,
                                             IntersectZ,
                                             windowX, windowY);
              glVertex2f(windowX, windowY);
              _imageIndexToWindowCoordinates(IntersectX,
                                             IntersectY,
                                             IntersectZ+1,
                                             windowX, windowY);
              glVertex2f(windowX, windowY);
                
            }
          }
          glEnd(); 
        }
      }
    }
    break;
  }
  case(IVIEW_AXIAL): //(x , y)
  {
    for (std::vector<IViewAnastructGroup>::iterator anaGroupIter 
           = _imageAnastructs.begin();
         anaGroupIter != _imageAnastructs.end();
         ++anaGroupIter)
    {
      Anastruct *currAnastruct = &anaGroupIter->anastruct;
      if (anaGroupIter->isVisible)
      {
        for (unsigned int contourIndex = 0; 
             contourIndex < currAnastruct->contours.size();
             ++contourIndex)
        {
          if (currAnastruct->contours[contourIndex].sliceNumber == 
              _windowCenter[2])
          {
            // draw this contour
            glColor3f(anaGroupIter->r, 
                      anaGroupIter->g, 
                      anaGroupIter->b);

            // tmp hack
            // if (currAnastruct->contours[contourIndex].vertices.size() < 20)
            // {
            //   glColor3f(0,1,0); 
            // }

            glLineWidth(_lineWidth);
            //glBegin(GL_LINE_LOOP);
            glBegin(GL_LINE_STRIP);
            for (unsigned int vertexIndex = 0; 
                 vertexIndex < currAnastruct->
                               contours[contourIndex].vertices.size();
                 vertexIndex++)
            {
              float windowX, windowY;
              _imageIndexToWindowCoordinates(
                currAnastruct->contours[contourIndex].vertices[vertexIndex].x,
                currAnastruct->contours[contourIndex].vertices[vertexIndex].y,
                contourIndex, windowX, windowY);
              glVertex2f(windowX, windowY);
            }
            glEnd();  

/*
            glPointSize(3);
            glBegin(GL_POINTS);
            for (unsigned int vertexIndex = 0; 
                 vertexIndex < currAnastruct->contours[contourIndex].
                                            vertices.size();
                 vertexIndex++)
            {
              float windowX, windowY;
              _imageIndexToWindowCoordinates(
                                             currAnastruct->contours[contourIndex].vertices[vertexIndex].x,
                                             currAnastruct->contours[contourIndex].vertices[vertexIndex].y,
                                             contourIndex, windowX, windowY);
              glVertex2f(windowX, windowY);
            }
            glEnd();  
*/
          }
        }
      }
    }
    break;
  }
    
  default:
    throw std::runtime_error("invalid orientation");
  }
}

template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_drawOverlayContours()
{
  glLineStipple(1,0xAAAA);
  glEnable(GL_LINE_STIPPLE);
  switch (_orientation)
  {
  case(IVIEW_SAGITTAL): // (y , z)
    {
      if (_haveValidImageData && _haveValidOverlayData && _checkImageCompatibility(_imageData, _overlayData))
      {
        for (std::vector<IViewAnastructGroup>::iterator anaGroupIter = _overlayAnastructs.begin();
        anaGroupIter != _overlayAnastructs.end();
        ++anaGroupIter)
        {
          Anastruct *currAnastruct = &anaGroupIter->anastruct;
          if (anaGroupIter->isVisible)
          {
            for (unsigned int contourIndex = 0; 
            contourIndex < currAnastruct->contours.size();
            ++contourIndex)
            {
              // draw this contour
              glColor3f(anaGroupIter->r, anaGroupIter->g, anaGroupIter->b);
              glLineWidth(_lineWidth);
              glBegin(GL_LINES);
              
              for (unsigned int vertexIndex = 0; 
              vertexIndex < (currAnastruct->contours[contourIndex].vertices.size()-1);
              vertexIndex++)
              {
                double VertexX = currAnastruct->contours[contourIndex].vertices[vertexIndex].x;
                double nextVertexX = currAnastruct->contours[contourIndex].vertices[vertexIndex + 1].x;
                double VertexY = currAnastruct->contours[contourIndex].vertices[vertexIndex].y;
                double nextVertexY = currAnastruct->contours[contourIndex].vertices[vertexIndex + 1].y;
                // test to know if we intersect a vertex
                float alpha = (_windowCenter[0] - VertexX)/(nextVertexX - VertexX);
                if ((alpha>=0)&&(alpha<=1))
                {
                  float IntersectX = _windowCenter[0];
                  float IntersectY = alpha*nextVertexY + (1-alpha)*VertexY;
                  float IntersectZ = currAnastruct->contours[contourIndex].sliceNumber;
                  
                  
                  float windowX, windowY;
                  _imageIndexToWindowCoordinates(IntersectX,
                    IntersectY,
                    IntersectZ,
                    windowX, windowY);
                  glVertex2f(windowX, windowY);
                  _imageIndexToWindowCoordinates(IntersectX,
                    IntersectY,
                    IntersectZ+1,
                    windowX, windowY);
                  glVertex2f(windowX, windowY);
                  
                }
              }
              glEnd(); 
            }
          }
        }
        break;
      }
    }
    
  case(IVIEW_CORONAL): // (x , z)
    {
      if (_haveValidImageData && _haveValidOverlayData && _checkImageCompatibility(_imageData, _overlayData))
      {
        for (std::vector<IViewAnastructGroup>::iterator anaGroupIter = _overlayAnastructs.begin();
        anaGroupIter != _overlayAnastructs.end();
        ++anaGroupIter)
        {
          Anastruct *currAnastruct = &anaGroupIter->anastruct;
          if (anaGroupIter->isVisible)
          {
            for (unsigned int contourIndex = 0; 
            contourIndex < currAnastruct->contours.size();
            ++contourIndex)
            {
              
              // draw this contour
              glColor3f(anaGroupIter->r, anaGroupIter->g, anaGroupIter->b);
              glLineWidth(_lineWidth);
              glBegin(GL_LINES);
              for (unsigned int vertexIndex = 0; 
              vertexIndex < (currAnastruct->contours[contourIndex].vertices.size()-1);
              vertexIndex++)
              {
                double VertexX = currAnastruct->contours[contourIndex].vertices[vertexIndex].x;
                double nextVertexX = currAnastruct->contours[contourIndex].vertices[vertexIndex + 1].x;
                double VertexY = currAnastruct->contours[contourIndex].vertices[vertexIndex].y;
                double nextVertexY = currAnastruct->contours[contourIndex].vertices[vertexIndex + 1].y;
                // test to know if we intersect a vertex
                float alpha = (_windowCenter[1] - VertexY)/(nextVertexY - VertexY);
                if ((alpha>=0)&&(alpha<=1))
                {
                  float IntersectX = static_cast<unsigned int>(alpha*nextVertexX + (1-alpha)*VertexX);
                  float IntersectY = _windowCenter[1];
                  float IntersectZ = currAnastruct->contours[contourIndex].sliceNumber;
                  
                  
                  float windowX, windowY;
                  _imageIndexToWindowCoordinates(IntersectX,
                    IntersectY,
                    IntersectZ,
                    windowX, windowY);
                  glVertex2f(windowX, windowY);
                  _imageIndexToWindowCoordinates(IntersectX,
                    IntersectY,
                    IntersectZ+1,
                    windowX, windowY);
                  glVertex2f(windowX, windowY);
                  
                }
              }
              glEnd(); 
            }
          }
        }
        break;
      }
    }
  case(IVIEW_AXIAL): // xy
    {
      if (_haveValidImageData && _haveValidOverlayData && _checkImageCompatibility(_imageData, _overlayData))
      {
        for (std::vector<IViewAnastructGroup>::iterator anaGroupIter = _overlayAnastructs.begin();
        anaGroupIter != _overlayAnastructs.end();
        ++anaGroupIter)
        {
          Anastruct *currAnastruct = &anaGroupIter->anastruct;
          if (anaGroupIter->isVisible)
          {
            for (unsigned int contourIndex = 0; 
            contourIndex < currAnastruct->contours.size();
            contourIndex++)
            {
              if (currAnastruct->contours[contourIndex].sliceNumber == 
                _windowCenter[2])
              {
                // draw this contour
                glColor3f(anaGroupIter->r, anaGroupIter->g, anaGroupIter->b);
                glLineWidth(_lineWidth);
                glBegin(GL_LINE_LOOP);
                for (unsigned int vertexIndex = 0; 
                vertexIndex < currAnastruct->contours[contourIndex].vertices.size();
                vertexIndex++)
                {
                  float windowX, windowY;
                  _imageIndexToWindowCoordinates(currAnastruct->contours[contourIndex].vertices[vertexIndex].x,
                    currAnastruct->contours[contourIndex].vertices[vertexIndex].y,
                    contourIndex,
                    windowX, windowY);
                  glVertex2f(windowX, windowY);
                }
                glEnd();  
              }
            }
          }
        }
      }
    }
    break;
  default:
    throw std::runtime_error("invalid orientation");
    }
    glDisable(GL_LINE_STIPPLE);
}

template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_drawImageInfo()
{
  if (_haveValidImageData)
  {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f((float)0.9, (float)0.9, (float)0.5, (float)0.8);
    
    
    // the gl_* functions are fltk gl interface functions
    //gl_font(FL_HELVETICA, 13); 
    
    _drawString(_imageName, 5, _windowHeight - 15); 
    
    std::ostringstream ossWorld;
    ossWorld << "W: [" 
      << _windowCenter[0] * _imageData->getSpacing()[0] + _imageData->getOrigin()[0] << ", "
      << _windowCenter[1] * _imageData->getSpacing()[1] + _imageData->getOrigin()[1] << ", "
      << _windowCenter[2] * _imageData->getSpacing()[2] + _imageData->getOrigin()[2] << "] ";
    _drawString(ossWorld.str(), 5, _windowHeight - 30); 
    
    std::ostringstream ossIndex;
    ossIndex << "I: [" 
      << _windowCenter[0] << ", "
      << _windowCenter[1] << ", "
      << _windowCenter[2] << "]";
    _drawString(ossIndex.str(), 5, _windowHeight - 45); 
    
    if (_windowCenter[0] >= 0 && _windowCenter[0] < _imageDim[0] &&
      _windowCenter[1] >= 0 && _windowCenter[1] < _imageDim[1] &&
      _windowCenter[2] >= 0 && _windowCenter[2] < _imageDim[2])
    {
      std::ostringstream ossValue;
      ossValue << _imageData->getDataPointer()[
        _windowCenter[2] * (_imageDim[0] * _imageDim[1]) +
        _windowCenter[1] * _imageDim[0] + _windowCenter[0]
        ];	  
      _drawString(ossValue.str(), 5, _windowHeight - 60); 
    }

    std::string left("L");       // left of the patient
    std::string right("R");      // right of the patient
    std::string anterior("A");   // anterior of the patient
    std::string posterior("P");  // posterior of the patient
    std::string superior("S");   // superior of the patient
    std::string inferior("I");   // inferior of the patient
    switch (_orientation)
    {
    case IVIEW_SAGITTAL: // (y,z)
      _drawString(posterior,5,_windowHeight/2);// left of the image
      _drawString(anterior,_windowWidth-15, _windowHeight/2);
      _drawString(superior,_windowWidth/2, _windowHeight - 15);// up of the image
      _drawString(inferior,_windowWidth/2, 15);
      break;
    case IVIEW_CORONAL: // (x,z)
      _drawString(left, _windowWidth-15, _windowHeight/2);// right of the image
      _drawString(right, 5, _windowHeight/2);
      _drawString(superior, _windowWidth/2, _windowHeight - 15);// up of the image
      _drawString(inferior, _windowWidth/2, 5);
      break;
    case IVIEW_AXIAL: // (x,y)
      _drawString(left, _windowWidth-15, _windowHeight/2);// right of the image
      _drawString(right, 5, _windowHeight/2);
      _drawString(anterior, _windowWidth/2, _windowHeight - 15);// up of the image
      _drawString(posterior, _windowWidth/2, 5);
      break;
    }
    
    glDisable(GL_BLEND);
  }
  else
  {
  }
}

template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_drawCrosshairs()
{
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glColor4f((float)0.1, (float)0.64, (float)0.2, _crosshairOpacity);
  
  double yMid = static_cast<double>(_windowHeight) / 2.0;
  double xMid = static_cast<double>(_windowWidth) / 2.0;
  
  glLineWidth(1.0);
  glBegin(GL_LINES);
  glVertex2d(0, yMid);
  glVertex2d(xMid - 2, yMid);
  glVertex2d(xMid + 2, yMid);
  glVertex2d(_windowWidth - 1, yMid);
  glVertex2d(xMid, 0);
  glVertex2d(xMid, yMid - 2);
  glVertex2d(xMid, yMid + 2);
  glVertex2d(xMid, _windowHeight - 1);
  glEnd();
  
  glDisable(GL_BLEND);
}

template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_drawROI()
{
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  float winX1, winY1;
  _imageIndexToWindowCoordinates(_roi.start[0], 
    _roi.start[1], 
    _roi.start[2],
    winX1, winY1);
  
  // +1 because window coord is to min side of index
  // we want max side (i.e. min side of next index)
  float winX2, winY2;
  _imageIndexToWindowCoordinates(_roi.stop[0], 
    _roi.stop[1] , 
    _roi.stop[2],
    winX2, winY2);
  
  glColor4f((float)0.1, (float)0.2, (float)0.64, (float)1);
  
  // draw the outline of the roi
  glBegin(GL_LINE_LOOP);
  glVertex2f(winX1, winY1);
  glVertex2f(winX1, winY2);
  glVertex2f(winX2, winY2);
  glVertex2f(winX2, winY1);
  glEnd();
  
  // fill in the roi if this slice is contained in the roi
  if (_windowCenter[_order[2]] >= _roi.start[_order[2]] &&
    _windowCenter[_order[2]] <= _roi.stop[_order[2]])
  {
    glColor4f((float)0.1, (float)0.2, (float)0.64, _roiOpacity);
    glBegin(GL_POLYGON);
    glVertex2f(winX1, winY1);
    glVertex2f(winX1, winY2);
    glVertex2f(winX2, winY2);
    glVertex2f(winX2, winY1);
    glEnd();
  }
  
  glDisable(GL_BLEND);
}

template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_drawMask()
{
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  float winX1, winY1, winX2, winY2;
  unsigned int x,y,z;
  double maskOpacity = 0.5;
  
  switch (_orientation)
  {
  case(IVIEW_SAGITTAL): // sagittal (y , z)
    {
      x = _windowCenter[0];
      for (z = 0 ; z < _mask.getSize().z ; z++){
        for (y = 0 ; y < _mask.getSize().y ; y++){
          _imageIndexToWindowCoordinates(x, y, z,winX1, winY1);
          _imageIndexToWindowCoordinates(x + 1, y + 1, z + 1,winX2, winY2);
          if(_mask.get(x,y,z) == 1.0 )
          {
            glColor4f((float)0, (float)1.0, (float)0, maskOpacity);
            glBegin(GL_POLYGON);
            glVertex2f(winX1, winY1);
            glVertex2f(winX1, winY2);
            glVertex2f(winX2, winY2);
            glVertex2f(winX2, winY1);
            glEnd();
          }
        }
      }
      break;
    }
  case(IVIEW_CORONAL): // coronal (x , z)
    {
      y = _windowCenter[1];
      for (z = 0 ; z < _mask.getSize().z ; z++){
        for (x = 0 ; x < _mask.getSize().x ; x++){
          _imageIndexToWindowCoordinates(x, y, z,winX1, winY1);
          _imageIndexToWindowCoordinates(x + 1, y + 1, z + 1,winX2, winY2);
          if(_mask.get(x,y,z) == 1.0 )
          {
            glColor4f((float)0, (float)1.0, (float)0, maskOpacity);
            glBegin(GL_POLYGON);
            glVertex2f(winX1, winY1);
            glVertex2f(winX1, winY2);
            glVertex2f(winX2, winY2);
            glVertex2f(winX2, winY1);
            glEnd();
          }
        }
      }
      break;
    }
  case(IVIEW_AXIAL): // axial (x , y)
    {
      z = _windowCenter[2];
      for (y = 0 ; y < _mask.getSize().y ; y++){
        for (x = 0 ; x < _mask.getSize().x ; x++){
          _imageIndexToWindowCoordinates(x, y, z,winX1, winY1);
          _imageIndexToWindowCoordinates(x + 1, y + 1, z + 1,winX2, winY2);
          if(_mask.get(x,y,z) == 1.0 )
          {
            glColor4f((float)0, (float)1.0, (float)0, maskOpacity);
            glBegin(GL_POLYGON);
            glVertex2f(winX1, winY1);
            glVertex2f(winX1, winY2);
            glVertex2f(winX2, winY2);
            glVertex2f(winX2, winY1);
            glEnd();
          }
        }
      }
      break;
    }
  }
  
  glDisable(GL_BLEND);
  // check for gl errors
  _checkForGLError("[IView::drawMask]: ");
}

template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_drawString(const std::string& text, const float& x, const float& y)
{
  glRasterPos2f(x, y);
  glCallLists(text.size(), GL_UNSIGNED_BYTE, text.c_str());
}

//
// update
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::update()
{
  if (_haveValidImageData && _needImageUpdate)
  {
    Timer t;
    t.start();
    _updateImagePixelBuffer();
    t.stop();
    // std::cerr << "[" << _order[2]
      // << "] Update Image Buffer (msec)   " 
      // << t.getMilliseconds() << std::endl;
  }
  if (_haveValidOverlayData && _needOverlayUpdate)
  {
    Timer t;
    t.start();
    _updateOverlayPixelBuffer();
    t.stop();
    // std::cerr << "[" << _order[2]
      // << "] Update Overlay Buffer (msec) " 
      // << t.getMilliseconds() << std::endl;
  }
}

//
// setClickCallback
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setClickCallback(void (*callbackFcn)(int button, 
                   float xIndex, 
                   float yIndex, 
                   float zIndex,
                   void *arg),
                   void *callbackArg)
{
  _clickCallback = callbackFcn;
  _clickCallbackArg = callbackArg;
}

//
// setROIChangedCallback
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setROIChangedCallback(
  void (*callbackFcn)(IViewOrientationType orientation, void *arg),
  void *callbackArg)
{
  _roiChangedCallback = callbackFcn;
  _roiChangedCallbackArg = callbackArg;
}

//
// setROI
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setROI(int startX, int startY, int startZ,
         int stopX, int stopY, int stopZ)
{
  // make sure roi start < stop
  if (startX > stopX)
  {
    int tmp = startX;
    startX = stopX;
    stopX = tmp;
  }
  if (startY > stopY)
  {
    int tmp = startY;
    startY = stopY;
    stopY = tmp;
  }
  if (startZ > stopZ)
  {
    int tmp = startZ;
    startZ = stopZ;
    stopZ = tmp;
  }
  
  _roi.start[0] = startX;
  _roi.start[1] = startY;
  _roi.start[2] = startZ;  
  
  _roi.stop[0] = stopX;
  _roi.stop[1] = stopY;
  _roi.stop[2] = stopZ;  
}


//
// setMask
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::setMask( const MaskType& mask)
{
  _mask.resize(mask.getSize());
  _mask.setData(mask);
}

//
// getROI
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::getROI(int& startX, int& startY, int& startZ,
         int& stopX, int& stopY, int& stopZ) const
{
  startX = _roi.start[0];
  startY = _roi.start[1];
  startZ = _roi.start[2];
  
  stopX = _roi.stop[0];
  stopY = _roi.stop[1];
  stopZ = _roi.stop[2];
}

//
// isImageCompatible
//
template <class ImagePixelType, class OverlayPixelType>
bool
IView<ImagePixelType, OverlayPixelType>
::isImageCompatible(const ImagePointer& image) const
{
  if (_haveValidOverlayData)
  {
    return _checkImageCompatibility(image, _overlayData);
  }
  else 
  {
    return true;
  }
}

//
// isOverlayCompatible
//
template <class ImagePixelType, class OverlayPixelType>
bool
IView<ImagePixelType, OverlayPixelType>
::isOverlayCompatible(const ImagePointer& overlay) const
{
  if (_haveValidImageData)
  {
    return _checkImageCompatibility(_imageData, overlay);
  }
  else 
  {
    return true;
  }
}

//
// saveWindowAsImage
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::saveWindowAsImage(const char *filename) 
{
  make_current();

  gl_font(FL_COURIER, 13); 
  if (!valid())
  {
    // this must be present or the images will distort
    // in certain cases (DONT REMOVE)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    
    // set up orientation
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    // ortho is a Fl_Gl_Window call that makes origin lower left    
    //ortho();    
    GLint v[2];
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, v);
    glLoadIdentity();
    glViewport(_windowWidth-v[0]/2, _windowHeight-v[1]/2, v[0]*3/2, v[1]*3/2);
    glOrtho(_windowWidth-v[0]/2, 
            _windowWidth+v[0]/2, 
            _windowHeight-v[1]/2, 
            _windowHeight+v[1]/2, 
            -1, 1);
    valid(1);
  }

  GLint OldReadBuffer;
  glGetIntegerv(GL_READ_BUFFER,&OldReadBuffer);
  glReadBuffer(GL_FRONT);

  GLint OldPackAlignment;
  glGetIntegerv(GL_PACK_ALIGNMENT,&OldPackAlignment);
  glPixelStorei(GL_PACK_ALIGNMENT,1);

  int WW = _windowWidth;
  int WH = _windowHeight;
  int NumPixels = WW*WH;

  printf("%d x %d: %s\n",WW,WH,filename);

  GLubyte* Pixels = new GLubyte[NumPixels*3];
  glReadPixels(0,0,WW,WH,GL_RGB,GL_UNSIGNED_BYTE,Pixels);

  std::string ppmFilename = StringUtils::forcePathExtension(filename, "ppm");
  FILE* fp = fopen(ppmFilename.c_str(), "wb");
  fprintf(fp, "P6\n%d %d\n255\n", WW, WH);
  for (int i=((WH-1)*WW)*3; i>=0; i-=(WW*3))
    fwrite(&(Pixels[i]),1,WW*3,fp);
  fclose(fp);

  delete[] Pixels;

  glPixelStorei(GL_PACK_ALIGNMENT,OldPackAlignment);
  glReadBuffer(OldReadBuffer); 
}

#ifdef NOT_DEFINED
//
// _updateImagePixelBuffer
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_updateImagePixelBuffer(void)
{
  if (!_haveValidImageData)
  {
    return;
  }
  
  // we only need to fill in slices that fall into the data region
  if (_windowCenter[_order[2]] >= 0 &&
      _windowCenter[_order[2]] < _imageDim[_order[2]])
  {
    // set up image region iterator
    ImageSizeType regionSize;
    regionSize[_order[0]] = _imageDim[_order[0]];
    regionSize[_order[1]] = _imageDim[_order[1]];
    regionSize[_order[2]] = 1;
    
    ImageIndexType regionIndex;
    regionIndex[_order[0]] = 0;
    regionIndex[_order[1]] = 0;
    regionIndex[_order[2]] = _windowCenter[_order[2]];
    
    ImageRegionType imageSliceRegion;
    imageSliceRegion.setStart(regionIndex);
    imageSliceRegion.setSize(regionSize);
    
    // fill in the pixel buffer
    float currValue;
    unsigned char *bufferPtr = _imagePixelBuffer;
    
    // !!! what to do if divide by zero?
    float scaleFactor = 255.0 / (_imageIWMax - _imageIWMin);
    
    for (int x = imageSliceRegion.getStart().x;
         x < imageSliceRegion.getStop().x; x++){
      for(int y= imageSliceRegion.getStart().y;
          y < imageSliceRegion.getStop().y; y++){
        
        currValue = (_imageData->get(x,y,_windowCenter[_order[2]]) -
                     _imageIWMin) * scaleFactor;
        
        // make sure currValue is in range [0,255]
        if (currValue > 255)
        {
          currValue = 255;
        }
        else if (currValue < 0)
        {
          currValue = 0;
        }
        *bufferPtr++ = static_cast<unsigned char>(currValue);
        //++fromIter;
      }
      
    }
  }
  else
  {
    // set image to background
    memset(_imagePixelBuffer, 0, 
           _imagePixelBufferWidth * _imagePixelBufferHeight);
  }
  
  _bufferedImageSliceIndex = _windowCenter[_order[2]];
  _needImageUpdate = false;
}
#endif

//
// _updateImagePixelBuffer
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_updateImagePixelBuffer(void)
{
  if (!_haveValidImageData)
  {
    return;
  }
  
  // we only need to fill in slices that fall into the data region
  if (_windowCenter[_order[2]] >= 0 &&
      _windowCenter[_order[2]] < _imageDim[_order[2]])
  {
    unsigned char  *pixelBuffer2D = _imagePixelBuffer;      
    unsigned int pixelBufferSize  = _imagePixelBufferWidth
      * _imagePixelBufferHeight;
    ImagePixelType currentPixelValue;
    float iwScaleFactor = 255.0 / (_imageIWMax - _imageIWMin);
    
    switch(_orientation)
    {
    case(IVIEW_SAGITTAL):
      {
        // sagittal (y, z)
        ImagePixelType *pixelBuffer3D = 
          &_imageData->getDataPointer()[_windowCenter[0]];   
        unsigned int xSize = _imageDim[0];
        
        for (unsigned int pixelIndex = 0;
             pixelIndex < pixelBufferSize;
             ++pixelIndex)
        {
          currentPixelValue = 
            (*pixelBuffer3D - _imageIWMin) * iwScaleFactor;
          pixelBuffer3D += xSize;
          
          // clamp to [0,255]
          if (currentPixelValue < 0) currentPixelValue = 0;
          else if (currentPixelValue > 255) currentPixelValue = 255;
          
          *pixelBuffer2D++ = 
            static_cast<unsigned char>(currentPixelValue);
        }
      }
      break;
    case(IVIEW_CORONAL):
      // coronal (x, z)
      {
        ImagePixelType *pixelBuffer3D = 
          &_imageData->getDataPointer()[_imagePixelBufferWidth 
                                        * _windowCenter[1]];   
        unsigned int bufferIncrement =
          (_imageDim[0] * _imageDim[1]) - _imageDim[0]; 
        for (unsigned int zIndex = 0, zIndexEnd = _imageDim[2];
             zIndex < zIndexEnd;
             ++zIndex)
          {
            for (unsigned int xIndex = 0, xIndexEnd = _imageDim[0];
                 xIndex < xIndexEnd;
                 ++xIndex)		
              {
                currentPixelValue = 
                  (*pixelBuffer3D++ - _imageIWMin) * iwScaleFactor;
                // clamp to [0,255]
                if (currentPixelValue < 0) currentPixelValue = 0;
                else if (currentPixelValue > 255) currentPixelValue = 255;
                
                *pixelBuffer2D++ = 
                  static_cast<unsigned char>(currentPixelValue);
              }
            pixelBuffer3D += bufferIncrement;
          }
      }
      break;
    case(IVIEW_AXIAL):
      // axial (x, y)
      {
        ImagePixelType *pixelBuffer3D = 
          &_imageData->getDataPointer()[pixelBufferSize * _windowCenter[2]];
        for (unsigned int pixelIndex = 0;
             pixelIndex < pixelBufferSize;
             ++pixelIndex)
          {
            currentPixelValue =
              (*pixelBuffer3D++ - _imageIWMin) * iwScaleFactor;
            
            // clamp to [0,255]
            if (currentPixelValue < 0) currentPixelValue = 0;
            else if (currentPixelValue > 255) currentPixelValue = 255;
            
            *pixelBuffer2D++ = 
              static_cast<unsigned char>(currentPixelValue);
          }
      }
      break;
    default:
      {
        throw std::runtime_error("invalid orientation");
      }
    }
  }
  else
  {
    // set image to background
    memset(_imagePixelBuffer, 0, 
           _imagePixelBufferWidth * _imagePixelBufferHeight);
  }
  
  _bufferedImageSliceIndex = _windowCenter[_order[2]];
  _needImageUpdate = false;
}


//
// _updateOverlayPixelBuffer
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_updateOverlayPixelBuffer(void)
{
  if (!_haveValidOverlayData)
  {
    return;
  }
  
  // get overlay slice index
  double sliceIndex = _getOverlaySliceIndex(_windowCenter[_order[2]]);
  
  // we only need to fill in slices that fall into the data region
  if (sliceIndex >= 0 &&
      sliceIndex <= _overlayDim[_order[2]] - 1)
  {
    
    Vector3D<double> continuousIndex;
    continuousIndex[_order[2]] = sliceIndex;
    
    // set the background
    OverlayPixelType OverlayBackground = 0.0;
    
    // fill in the pixel buffer
    float currValue;
    unsigned char *bufferPtr = _overlayPixelBuffer;
    
    // !!! what to do if divide by zero?
    float scaleFactor = 255.0 / (_overlayIWMax - _overlayIWMin);
    unsigned char alpha = 
      static_cast<unsigned char>(_overlayOpacity * 255.0);
    for (int y = 0; y < _overlayDim[_order[1]]; y++)
    {
      continuousIndex[_order[1]] = y;
      for (int x = 0; x < _overlayDim[_order[0]]; x++)
      {
        continuousIndex[_order[0]] = x;
        
        // get value at this index
        currValue = 
          (Array3DUtils::trilerp(*_overlayData,
                                 continuousIndex,
                                 OverlayBackground) 
           - _overlayIWMin) 
          * scaleFactor;
        
        // make sure currValue is in range [0,255]
        if (currValue > 255)
        {
          currValue = 255;
        }
        else if (currValue < 0)
        {
          currValue = 0;
        }
        
        *bufferPtr++ = static_cast<unsigned char>(currValue);
        *bufferPtr++ = alpha;
      }
    }
  }
  else
  {
    // set overlay to background (*2 for alpha)
    memset(_overlayPixelBuffer, 0, 
           _overlayPixelBufferWidth * _overlayPixelBufferHeight * 2);
  }
  
  _bufferedImageSliceIndex = _windowCenter[_order[2]];
  _needOverlayUpdate = false;
}

//
// _computeImageDataMaxMin
//
template <class ImagePixelType, class OverlayPixelType>
void 
IView<ImagePixelType, OverlayPixelType>
::_computeImageDataMaxMin(const ImagePointer& image,
                          double& max,
                          double& min)
{
  Array3DUtils::getMinMax(*image,min,max);
}

//
// _checkImageCompatibility
//
template <class ImagePixelType, class OverlayPixelType>
bool
IView<ImagePixelType, OverlayPixelType>
::_checkImageCompatibility(ImagePointer image, OverlayPointer overlay)
{
  ImageRegionType imageRegion;
  imageRegion.setSize(Vector3D<double>(image->getSize()));
  ImageSizeType imageSize = imageRegion.getSize();
  
  OverlayRegionType overlayRegion;
  overlayRegion.setSize(Vector3D<double>(overlay->getSize()));
  OverlaySizeType   overlaySize   = overlayRegion.getSize();
  
  // see if dimensions match
  if (imageSize != overlaySize)
  {
    return false;
  }
  
  // see if voxel spacing matches
  ImageSizeType imageSpacing = image->getSpacing();
  OverlaySizeType overlaySpacing = overlay->getSpacing();
  if (imageSpacing[0] != overlaySpacing[0] ||
    imageSpacing[1] != overlaySpacing[1] ||
    imageSpacing[2] != overlaySpacing[2])
  {
    return false;
  }
  
  // see if position (origin) matches
  ImageIndexType imageOrigin = image->getOrigin();
  OverlayIndexType overlayOrigin = overlay->getOrigin();
  if (imageOrigin[0] != overlayOrigin[0] ||
    imageOrigin[1] != overlayOrigin[1] ||
    imageOrigin[2] != overlayOrigin[2])
  {
    return false;
  }
  
  return true;
}

//
// _setImageSize : sets size for image pixel buffer
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::_setImageSize(int xDim, 
                int yDim, 
                int zDim,
                const double& xSpacing, 
                const double& ySpacing, 
                const double& zSpacing,
                const double& xOffset,
                const double& yOffset,
                const double& zOffset)
{
  _imageDim[0] = xDim;
  _imageDim[1] = yDim;
  _imageDim[2] = zDim;
  
  _imageSpacing[0] = xSpacing;
  _imageSpacing[1] = ySpacing;
  _imageSpacing[2] = zSpacing;
  
  _imageOffset[0]  = xOffset;
  _imageOffset[1]  = yOffset;
  _imageOffset[2]  = zOffset;
  
  // set pixel buffer dimensions
  _imagePixelBufferWidth = _imageDim[_order[0]];
  _imagePixelBufferHeight = _imageDim[_order[1]];
  
  // set max image dimensions (for zoom purposes)
  _maxImageDim = _imageDim[0];
  if (_maxImageDim < _imageDim[1])
  {
    _maxImageDim = _imageDim[1];
  }
  if (_maxImageDim < _imageDim[2])
  {
    _maxImageDim = _imageDim[2];
  }
  
  // get rid of old image display buffer
  if (_imagePixelBuffer != 0)
  {
    if (_verbose)
    {
      std::cerr << "####### deleting imagepixel buffer ########" << std::endl;
    }
    delete [] _imagePixelBuffer;
  }
  
  _imagePixelBuffer = new unsigned char[_imagePixelBufferWidth
    * _imagePixelBufferHeight];
  
  if (_verbose)
  {
    std::cerr << "IView: set image size..." << std::endl;
    std::cerr << "dim = [" 
      << _imageDim[0] << ", " 
      << _imageDim[1] << ", " 
      << _imageDim[2] << "]" << std::endl;
    std::cerr << "spc = [" 
      << _imageSpacing[0] << ", " 
      << _imageSpacing[1] << ", " 
      << _imageSpacing[2] << "]" << std::endl;
    std::cerr << "off = [" 
      << _imageOffset[0] << ", " 
      << _imageOffset[1] << ", " 
      << _imageOffset[2] << "]" << std::endl;
    std::cerr << "pixel buffer: " 
      << _imagePixelBufferWidth << " x " 
      << _imagePixelBufferHeight << std::endl;
  }
}

//
// _setOverlaySize : sets size for overlay pixel buffer
//
template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::_setOverlaySize(int xDim, 
                  int yDim, 
                  int zDim,
                  const double& xSpacing, 
                  const double& ySpacing, 
                  const double& zSpacing,
                  const double& xOffset,
                  const double& yOffset,
                  const double& zOffset)
{
  _overlayDim[0] = xDim;
  _overlayDim[1] = yDim;
  _overlayDim[2] = zDim;
  
  _overlaySpacing[0] = xSpacing;
  _overlaySpacing[1] = ySpacing;
  _overlaySpacing[2] = zSpacing;
  
  _overlayOffset[0]  = xOffset;
  _overlayOffset[1]  = yOffset;
  _overlayOffset[2]  = zOffset;
  
  // set pixel buffer dimensions
  _overlayPixelBufferWidth = _overlayDim[_order[0]];
  _overlayPixelBufferHeight = _overlayDim[_order[1]];
  
  // get rid of old overlay display buffer
  if (_overlayPixelBuffer != 0)
  {
    delete [] _overlayPixelBuffer;
  }
  
  _overlayPixelBuffer = new unsigned char[_overlayPixelBufferWidth
    * _overlayPixelBufferHeight * 2];
  
  if (_verbose)
  {
    std::cerr << "IView: set overlay size..." << std::endl;
    std::cerr << "dim = [" 
      << _overlayDim[0] << ", " 
      << _overlayDim[1] << ", " 
      << _overlayDim[2] << "]" << std::endl;
    std::cerr << "spc = [" 
      << _overlaySpacing[0] << ", " 
      << _overlaySpacing[1] << ", " 
      << _overlaySpacing[2] << "]" << std::endl;
    std::cerr << "off = [" 
      << _overlayOffset[0] << ", " 
      << _overlayOffset[1] << ", " 
      << _overlayOffset[2] << "]" << std::endl;
    std::cerr << "pixel buffer: " 
      << _overlayPixelBufferWidth << " x " 
      << _overlayPixelBufferHeight << std::endl;
  }
}

template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::_setNewImageCenter()
{
  // center if current center is out of range
  // or 0
  if (_windowCenter[0] >= _imageDim[0] ||
    _windowCenter[1] >= _imageDim[1] ||
    _windowCenter[2] >= _imageDim[2] ||
      (_windowCenter[0] == 0 &&
       _windowCenter[1] == 0 &&
       _windowCenter[2] == 0))
  {
    _windowCenter[0] = _imageDim[0] / 2;
    _windowCenter[1] = _imageDim[1] / 2;
    _windowCenter[2] = _imageDim[2] / 2;
    
    _windowCenterf[0] = _imageDim[0] / 2;
    _windowCenterf[1] = _imageDim[1] / 2;
    _windowCenterf[2] = _imageDim[2] / 2;
  }
}

template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::_imagePixelZoomX() const
{
  double spacingZoomX = _imageSpacing[_order[0]] / fabs(_imageSpacing[0]);
  // double spacingZoomX = _imageSpacing[_order[0]] / imageSpacing[0];
  double fitToWindowZoom = static_cast<double>(_windowWidth)
    / static_cast<double>(_maxImageDim);
  return _axisSign[_order[0]] * fitToWindowZoom * spacingZoomX * _windowZoom;
}

template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::_imagePixelZoomY() const
{
  double spacingZoomY = _imageSpacing[_order[1]] / fabs(_imageSpacing[0]);
  // double spacingZoomY = _imageSpacing[_order[1]] / imageSpacing[0];
  double fitToWindowZoom = static_cast<double>(_windowWidth)
    / static_cast<double>(_maxImageDim);
  return _axisSign[_order[1]] * fitToWindowZoom * spacingZoomY * _windowZoom;
  // return -fabs(fitToWindowZoom * spacingZoomY * _windowZoom);
}

template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::_overlayPixelZoomX() const
{
  double spacingZoomX = _overlaySpacing[_order[0]] / fabs(_imageSpacing[0]);
  // double spacingZoomX = _overlaySpacing[_order[0]] / _imageSpacing[0];
  double fitToWindowZoom = static_cast<double>(_windowWidth)
    / static_cast<double>(_maxImageDim);
  //double imageToOverlayZoom = _overlaySpacing[_order[0]]
  //  / _imageSpacing[_order[0]];
  return _axisSign[_order[0]] * fitToWindowZoom * spacingZoomX * _windowZoom;
  // return fabs(fitToWindowZoom * spacingZoomX * _windowZoom);
}

template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::_overlayPixelZoomY() const
{
  double spacingZoomY = _overlaySpacing[_order[1]] / fabs(_imageSpacing[0]);
  // double spacingZoomY = _overlaySpacing[_order[1]] / _imageSpacing[0];
  double fitToWindowZoom = static_cast<double>(_windowWidth)
    / static_cast<double>(_maxImageDim);
  //  double imageToOverlayZoom = _overlaySpacing[_order[1]] 
  //  / _IMAGESPACING[_order[1]];
  return _axisSign[_order[1]] * fitToWindowZoom * spacingZoomY * _windowZoom;
  // return -fabs(fitToWindowZoom * spacingZoomY * _windowZoom);
}

template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::_windowToImageIndexCoordinates(float windowX,
                                 float windowY,
                                 float& imageX,
                                 float& imageY,
                                 float& imageZ) const
{
  float windowMidpointX = static_cast<float>(_windowWidth) / 2.0;
  float windowMidpointY = static_cast<float>(_windowHeight) / 2.0;
  windowY = static_cast<float>(_windowHeight) - windowY;
  
  switch (_orientation)
  {
  case IVIEW_SAGITTAL://sagittal (y,z)
    imageX = _windowCenter[0];
    imageY = ((windowX - windowMidpointX) / _imagePixelZoomX())
      + _windowCenterf[1];
    imageZ = ((windowY - windowMidpointY) / _imagePixelZoomY())
      + _windowCenterf[2];
    break;
  case IVIEW_CORONAL://coronal (x,z)
    imageX = ((windowX - windowMidpointX) / _imagePixelZoomX())
      + _windowCenterf[0];
    imageY = _windowCenter[1];
    imageZ = ((windowY - windowMidpointY) / _imagePixelZoomY())
      + _windowCenterf[2];
    break;
  case IVIEW_AXIAL://axial (x,y)
    imageX = ((windowX - windowMidpointX) / _imagePixelZoomX())
      + _windowCenterf[0];
    imageY = ((windowY - windowMidpointY) / _imagePixelZoomY())
      + _windowCenterf[1];
    imageZ = _windowCenter[2];
    break;
  }
}

template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::_imageIndexToWindowCoordinates(float imageX,
                                 float imageY,
                                 float imageZ,
                                 float& windowX,
                                 float& windowY) const
{
  float windowMidpointX = static_cast<float>(_windowWidth)  / 2.0;
  float windowMidpointY = static_cast<float>(_windowHeight) / 2.0;
  
  switch (_orientation)
  {
  case IVIEW_SAGITTAL://sagittal (y,z)
    windowX = (_imagePixelZoomX() * (imageY - _windowCenterf[1])
      + windowMidpointX);
    windowY = (_imagePixelZoomY() * (imageZ - _windowCenterf[2])
      + windowMidpointY);
    break;
  case IVIEW_CORONAL://coronal (x,z)
    windowX = (_imagePixelZoomX() * (imageX - _windowCenterf[0])
      + windowMidpointX);
    windowY = (_imagePixelZoomY() * (imageZ - _windowCenterf[2])
      + windowMidpointY);
    break;
  case IVIEW_AXIAL://axial (x,y)
    windowX = (_imagePixelZoomX() * (imageX - _windowCenterf[0])
      + windowMidpointX);
    windowY = (_imagePixelZoomY() * (imageY - _windowCenterf[1])
      + windowMidpointY);
    break;
  }
}

template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::_setupGLDrawPixelsImage()
{
  // compute zoom
  double pixelZoomX = _imagePixelZoomX();
  double pixelZoomY = _imagePixelZoomY();
  
  // compute translation
  double windowMidpointX = static_cast<double>(_windowWidth)  / 2.0;    
  double windowMidpointY = static_cast<double>(_windowHeight) / 2.0;
  double dataMidpointX   = static_cast<double>(_windowCenterf[_order[0]])
    * pixelZoomX;
  double dataMidpointY   = static_cast<double>(_windowCenterf[_order[1]])
    * pixelZoomY;   
  
  // apply zoom and translation
  float startX = static_cast<float>(windowMidpointX - dataMidpointX);
  float startY = static_cast<float>(windowMidpointY - dataMidpointY);
  glRasterPos2f(startX, startY );
  glPixelZoom(pixelZoomX, pixelZoomY);
}

template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::_setupGLDrawPixelsOverlay()
{
  // compute zoom
  double imagePixelZoomX = _imagePixelZoomX();
  double imagePixelZoomY = _imagePixelZoomY();
  double overlayPixelZoomX = _overlayPixelZoomX();
  double overlayPixelZoomY = _overlayPixelZoomY();
  
  double worldToWindowScaleFactorX = imagePixelZoomX / _imageSpacing[_order[0]];
  double worldToWindowScaleFactorY = imagePixelZoomY / _imageSpacing[_order[1]];
  // compute translation
  double windowMidpointX = static_cast<double>(_windowWidth)  / 2.0;    
  double windowMidpointY = static_cast<double>(_windowHeight) / 2.0;
  double dataMidpointX   = 
    static_cast<double>(_windowCenterf[_order[0]]) * imagePixelZoomX
    + worldToWindowScaleFactorX
    * (_imageOffset[_order[0]] - _overlayOffset[_order[0]]);
  double dataMidpointY   = 
    static_cast<double>(_windowCenterf[_order[1]]) * imagePixelZoomY
    + worldToWindowScaleFactorY
    * (_imageOffset[_order[1]] - _overlayOffset[_order[1]]);
  
  // apply zoom and translation
  float startX = static_cast<float>(windowMidpointX - dataMidpointX);
  float startY = static_cast<float>(windowMidpointY - dataMidpointY);
  
  glRasterPos2f(startX, startY);
  glPixelZoom(overlayPixelZoomX, overlayPixelZoomY);
}

template <class ImagePixelType, class OverlayPixelType>
double
IView<ImagePixelType, OverlayPixelType>
::_getOverlaySliceIndex(const int& imageSliceIndex)
{
  return (imageSliceIndex * _imageSpacing[_order[2]] 
    + _imageOffset[_order[2]] - _overlayOffset[_order[2]]) 
    / _overlaySpacing[_order[2]];
}

template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::_checkForGLError(const char *message)
{
  GLenum errorCode = glGetError();
  if (_verbose && (errorCode != GL_NO_ERROR))
  {
    // !!!bcd!!!
    //    const GLubyte *errString = gluErrorString(errorCode);
    if (message != 0)
    {
      std::cerr << message;
      //  std::cerr << "OpenGL Error: " << errString << std::endl;
    }
  }
}

template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::_updateROIDraggingFlags(bool isPress, int mouseX, int mouseY)
{
    if (!isPress)
    {
        // the roi button was released
        // all dragging flags should be set to false
        _roiDraggingXMin = false;
        _roiDraggingXMax = false;
        _roiDraggingYMin = false;
        _roiDraggingYMax = false;
        _roiDraggingZMin = false;
        _roiDraggingZMax = false;
        return;
    }

    // Since we got here, the roi button was pressed
    mouseY = _windowHeight - mouseY;
    
    int clickTolerance = 6;
    
    // get window coordinates of roi
    float roiXMin, roiXMax, roiYMin, roiYMax;
    
    _imageIndexToWindowCoordinates(_roi.start[0], 
                                   _roi.start[1], 
                                   _roi.start[2],
                                   roiXMin,
                                   roiYMin);
    _imageIndexToWindowCoordinates(_roi.stop[0], 
                                   _roi.stop[1], 
                                   _roi.stop[2],
                                   roiXMax,
                                   roiYMax);

    std::cout << 
      _roi.start[0] << " " << 
      _roi.start[1] << " " << 
      _roi.start[2] << " " <<
      roiXMin << " " <<
      roiYMin << std::endl;

    std::cout << 
      _roi.stop[0] << " " << 
      _roi.stop[1] << " " << 
      _roi.stop[2] << " " <<
      roiXMax << " " <<
      roiYMax << std::endl;

    // Determine whether XMin in the window corresponds to a max in
    // the image
    bool flipX = false;
    if (roiXMin > roiXMax)
    {
        std::swap(roiXMin, roiXMax);
        flipX = true;
    }

    // Determine whether YMin in the window corresponds to a max in
    // the image
    bool flipY = false;
    if (roiYMin > roiYMax)
    {
        std::swap(roiYMin, roiYMax);
        flipY = true;
    }

    bool attachedToLeft = false;
    bool attachedToRight = false;
    bool attachedToBottom = false;
    bool attachedToTop = false;

    int innerTolerance = clickTolerance;
    if (roiXMax - roiXMin <= 2 * clickTolerance)
    {
        innerTolerance = (int) ((roiXMax - roiXMin) * .5);
    }
    if (mouseX > (roiXMin - clickTolerance) &&
        mouseX < (roiXMin + innerTolerance) &&
        mouseY > (roiYMin - clickTolerance) &&
        mouseY < (roiYMax + clickTolerance))
    {
        attachedToLeft = true;
    } 
    else if (mouseX > (roiXMax - innerTolerance) &&
             mouseX < (roiXMax + clickTolerance) &&
             mouseY > (roiYMin - clickTolerance) &&
             mouseY < (roiYMax + clickTolerance))
    {
        attachedToRight = true;
    }

    if (flipX)
    {
        std::swap(attachedToLeft, attachedToRight);
    }

    innerTolerance = clickTolerance;
    if (roiYMax - roiYMin <= 2 * clickTolerance)
    {
        innerTolerance = (int) ((roiYMax - roiYMin) * .5);
    }
    if (mouseX > (roiXMin - clickTolerance) &&
        mouseX < (roiXMax + clickTolerance) &&
        mouseY > (roiYMin - clickTolerance) &&
        mouseY < (roiYMin + innerTolerance))
    {
        attachedToBottom = true;
    }
    else if (mouseX > (roiXMin - clickTolerance) &&
             mouseX < (roiXMax + clickTolerance) &&
             mouseY > (roiYMax - innerTolerance) &&
             mouseY < (roiYMax + clickTolerance))
    {
        attachedToTop = true;
    }

    if (flipY)
    {
        std::swap(attachedToBottom, attachedToTop);
    }

    if (_orientation == IVIEW_AXIAL)
    {
        _roiDraggingXMin = attachedToLeft;
        _roiDraggingXMax = attachedToRight;
        _roiDraggingYMin = attachedToBottom;
        _roiDraggingYMax = attachedToTop;
    }

    if (_orientation == IVIEW_CORONAL)
    {
        _roiDraggingXMin = attachedToLeft;
        _roiDraggingXMax = attachedToRight;
        _roiDraggingZMin = attachedToBottom;
        _roiDraggingZMax = attachedToTop;
    }

    if (_orientation == IVIEW_SAGITTAL)
    {
        _roiDraggingYMin = attachedToLeft;
        _roiDraggingYMax = attachedToRight;
        _roiDraggingZMin = attachedToBottom;
        _roiDraggingZMax = attachedToTop;
    }

}


template <class ImagePixelType, class OverlayPixelType>
void
IView<ImagePixelType, OverlayPixelType>
::_dragROI(int newMouseX, int newMouseY)
{

  if (!(_roiDraggingXMin || _roiDraggingXMax ||
        _roiDraggingYMin || _roiDraggingYMax ||
        _roiDraggingZMin || _roiDraggingZMax))
  {
    return;
  }

  float roiX, roiY, roiZ;
  _windowToImageIndexCoordinates(newMouseX, newMouseY, 
                                 roiX, roiY, roiZ);

  if (roiX < 0) roiX = 0;
  if (roiY < 0) roiY = 0;
  if (roiZ < 0) roiZ = 0;
  if (roiX > _imageDim[0]-1) roiX = _imageDim[0]-1;
  if (roiY > _imageDim[1]-1) roiY = _imageDim[1]-1;
  if (roiZ > _imageDim[2]-1) roiZ = _imageDim[2]-1;
    
  if (_roiDraggingXMin) _roi.start[0] = static_cast<int>(roiX);
  if (_roiDraggingXMax) _roi.stop[0] = static_cast<int>(roiX);
  if (_roiDraggingYMin) _roi.start[1] = static_cast<int>(roiY);
  if (_roiDraggingYMax) _roi.stop[1] = static_cast<int>(roiY);
  if (_roiDraggingZMin) _roi.start[2] = static_cast<int>(roiZ);
  if (_roiDraggingZMax) _roi.stop[2] = static_cast<int>(roiZ);

  redraw();	
  if (_roiChangedCallback != 0)
  {
    _roiChangedCallback(_orientation, _roiChangedCallbackArg);
  }

}

#endif
