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

#include "ImMap.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>     

#include <AnastructUtils.h>
#include <Array3DUtils.h>
#include <BasicException.h>
#include <BasicFileParser.h>
#include <BinaryIO.h>
#include <Debugging.h>
#include <DownsampleFilter3D.h>
#include <EstimateAffine.h>
#include <HField3DIO.h>
#include <HField3DUtils.h>
#include <MultiscaleFluid.h>
#include <ROIUtils.h>
#include <ApplicationUtils.h>
#include <StringUtils.h>
#include <Surface.h>
#include <SurfaceUtils.h>

#include <FL/Fl_Color_Chooser.H>
#include <FL/Fl_File_Chooser.H>
#include <Fl_Table.H>
#include <Fl_Table_Row.H>
#include <FL/fl_ask.H>

#ifdef WIN32
#pragma warning (disable: 4800) // Warning about forcing int to bool.
#pragma warning (disable: 4786) // truncated debug info
#endif

#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#endif

#include <time.h>

int HISTOGRAM_HEIGHT = 450;

/////////////////
// constructor //
/////////////////

ImMap
::ImMap() 
: ImMapGUI()
{
 _totalImagesLoaded = 0;
  // add "None" choices to image choice menues
  imageChoice->add("None");
  imageChoice->value(0);
  overlayChoice->add("None");
  overlayChoice->value(0);
  imageSaveChoice->add("None");
  imageSaveChoice->value(0);
  _loadedImages.push_back(0);

  //preset organ intensity
  Intensity original;
  // original Intensity reset when an image is loaded
  original.relativeMin = 0.0;
  original.relativeMax = 0.0;
  imagePresetIntensityChoice->add("Original");
  overlayPresetIntensityChoice->add("Original");
  _presetIntensity.push_back(original);

  lockedIntensityCallback();

  //mask
  maskChoice->add("none");
  maskChoice->value(0);
  _mask.push_back(MaskType());

  // To have the same index as _loadedImages
  HistogramDat tempHisto;
  _histograms.push_back(tempHisto); 

  // The same thing for _imageNames
  _imageNames.push_back("NONE");
  _imageFullFileNames.push_back("NONE");


  //idem for ImAnaVectors
  _imAnaVectors.push_back(std::vector<ImAna>());

  BYUImageList->value(0);

  // turn on view image
  axial2DWindow->setViewImage(true);
  coronal2DWindow->setViewImage(true);
  sagittal2DWindow->setViewImage(true);

  // turn on view overlay
  axial2DWindow->setViewOverlay(true);
  coronal2DWindow->setViewOverlay(true);
  sagittal2DWindow->setViewOverlay(true);

  // turn on view crosshairs
  axial2DWindow->setViewCrosshairs(true);
  coronal2DWindow->setViewCrosshairs(true);
  sagittal2DWindow->setViewCrosshairs(true);

  // turn on view ROI
  axial2DWindow->setViewROI(false);
  coronal2DWindow->setViewROI(false);
  sagittal2DWindow->setViewROI(false);

   // turn on view Mask
  axial2DWindow->setViewMask(false);
  coronal2DWindow->setViewMask(false);
  sagittal2DWindow->setViewMask(false);

  // set up orthogonal viewing directions
  axial2DWindow->setOrientation(IVIEW_AXIAL);
  coronal2DWindow->setOrientation(IVIEW_CORONAL);
  sagittal2DWindow->setOrientation(IVIEW_SAGITTAL);

  // Adjust scrollbars
  axialScrollBar->slider_size(.08);
  coronalScrollBar->slider_size(.08);
  sagittalScrollBar->slider_size(.08);

  // set the color preset
  ColorPreset greyScale;
  greyScale.image.red = greyScale.image.green = greyScale.image.blue = 1.0;
  greyScale.overlay.red = greyScale.overlay.green = greyScale.overlay.blue = 1.0;
  colorPresetChoice->add("Greyscale");
  _colorPresetIntensity.push_back(greyScale);

  //set the default fluid warp parameters 
  fluidParameters defaultParameters;
  //fine
  defaultParameters.params[0].numIterations = 1;
  defaultParameters.params[0].alpha = 0.01;
  defaultParameters.params[0].beta = 0.01;
  defaultParameters.params[0].gamma = 0.001;
  defaultParameters.params[0].maxPerturbation=0.5;
  defaultParameters.params[0].numBasis=2000;

  //medium
  defaultParameters.params[1].numIterations = 25;
  defaultParameters.params[1].alpha = 0.01;
  defaultParameters.params[1].beta = 0.01;
  defaultParameters.params[1].gamma = 0.001;
  defaultParameters.params[1].maxPerturbation=0.5;
  defaultParameters.params[1].numBasis=2000;

  //coarse
  defaultParameters.params[2].numIterations = 500;
  defaultParameters.params[2].alpha = 0.1;
  defaultParameters.params[2].beta = 0.1;
  defaultParameters.params[2].gamma = 0.01;
  defaultParameters.params[2].maxPerturbation=0.5;
  defaultParameters.params[2].numBasis=2000;

  fluidParamChoice->add("default");
  _fluidParamPreset.push_back(defaultParameters);
  fluidParamChoice->value(0);
  fluidParametersCallback();

  //set BYU property widget
  colorSelectedObjectMenu->add("Red");
  colorSelectedObjectMenu->add("Lime");
  colorSelectedObjectMenu->add("Blue");
  colorSelectedObjectMenu->add("Yellow");
  colorSelectedObjectMenu->add("Yellow75");  
  colorSelectedObjectMenu->add("Magenta");
  colorSelectedObjectMenu->add("Magenta75");
  colorSelectedObjectMenu->add("Cyan");
  colorSelectedObjectMenu->add("Cyan75");
  colorSelectedObjectMenu->add("White");
  colorSelectedObjectMenu->add("Black");
  colorSelectedObjectMenu->add("Silver");
  colorSelectedObjectMenu->add("Gray");
  colorSelectedObjectMenu->add("Maroon");
  colorSelectedObjectMenu->add("Green");
  colorSelectedObjectMenu->add("Navy");
  colorSelectedObjectMenu->add("Purple");
  colorSelectedObjectMenu->add("Olive");
  colorSelectedObjectMenu->add("Teal");
  colorSelectedObjectMenu->add("Color Panel");

  aspectSelectedObjectChooser->add("Surface");
  aspectSelectedObjectChooser->add("Wireframe");
  aspectSelectedObjectChooser->add("Contour");


  // set RGB
  axial2DWindow->setImageColorRGB(_colorPresetIntensity[0].image.red,
                                  _colorPresetIntensity[0].image.green,
                                  _colorPresetIntensity[0].image.blue);
  axial2DWindow->setOverlayColorRGB(_colorPresetIntensity[0].overlay.red,
                                    _colorPresetIntensity[0].overlay.green,
                                    _colorPresetIntensity[0].overlay.blue);
  coronal2DWindow->setImageColorRGB(_colorPresetIntensity[0].image.red,
                                    _colorPresetIntensity[0].image.green,
                                    _colorPresetIntensity[0].image.blue);
  coronal2DWindow->setOverlayColorRGB(_colorPresetIntensity[0].overlay.red,
                                      _colorPresetIntensity[0].overlay.green,
                                      _colorPresetIntensity[0].overlay.blue);
  sagittal2DWindow->setImageColorRGB(_colorPresetIntensity[0].image.red,
                                     _colorPresetIntensity[0].image.green,
                                     _colorPresetIntensity[0].image.blue);
  sagittal2DWindow->setOverlayColorRGB(_colorPresetIntensity[0].overlay.red,
                                       _colorPresetIntensity[0].overlay.green,
                                       _colorPresetIntensity[0].overlay.blue);

  // set opacity
  axial2DWindow->setOverlayOpacity(overlayAlphaSlider->value());
  coronal2DWindow->setOverlayOpacity(overlayAlphaSlider->value());
  sagittal2DWindow->setOverlayOpacity(overlayAlphaSlider->value());

  // set 3D window background
  // surface3DWindow->setBackgroundColor(1, 1, 1);

  // set zoom
  _zoomValue = 1.0;
  axial2DWindow->setWindowZoom(_zoomValue);
  coronal2DWindow->setWindowZoom(_zoomValue);
  sagittal2DWindow->setWindowZoom(_zoomValue);

  // set click callback
  axial2DWindow->setClickCallback(&(ImMap::clickCallback), (void*)this);
  coronal2DWindow->setClickCallback(&(ImMap::clickCallback), (void*)this);
  sagittal2DWindow->setClickCallback(&(ImMap::clickCallback), (void*)this);

  // set roiChanged callback
  axial2DWindow->setROIChangedCallback(&(ImMap::roiChangedCallback), 
                                       (void*)this);
  coronal2DWindow->setROIChangedCallback(&(ImMap::roiChangedCallback), 
                                         (void*)this);
  sagittal2DWindow->setROIChangedCallback(&(ImMap::roiChangedCallback), 
                                          (void*)this);

  // set roiChanged callback

  HistogramDisplay->setMinMaxChangedCallback(&(ImMap::MinMaxChangedCallback),(void*)this);
  HistogramOverlayDisplay->setMinMaxChangedCallback(&(ImMap::MinMaxChangedCallback),(void*)this);

  // set BYUTable Callback
  BYUTableDisplay->callback(ImMap::BYUTableChangedCallback, (void*)this);

  HistogramDisplay->setLUTcolor(_colorPresetIntensity[0].image.red,
    _colorPresetIntensity[0].image.green,
    _colorPresetIntensity[0].image.blue);
  HistogramOverlayDisplay->setLUTcolor(_colorPresetIntensity[0].overlay.red,
    _colorPresetIntensity[0].overlay.green,
    _colorPresetIntensity[0].overlay.blue);
  // set ROI
  axial2DWindow->setROI(static_cast<int>(roiStartX->value()), 
    static_cast<int>(roiStartY->value()), 
    static_cast<int>(roiStartZ->value()),
    static_cast<int>(roiStopX->value()), 
    static_cast<int>(roiStopY->value()), 
    static_cast<int>(roiStopZ->value()));
  axial2DWindow->setROIOpacity(roiOpacitySlider->value());

  coronal2DWindow->setROI(static_cast<int>(roiStartX->value()), 
    static_cast<int>(roiStartY->value()), 
    static_cast<int>(roiStartZ->value()),
    static_cast<int>(roiStopX->value()), 
    static_cast<int>(roiStopY->value()), 
    static_cast<int>(roiStopZ->value()));
  coronal2DWindow->setROIOpacity(roiOpacitySlider->value());

  sagittal2DWindow->setROI(static_cast<int>(roiStartX->value()), 
    static_cast<int>(roiStartY->value()), 
    static_cast<int>(roiStartZ->value()),
    static_cast<int>(roiStopX->value()), 
    static_cast<int>(roiStopY->value()), 
    static_cast<int>(roiStopZ->value()));
  sagittal2DWindow->setROIOpacity(roiOpacitySlider->value());  

  // don't use entire image for registration by default
  useROIButton->value(1);
  axial2DWindow->setViewROI(true);
  coronal2DWindow->setViewROI(true);
  sagittal2DWindow->setViewROI(true);
  _ROIcreated = false;

  // By default, the histogram in the main Window is the Image's one
  _histogramImage = true;

  // If true, both histograms change together
  _histogramLinked = true;

  // affine transformation type
  affineChoice->add("Translate");
  affineChoice->add("Rigid");
  affineChoice->add("Affine");
  affineChoice->value(0);

  textbuf = new Fl_Text_Buffer;
  matrixDisplay->buffer(textbuf);
  matrixDisplay->textsize(10);

  //deformation type
  _deformationChoices.push_back("Fluid");
  _deformationChoices.push_back("Elastic");
  _deformationChoices.push_back("Elastic With Mask");
  _deformationChoices.push_back("Shrink");
  _deformationChoices.push_back("Elastic Shrink With Mask");
  _deformationChoices.push_back("Deflate Reverse");
  _deformationChoices.push_back("Deflate Forward");
  for (unsigned int i = 0; i < _deformationChoices.size(); ++i) {
    deformationChoice->add( _deformationChoices[i].c_str() );
  }
  deformationChoice->value(0);
  fluidTrilinearButton->type(FL_RADIO_BUTTON);
  fluidNearestNeighborButton->type(FL_RADIO_BUTTON);
  fluidTrilinearButton->value(1);

  for (unsigned int row = 0 ; row < 100 ; row++)
  { _lastRowsSelected[row]=0 ; }


  // To Load if Extension unknown

  _header_saved=false;

  ElementType_value->add("MET_CHAR");
  ElementType_value->add("MET_UCHAR");
  ElementType_value->add("MET_SHORT");
  ElementType_value->add("MET_USHORT");
  ElementType_value->add("MET_INT");
  ElementType_value->add("MET_UINT");
  ElementType_value->add("MET_DOUBLE");
  ElementType_value->add("MET_FLOAT");

  ElementType_value->value(3);

}

////////////////
// destructor //
////////////////

ImMap
::~ImMap()
{
  std::cout << "[ImMap::~ImMap]" << std::endl;
}

/////////////////
// loadImage() //
/////////////////

void ImMap::loadImage()
{
  char* fileName = fl_file_chooser("Choose image file...","","", 0);
  if( fileName == (char*)0 ) 
  {
    std::cout << "[ImMap::loadImage] file chooser failed to get file";
    std::cout << std::endl;
    return;
  }
  try
  {
    loadImage(fileName);
  }
  catch(std::exception e)
  {
    fl_alert(e.what());
    _updateStatusBuffer(e.what());
    return;
  }
}

//////////////////////
// loadImage(char*) //
//////////////////////

void
ImMap
::loadImage(char *fileName)
{ 
  std::string fileNameString(fileName);
  // Load image data
  std::ostringstream oss;
  oss << "Loading " << fileName;
  _updateStatusBuffer(oss.str().c_str());

  ImagePointer imagePtr = 0;


  // differentiate between dicom set and a regular image
  if (fileNameString.find('*') != std::string::npos)
  {
    Timer timerDicom;
    timerDicom.start();
    std::cerr << "found a '*' in the filename, trying to load DICOM stack..."
              << std::endl;
    imagePtr = _loadImageDICOM(fileName);
    timerDicom.stop();
    std::cerr << "LoadImageDicom (msec) " << timerDicom.getMilliseconds()
              << std::endl;
  }
  else
  {

    std::cout <<"Loading non-DICOM image." << std::endl;

    Timer timerImage;
    timerImage.start();
    imagePtr = _loadOtherImage(fileName);
    timerImage.stop();
    std::cout << "Image took "
              << timerImage.getMilliseconds()
              << " msec to load." << std::endl; 
  }

  ImageUtils::makeImageUnsigned(*imagePtr);

  // update gui
  if( imagePtr == 0 ) 
  {
    std::ostringstream oss2;
    oss2 << "Error: could not load " << fileName;
    _updateStatusBuffer(oss2.str().c_str());
    //_updateStatusBuffer("Error loading image file.");
  }
  else 
  {
    // update status bar
    _updateStatusBuffer("Image loaded");
    // add image to list


    _addImageToList(imagePtr);
    std::cout<<"_totalImageLoaded " << _totalImagesLoaded<<std::endl;

    if (imageChoice->value() == 0) {
      imageChoice->value(_loadedImages.size() - 1);
      imageChoice->do_callback();
    } else if (overlayChoice->value() == 0) {
      overlayChoice->value(_loadedImages.size() - 1);
      overlayChoice->do_callback();
    }
  }
}

void
ImMap
::screenCaptureAxialCallback()
{
  std::string fileNameSuggestion = "screenShot_Axial.ppm";
  char* fileName = fl_file_chooser("Choose image name (file will be ppm)",
                                   "", fileNameSuggestion.c_str(), 0);
  if (fileName == 0) return;
  _screenCaptureAxial(fileName);
}

void
ImMap
::screenCaptureCoronalCallback()
{
  std::string fileNameSuggestion = "screenShot_Coronal.ppm";
  char* fileName = fl_file_chooser("Choose image name (file will be ppm)"
                                   "", fileNameSuggestion.c_str(), 0);
  if (fileName == 0) return;
  _screenCaptureCoronal(fileName);
}

void
ImMap
::screenCaptureSagittalCallback()
{
  std::string fileNameSuggestion = "screenShot_Sagittal.ppm";
  char* fileName = fl_file_chooser("Choose image name (file will be ppm)",
                                   "", fileNameSuggestion.c_str(), 0);
  if (fileName == 0) return;
  _screenCaptureSagittal(fileName);
}

void
ImMap
::screenCapture3DCallback()
{
  std::string fileNameSuggestion = "screenShot_3D.ppm";
  char* fileName = fl_file_chooser("Choose image name (file will be png)",
                                   "", fileNameSuggestion.c_str(), 0);
  if (fileName == 0) return;
  _screenCapture3D(fileName);
}

void
ImMap
::screenCaptureAllCallback()
{
  std::string fileNameSuggestion = "screenShot_";
  char* fileName = fl_file_chooser("Choose image prefix...",
                                   "", fileNameSuggestion.c_str(), 0);
  if (fileName == 0) return;
  _screenCaptureAll(fileName);
}

void
ImMap
::_screenCaptureAxial(const std::string& filename)
{
  axial2DWindow->saveWindowAsImage(filename.c_str());
}

void
ImMap
::_screenCaptureCoronal(const std::string& filename)
{
  coronal2DWindow->saveWindowAsImage(filename.c_str());
}

void
ImMap
::_screenCaptureSagittal(const std::string& filename)
{
  sagittal2DWindow->saveWindowAsImage(filename.c_str());
}

void
ImMap
::_screenCapture3D(const std::string& filename)
{
  surface3DWindow->saveWindowAsImage(filename.c_str());
}

void
ImMap
::_screenCaptureAll(const std::string& fileName)
{
  std::string axialFilename = std::string(fileName) + "Axial.ppm";
  std::string coronalFilename = std::string(fileName) + "Coronal.ppm";
  std::string sagittalFilename = std::string(fileName) + "Sagittal.ppm";


  axial2DWindow->saveWindowAsImage(axialFilename.c_str());
  coronal2DWindow->saveWindowAsImage(coronalFilename.c_str());
  sagittal2DWindow->saveWindowAsImage(sagittalFilename.c_str());
}

void
ImMap
::_parseVector3DDouble(const std::string& s, Vector3D<double>& v)
{
  std::list<std::string> tokens;
  StringUtils::tokenize(s, " ", tokens);
  if (tokens.size() == 3)
    {
      v.x = StringUtils::toDouble(tokens.front());
      tokens.pop_front();
      v.y = StringUtils::toDouble(tokens.front());
      tokens.pop_front();
      v.z = StringUtils::toDouble(tokens.front());
    }
}

void
ImMap
::_parseVector3DInt(const std::string& s, Vector3D<int>& v)
{
  std::list<std::string> tokens;
  StringUtils::tokenize(s, " ", tokens);
  if (tokens.size() == 3)
    {
      v.x = StringUtils::toInt(tokens.front());
      tokens.pop_front();
      v.y = StringUtils::toInt(tokens.front());
      tokens.pop_front();
      v.z = StringUtils::toInt(tokens.front());
    }
}

void
ImMap
::_parseIntensityWindows(const std::string& s, double& min, double& max)
{
  std::list<std::string> tokens;
  StringUtils::tokenize(s, " ", tokens);
  if (tokens.size() == 2)
    {
      min = StringUtils::toDouble(tokens.front());
      tokens.pop_front();
      max = StringUtils::toDouble(tokens.front());
    }
}

/////////////////
// applyScript //
/////////////////
bool 
ImMap
::applyScript(const std::string& key, const std::string& value)
{
  static unsigned int nbAnastructsLoaded;
  static std::string screenCaptureFilename;

  if (key == "LOAD_IMAGE")
  {
    std::string filename(value);

            //format the filename to windows syntax
#ifdef WIN32
    filename.replace(filename.begin(), filename.begin() + 20, "R:/");
#else
#endif

    loadImage((char*)filename.c_str());
    nbAnastructsLoaded = 0;
    unsigned int imageIndex = _loadedImages.size() - 1;
    imageChoice->value(imageIndex);
    imageChoice->do_callback();
  }
  else if (key == "LOAD_OVERLAY")
  {
    std::string filename(value);

    loadImage((char*)filename.c_str());
    nbAnastructsLoaded = 0;
    unsigned int overlayIndex = overlayChoice->size()-2;
    overlayChoice->value(overlayIndex);
    overlayChoice->do_callback();
  }
  else if (key == "IMAGE_CHOICE")
  {
    imageChoice->value(StringUtils::toInt(value));
    imageChoice->do_callback();
  }
  else if (key == "OVERLAY_CHOICE")
  {
    overlayChoice->value(StringUtils::toInt(value));
    overlayChoice->do_callback();
  }
  else if(key == "LOAD_ANASTRUCT")
  {
    std::string fileName(value);
    //unsigned int imageIndex = (imageChoice->size() - 2);
    unsigned int imageIndex = _loadedImages.size() - 1;
    _loadAnastruct(fileName,imageIndex);
    nbAnastructsLoaded++;
  }
  else if(key == "LOAD_BYU")
  {
    std::string fileName(value);
    //unsigned int imageIndex = (imageChoice->size() - 2);
    unsigned int imageIndex = _loadedImages.size() - 1;
    //format the filename to windows syntax
#ifdef WIN32
    fileName.replace(fileName.begin(), fileName.begin() + 20, "R:/");
#else
#endif

    _loadBYU(fileName,imageIndex);
    nbAnastructsLoaded++;
  }
  else if(key == "COLOR_ANASTRUCT")
  {
    Vector3D<double> color;
    //unsigned int imageIndex = imageChoice->size()-2;
    unsigned int imageIndex = _loadedImages.size() - 1;
    _parseVector3DDouble(value,color);
    _imAnaVectors[imageIndex][nbAnastructsLoaded-1].color = color;

    axial2DWindow->setImageAnastructColor(nbAnastructsLoaded-1,
                                          color[0], color[1], color[2]);
    int surfaceIndex = surface3DWindow->getSurfaceIndex(imageIndex,
                                                        nbAnastructsLoaded-1);
    surface3DWindow->setSurfaceColor(surfaceIndex,
                                     color[0], color[1], color[2]);
    surface3DWindow->setAnastructColor(surfaceIndex,
                                       color[0], color[1], color[2]);

    _updateContours();
    surface3DWindow->updateDisplay();
    _redrawImageWindows(); 
    displayBYUListCallback();
  }
  else if(key == "POSITION")
  {
    Vector3D<int> position;
    _parseVector3DInt(value,position);
    _centerImage(position.x, position.y, position.z);
    _redrawImageWindows();
  }
  else if(key == "INTENSITY")
  {
    double min, max;
    _parseIntensityWindows(value, min, max);
    imageValMin->value(min);
    imageValMax->value(max);
    imageValMin->do_callback();
    imageValMax->do_callback();
  }
  else if(key == "ROI_START")
  {
    ROIvisibleCheckButton->value(1);
    ROIvisibleCheckButton->do_callback();
    Vector3D<int> roiStart;
    _parseVector3DInt(value,roiStart);
    roiStartX->value(roiStart.x);
    roiStartY->value(roiStart.y);
    roiStartZ->value(roiStart.z);
    roiPropertyChangedCallback();
  }
  else if(key == "ROI_STOP")
  {  
    Vector3D<int> roiStop;
    _parseVector3DInt(value,roiStop);
    roiStopX->value(roiStop.x);
    roiStopY->value(roiStop.y);
    roiStopZ->value(roiStop.z);
    roiPropertyChangedCallback();
  }
  else if(key == "VIEW_CROSSHAIRS")
  {  
    viewCrosshairsButton->value(StringUtils::toBool(value));
    viewCrosshairsButton->do_callback();
  }
  else if(key == "VIEW_IMAGE_INFO")
  {  
    viewImageInfoButton->value(StringUtils::toBool(value));
    viewImageInfoButton->do_callback();
  }
  else if(key == "LOCK_INTENSITIES")
  {  
    lockedIntensity->value(StringUtils::toBool(value));
    lockedIntensity->do_callback();
  }
  else if(key == "LINE_WIDTH")
  {
    lineWidthCounter->value(StringUtils::toDouble(value));
    lineWidthCounter->do_callback();
  }
  else if(key == "WAIT_FOR_CLICK")
  {
    fl_alert(value.c_str());
  }
  else if(key == "WAIT")
  {
    //double time = StringUtils::toDouble(value);
    int time = StringUtils::toInt(value);
    std::cout<<"wait start...";
    for(int t=0;t<time;t++){
      Fl::flush();
      Fl::wait(1.0);
      sleep(1);
    }
    std::cout<<"done"<<std::endl;
  }
  else if(key == "SCREEN_CAPTURE_FILENAME")
  {
    screenCaptureFilename = value;
  }
  else if(key == "TAKE_SCREEN_CAPTURE")
  {
    if (value == "all")
    {
      screenCaptureAllCallback(); 
    }
    else if(value == "axial")
    {
      screenCaptureAxialCallback(); 
    }
    else if(value == "sagittal")
    {
      screenCaptureSagittalCallback(); 
    }
    else if(value == "coronal")
    {
      screenCaptureCoronalCallback(); 
    }
    else
    {
      std::cout<<"Error : Wrong screen capture "<<std::endl;
    }
  }
  else if(key == "TAKE_SCREEN_CAPTURE_WITH_FILENAME")
  {
    //format the filename to windows syntax
#ifdef WIN32
    screenCaptureFilename.replace(screenCaptureFilename.begin(), screenCaptureFilename.begin() + 20, "R:/");
#else
#endif
    if (value == "all")
    {
      _screenCaptureAll(screenCaptureFilename); 
    }
    else if(value == "axial")
    {
      screenCaptureFilename+="_Axial.ppm";
      _screenCaptureAxial(screenCaptureFilename); 
    }
    else if(value == "sagittal")
    {
      screenCaptureFilename+="_Sagittal.ppm";
      _screenCaptureSagittal(screenCaptureFilename); 
    }
    else if(value == "coronal")
    {
      screenCaptureFilename+="_Coronal.ppm";
      _screenCaptureCoronal(screenCaptureFilename); 
    }
    else
    {
      std::cout<<"Error : Wrong screen capture "<<std::endl;
    }
  }
  else if(key == "UNLOAD_IMAGE")
  {
    unsigned int imageIndex=StringUtils::toInt(value);  
    //little trick : if the imageIndex is 0, we remove the current image loaded
    if(imageIndex == 0)
    {
      _unloadImage(imageChoice->value());
    }
    else
    {
      _unloadImage(imageIndex);
    }
  }
  else if(key == "UNLOAD_IMAGE_WITH_QUESTION")
  {
    unsigned int imageIndex=StringUtils::toInt(value);  
    //little trick : if the imageIndex is 0, we remove the current image loaded
    if(imageIndex == 0)
    {
      if(fl_choice("Do you want to unload this image","No","Yes",NULL) == 1)
      {
        _unloadImage(imageChoice->value());
      }
    }
    else
    {
      if(fl_choice("Do you want to unload this image","NO","Yes",NULL) == 1)
      {
        _unloadImage(imageIndex);
      }
    }
  } 
  else if (key == "EXIT")
  {
    std::exit(0);
  }
  else
  {
    std::cerr << "Unknown key: " << key << std::endl;
    return false;
  }  
  return true;
}


void
ImMap
::scriptCallback()
{
  char* fileName = fl_file_chooser(" Choose a script file ","","");
  if (fileName == NULL)
  {
    std::cout<<"Error : incorrect script file .."<<std::endl;
      return;
  }
  runScript(fileName);
}

void
ImMap
::runScript(const std::string& filename)
{

  BasicFileParser bfp;
  bfp.parseFile(filename);

  for (BasicFileParser::StrStrList::const_iterator p = 
    bfp.keyValuePairs.begin(); 
  p != bfp.keyValuePairs.end(); ++p)
  {
    std::cerr << p->first << " -->> " << p->second << std::endl;
    applyScript(p->first, p->second);
  }

}


/////////////////////////
// saveImageCallback() //
/////////////////////////

void
ImMap
::saveImageCallback()
{
  int saveIndex = imageSaveChoice->value();

  // don't save "NONE"
  if (saveIndex == 0)
  {
    return;
  }

  // Open a file selection dialog
  char *fileName = fl_file_chooser("Choose Save As file...", "", 0);
  if( fileName == (char*)0 ) {
    std::cout << "[ImMap::saveImageCallback] file chooser failed to get file";
    std::cout << std::endl;
    return;
  }

  // update status bar
  std::ostringstream oss;
  oss << "Saving " << fileName;
  _updateStatusBuffer(oss.str().c_str());

  // save image data
  try {
    _saveImage(_loadedImages[saveIndex], fileName);
  }
  catch (BasicException e)
  {
    fl_message(e.getMessage());
    _updateStatusBuffer("Save cancelled");

    return;
  }
  catch (...)
  {
    fl_alert("Save failed.");
  }

  // update status bar
  _updateStatusBuffer("Saving ... Done");
}

///////////////////////////
// unloadImageCallback() //
///////////////////////////

void
ImMap
::unloadImageCallback()
{
  // get selected image index
  int removeIndex = imageSaveChoice->value();
  // dont remove "NONE"
  if (removeIndex == 0) 
  {
    return;
  }
  _unloadImage(removeIndex);

}

///////////////////////////////
// _unloadImage(removeIndex) //
///////////////////////////////

void 
ImMap
::_unloadImage(const unsigned int& removeIndex)
{
  unsigned int imageIndex = imageChoice->value();
  unsigned int overlayIndex = overlayChoice->value();
  // update status bar
  std::ostringstream oss;
  oss << "Unloading Image " << removeIndex;
  _updateStatusBuffer(oss.str().c_str());

  // if unloading currently showing image, switch to not show image
  if (imageIndex == removeIndex)
  {
    imageChoice->value(0);
    imageChoiceCallback();
  }
  if (overlayIndex == removeIndex)
  {
    overlayChoice->value(0);
    overlayChoiceCallback();
  }

  // remove item from selection lists
  imageChoice->remove(removeIndex);
  overlayChoice->remove(removeIndex);
  imageSaveChoice->remove(removeIndex);

  //BYU table 
  _totalImagesLoaded--;
  unsigned int BYUImageListValue = BYUImageList->value();
  if ( removeIndex == BYUImageListValue )
  { 
    BYUImageList->value(0);
    displayBYUListCallback();
  }
  BYUImageList->remove(removeIndex);
  displayBYUListCallback();

  // remove image pointer
  delete _loadedImages[removeIndex];
  _loadedImages.erase(_loadedImages.begin() + removeIndex);  


  // remove the corresponding histogram, name, Anastruct vector and Surface vector
  _histograms.erase(_histograms.begin() + removeIndex);  
  _imageNames.erase(_imageNames.begin() + removeIndex);
  _imageFullFileNames.erase(_imageFullFileNames.begin() + removeIndex);

  _imAnaVectors.erase(_imAnaVectors.begin() + removeIndex);

  // clean up selection lists
  imageSaveChoice->value(removeIndex - 1);

  if (removeIndex < imageIndex)
  {
    imageChoice->value(imageIndex - 1);
  }

  if (removeIndex < overlayIndex)
  {
    overlayChoice->value(overlayIndex - 1);
  }
  if ( removeIndex < BYUImageListValue ){
    BYUImageList->value(BYUImageListValue-1);
    displayBYUListCallback();
  }

  // redraw choice boxes
  imageChoice->redraw();
  overlayChoice->redraw();
  imageSaveChoice->redraw();

  // update status bar
  _updateStatusBuffer("Unloading Image ... done");
}

///////////////////////////////////////////
// clickCallback(int,int,int,int, void*) //
///////////////////////////////////////////

void 
ImMap
::clickCallback(int button, float xIndex, float yIndex, float zIndex, void* obj)
{
  switch (button)
  {
  case 1:
    break;
  case 2:
    // move click point to center of window
    ((ImMap*)obj)->_centerImage(xIndex, yIndex, zIndex);
    ((ImMap*)obj)->_redrawImageWindows();
    std::cout << "ImageCenter: " << ((ImMap*)obj)->getImageCenter() << std::endl;
    break;
  case 3:

    break;
  default:
    break;
  }
}

// should probably send pointer to 
// proper window instead of orientation

////////////////////////////////////
// roiChangedCallback(int, void*) //
////////////////////////////////////

void 
ImMap
::roiChangedCallback(IViewOrientationType orientation, void* obj)
{
  int startX, startY, startZ,
    stopX, stopY, stopZ;

  switch (orientation)
  {
  case (IVIEW_AXIAL):
    ((ImMap*)obj)->axial2DWindow->getROI
      (startX, startY, startZ, stopX, stopY, stopZ);
    break;
  case (IVIEW_CORONAL):
    ((ImMap*)obj)->coronal2DWindow->getROI
      (startX, startY, startZ, stopX, stopY, stopZ);
    break;
  default:
  case (IVIEW_SAGITTAL):
    ((ImMap*)obj)->sagittal2DWindow->getROI
      (startX, startY, startZ, stopX, stopY, stopZ);
    break;
  }
  ((ImMap*)obj)->_setROI(startX, startY, startZ,
    stopX, stopY, stopZ);
}

//////////////////////////////////
// MinMaxChangedCallback(void*) //
//////////////////////////////////

void
ImMap
::MinMaxChangedCallback(void* obj)
{
  double ImageRelativeMin = (double)((ImMap*)obj)->HistogramDisplay->updateRelativeMin();
  double ImageRelativeMax = (double)((ImMap*)obj)->HistogramDisplay->updateRelativeMax();
  double OverlayRelativeMin = (double)((ImMap*)obj)->HistogramOverlayDisplay->updateRelativeMin();
  double OverlayRelativeMax = (double)((ImMap*)obj)->HistogramOverlayDisplay->updateRelativeMax();

  ((ImMap*)obj)->imageValMin->value(ImageRelativeMin);
  ((ImMap*)obj)->imageValMax->value(ImageRelativeMax);
  ((ImMap*)obj)->imageValMinMaxCallback();

  if ((((ImMap*)obj)->lockedIntensity->value() == 1) && (((ImMap*)obj)->overlayChoice->value()!=0))
  {
    ((ImMap*)obj)->overlayValMin->value(ImageRelativeMin);
    ((ImMap*)obj)->overlayValMax->value(ImageRelativeMax);
  }
  else
  {
    ((ImMap*)obj)->overlayValMin->value(OverlayRelativeMin);
    ((ImMap*)obj)->overlayValMax->value(OverlayRelativeMax);
    ((ImMap*)obj)->overlayValMinMaxCallback();
  }

}




//////////////////////////
// imageChoiceCallack() //
//////////////////////////

void
ImMap
::imageChoiceCallback()
{
  unsigned int imageIndex = imageChoice->value();
  _imageChoice(imageIndex);
}

//////////////////////////////
// _imageChoice(imageIndex) //
//////////////////////////////
void
ImMap
::_imageChoice(const unsigned int& imageIndex)
{
  ImagePointer chosenImage = _loadedImages[imageIndex];
  float minIntensity = _histograms[imageIndex].getAbsoluteMin();
  float maxIntensity = _histograms[imageIndex].getAbsoluteMax();
  std::string imageName = _imageNames[imageIndex];
 
#if 0
  std::cerr << "Image Selection: " << imageIndex << std::endl;
  std::cerr << "\tName: " << imageName << std::endl;
  std::cerr << "\tMin Intensity: " << minIntensity << std::endl;
  std::cerr << "\tMax Intensity: " << maxIntensity << std::endl;
#endif

  try
  {
    axial2DWindow->setImage(chosenImage, minIntensity, maxIntensity);
    axial2DWindow->setImageName(imageName);
    coronal2DWindow->setImage(chosenImage, minIntensity, maxIntensity);
    coronal2DWindow->setImageName(imageName);
    sagittal2DWindow->setImage(chosenImage, minIntensity, maxIntensity);
    sagittal2DWindow->setImageName(imageName);
  }
  catch (std::invalid_argument e)
  {
    std::cerr << e.what() << std::endl;
    _updateStatusBuffer("Image incompatible with loaded overlay, deselect overlay first");
    imageChoice->value(0);
  }

  //clear the image Info
  DimensionsInfoImage->value("");
  PixelSizeInfoImage->value("");
  OriginInfoImage->value("");
  OrientationInfoImage->value("");
  MinMaxInfoImage->value("");
  DataTypeInfoImage->value("");

  if(imageIndex != 0)
  {
    // We update the histogram, in case it is displayed
    histogramImageCallback();
    infoImageCallback();
    // presetIntensityCallback(1);
    imagePropertyChangedCallback();
  }
  else
  {

    //clear the histogram
    HistogramDisplay->clear();

    //clear overlay Min Max value
    imageValMin->value(0);
    imageValMax->value(0);
  }


  _updateContours();
  _updateSurfaces();
  _updateImagePanel();
  _updateROIPanel();
  _updatePositionSliders();
  _redrawImageWindows();
}
/////////////////////////////
// overlayChoiceCallback() //
/////////////////////////////

void
ImMap
::overlayChoiceCallback()
{
  int overlayIndex = overlayChoice->value();

  try
  {
    axial2DWindow->setOverlay(_loadedImages[overlayIndex],
      _histograms[overlayIndex].getAbsoluteMin(),
      _histograms[overlayIndex].getAbsoluteMax());
    coronal2DWindow->setOverlay(_loadedImages[overlayIndex],
      _histograms[overlayIndex].getAbsoluteMin(),
      _histograms[overlayIndex].getAbsoluteMax());
    sagittal2DWindow->setOverlay(_loadedImages[overlayIndex],
      _histograms[overlayIndex].getAbsoluteMin(),
      _histograms[overlayIndex].getAbsoluteMax());
  }
  catch (std::invalid_argument e)
  {
    std::cerr << e.what() << std::endl;
    _updateStatusBuffer("Overlay incompatible with loaded image, deselect image first");

    overlayChoice->value(0);
  }

  //clear the overlay Info
  DimensionsInfoOverlay->value("");
  PixelSizeInfoOverlay->value("");
  OriginInfoOverlay->value("");
  OrientationInfoOverlay->value("");
  MinMaxInfoOverlay->value("");
  DataTypeInfoOverlay->value("");


  if (overlayIndex != 0){
    // We update the histogramWindow
    histogramOverlayCallback();
    infoOverlayCallback();
    // presetIntensityCallback(2);
    overlayPropertyChangedCallback();
  }
  else
  {


    //clear the histogram
    HistogramOverlayDisplay->clear();

    //clear overlay Min Max value
    overlayValMin->value(0);
    overlayValMax->value(0);
  }
  _updateContours();
  _updateSurfaces();
  _updateImagePanel();
  _updateROIPanel();
  _updatePositionSliders();
  _redrawImageWindows();
}

/////////////////////////
// scrollbarCallback() //
/////////////////////////

void
ImMap
::scrollbarCallback()
{
  //display the crossair in the middle of the voxel (+0.5)
  _centerImage(sagittalScrollBar->value()+0.5,
    coronalScrollBar->value()+0.5,
    axialScrollBar->value()+0.5);
  _redrawImageWindows();
}

////////////////////////////////////
// imagePropertyChangedCallback() //
////////////////////////////////////

void
ImMap
::imagePropertyChangedCallback()
{
  int colorIndex = colorPresetChoice->value();
  axial2DWindow->setImageColorRGB(_colorPresetIntensity[colorIndex].image.red,
    _colorPresetIntensity[colorIndex].image.green,
    _colorPresetIntensity[colorIndex].image.blue);

  coronal2DWindow->setImageColorRGB(_colorPresetIntensity[colorIndex].image.red,
    _colorPresetIntensity[colorIndex].image.green,
    _colorPresetIntensity[colorIndex].image.blue);

  sagittal2DWindow->setImageColorRGB(_colorPresetIntensity[colorIndex].image.red,
    _colorPresetIntensity[colorIndex].image.green,
    _colorPresetIntensity[colorIndex].image.blue);
  HistogramDisplay->setLUTcolor(_colorPresetIntensity[colorIndex].image.red,
    _colorPresetIntensity[colorIndex].image.green,
    _colorPresetIntensity[colorIndex].image.blue);

  int imageIndex = imageChoice->value();
  if (imageIndex != 0)
  {
    _redrawImageWindows();
  }
} 

//////////////////////////////
// imageIWChangedCallback() //
//////////////////////////////

void
ImMap
::imageIWChangedCallback()
{
  axial2DWindow->setImageIWMin(imageValMin->value());
  axial2DWindow->setImageIWMax(imageValMax->value());

  coronal2DWindow->setImageIWMin(imageValMin->value());
  coronal2DWindow->setImageIWMax(imageValMax->value());

  sagittal2DWindow->setImageIWMin(imageValMin->value());
  sagittal2DWindow->setImageIWMax(imageValMax->value());

  int imageIndex = imageChoice->value();
  if (imageIndex != 0)
  {
    _redrawImageWindows();
  }
}

//////////////////////////////////////
// overlayPropertyChangedCallback() //
//////////////////////////////////////

void
ImMap
::overlayPropertyChangedCallback()
{
  int colorIndex = colorPresetChoice->value();
  axial2DWindow->setOverlayColorRGB(_colorPresetIntensity[colorIndex].overlay.red,
    _colorPresetIntensity[colorIndex].overlay.green,
    _colorPresetIntensity[colorIndex].overlay.blue);
  axial2DWindow->setOverlayOpacity(overlayAlphaSlider->value());

  coronal2DWindow->setOverlayColorRGB(_colorPresetIntensity[colorIndex].overlay.red,
    _colorPresetIntensity[colorIndex].overlay.green,
    _colorPresetIntensity[colorIndex].overlay.blue);
  coronal2DWindow->setOverlayOpacity(overlayAlphaSlider->value());

  sagittal2DWindow->setOverlayColorRGB(_colorPresetIntensity[colorIndex].overlay.red,
    _colorPresetIntensity[colorIndex].overlay.green,
    _colorPresetIntensity[colorIndex].overlay.blue);
  sagittal2DWindow->setOverlayOpacity(overlayAlphaSlider->value());

  HistogramOverlayDisplay->setLUTcolor(_colorPresetIntensity[colorIndex].overlay.red,
    _colorPresetIntensity[colorIndex].overlay.green,
    _colorPresetIntensity[colorIndex].overlay.blue);

  int overlayIndex = overlayChoice->value();
  if (overlayIndex != 0)
  {
    _redrawImageWindows();
  }
} 

////////////////////////////////
// overlayIWChangedCallback() //
////////////////////////////////

void
ImMap
::overlayIWChangedCallback()
{
  axial2DWindow->setOverlayIWMin(overlayValMin->value());
  axial2DWindow->setOverlayIWMax(overlayValMax->value());

  coronal2DWindow->setOverlayIWMin(overlayValMin->value());
  coronal2DWindow->setOverlayIWMax(overlayValMax->value());

  sagittal2DWindow->setOverlayIWMin(overlayValMin->value());
  sagittal2DWindow->setOverlayIWMax(overlayValMax->value());

  int overlayIndex = overlayChoice->value();
  if (overlayIndex != 0)
  {
    _redrawImageWindows();
  }
}


////////////////////
// zoomCallback() //
////////////////////

void
ImMap
::zoomCallback()
{
  axial2DWindow->setWindowZoom(imageZoomVal->value());
  coronal2DWindow->setWindowZoom(imageZoomVal->value());
  sagittal2DWindow->setWindowZoom(imageZoomVal->value());

  _redrawImageWindows();
} 

//////////////////////////////////
// roiPropertyChangedCallback() //
//////////////////////////////////
void
ImMap
::roiPropertyChangedCallback()
{

  int imageIndex = imageChoice->value();
  int overlayIndex = overlayChoice->value();
  if ((imageIndex !=0)||(overlayIndex != 0))
  {
    if (_ROIcreated == false)
    {
      roiInitializeCallback();
    }
    bool isVisible = ROIvisibleCheckButton->value() ? true : false;
    axial2DWindow->setViewROI(isVisible);
    coronal2DWindow->setViewROI(isVisible);
    sagittal2DWindow->setViewROI(isVisible);
    surface3DWindow->setViewROI(isVisible);

    axial2DWindow->setROIOpacity(roiOpacitySlider->value());
    coronal2DWindow->setROIOpacity(roiOpacitySlider->value());
    sagittal2DWindow->setROIOpacity(roiOpacitySlider->value());
    surface3DWindow->setROIOpacity(roiOpacitySlider->value());

    int startX = static_cast<int>(roiStartX->value());
    int startY = static_cast<int>(roiStartY->value());
    int startZ = static_cast<int>(roiStartZ->value());

    int stopX = static_cast<int>(roiStopX->value());
    int stopY = static_cast<int>(roiStopY->value());
    int stopZ = static_cast<int>(roiStopZ->value());

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

    axial2DWindow->setROI(startX, startY, startZ,
      stopX, stopY, stopZ);
    coronal2DWindow->setROI(startX, startY, startZ,
      stopX, stopY, stopZ);
    sagittal2DWindow->setROI(startX, startY, startZ,
      stopX, stopY, stopZ);
    _updateROIInfo();
    _updateROIin3DWindow();
    _redrawImageWindows();
  }
}

////////////////////////
// maskViewCallback() //
////////////////////////

void 
ImMap
::maskViewCallback()
{
  int imageIndex = imageChoice->value();

  if(imageIndex == 0)
  {
    fl_alert("Please select an image first!");
    viewMaskButton->value(0);
    return;
  }

  axial2DWindow->setMask(_mask[maskChoice->value()]);
  coronal2DWindow->setMask(_mask[maskChoice->value()]);
  sagittal2DWindow->setMask(_mask[maskChoice->value()]);

  bool isVisible = viewMaskButton->value() ? true : false;
  axial2DWindow->setViewMask(isVisible);
  coronal2DWindow->setViewMask(isVisible);
  sagittal2DWindow->setViewMask(isVisible);
      _redrawImageWindows();


}

///////////////////////
// roiLockCallback() //
///////////////////////

void
ImMap
::roiLockCallback()
{
  bool activateGroup = roiLockedButton->value() ? false : true;
  if (activateGroup)
  {
    roiCoordinatesGroup->activate();
    axial2DWindow->setROILocked(false);
    coronal2DWindow->setROILocked(false);
    sagittal2DWindow->setROILocked(false);
  }
  else
  {
    roiCoordinatesGroup->deactivate();
    axial2DWindow->setROILocked(true);
    coronal2DWindow->setROILocked(true);
    sagittal2DWindow->setROILocked(true);
  }

}

//////////////////////////////////
// roiImageAddAsImageCallback() //
//////////////////////////////////

void
ImMap
::roiImageAddAsImageCallback()
{
  // make sure image is loaded
  if (!axial2DWindow->haveValidImage()) 
  {
    _updateStatusBuffer("Error: ROI as Image: no image loaded");
    return;
  }

  ImagePointer ImageFromROI = 0;

  ImageFromROI = new ImageType;

  int imageIndex = imageChoice->value();

  _createImageFromROI(_loadedImages[imageIndex], ImageFromROI);

  //add an empty ImAna vector to the list
  _imAnaVectors.push_back(std::vector<ImAna>());

  _imageNames.push_back("Image ROI");
  _imageFullFileNames.push_back("Image ROI");

  // add image to list
  ImageFromROI->setDataType( Image<float>::Float );
  ImageFromROI->setOrientation(_loadedImages[imageIndex]->getOrientation());
  _addImageToList(ImageFromROI);
}

////////////////////////////////////
// roiOverlayAddAsImageCallback() //
////////////////////////////////////

void
ImMap
::roiOverlayAddAsImageCallback()
{
  // make sure overlay is loaded
  if (!axial2DWindow->haveValidOverlay()) 
  {
    _updateStatusBuffer("Error: ROI as Image: no overlay loaded");
    return;
  }

  ImagePointer OverlayFromROI = 0;
  OverlayFromROI = new ImageType;

  int overlayIndex = overlayChoice->value();

  _createImageFromROI(_loadedImages[overlayIndex], OverlayFromROI);

  //add an empty ImAna vector to the list
  _imAnaVectors.push_back(std::vector<ImAna>());

  _imageNames.push_back("Overlay ROI");
  _imageFullFileNames.push_back("Overlay ROI");
  OverlayFromROI->setDataType( Image<float>::Float );
  OverlayFromROI->setOrientation(_loadedImages[overlayIndex]->getOrientation());
  _addImageToList(OverlayFromROI);


}

////////////////////////////
// roiInitializeCallback() //
////////////////////////////

void
ImMap
::roiInitializeCallback()
{
  if (axial2DWindow->haveValidImage() || axial2DWindow->haveValidOverlay())
  {
    int startX = axial2DWindow->getImageDimX() / 3;
    int startY = axial2DWindow->getImageDimY() / 3;
    int startZ = axial2DWindow->getImageDimZ() / 3;
    int stopX  = startX * 2;
    int stopY  = startY * 2;
    int stopZ  = startZ * 2;
    surface3DWindow->createROI();
    _setROI(startX, startY, startZ, stopX, stopY, stopZ);
    ROIvisibleCheckButton->value(true);
    _ROIcreated = true;
    roiPropertyChangedCallback(); 
  }
}

////////////////////////////////////////
//roiInitializeRegistrationCallback() //
////////////////////////////////////////

void
ImMap
::roiInitializeRegistrationCallback()
{
  int next = nextStepChoice->value();
  if (next == 0)
  {
    fl_alert("Please select the organ studied !");
    return;
  }

  step3Button->value(1);
  selectDailyGroup->hide();
  if (next == 2)
  {
    resampleCallback();
    step3Button->value(1);
    selectDailyGroup->hide();
    int indexImage = _loadedImages.size()-1;
    overlayChoice->value(indexImage);
    overlayChoiceCallback();
    loadAnastructsGroup->show();
    wizardLoadAnastructsCallback();
  }
  else{
    step3Button->value(1);
    selectDailyGroup->hide();
    setROIGroup->show();
    imagePresetIntensityChoice->value(1);
    presetIntensityCallback(1);
    overlayPresetIntensityChoice->value(1);
    presetIntensityCallback(2);

  }
  if (axial2DWindow->haveValidImage() || axial2DWindow->haveValidOverlay())
  {
    int startX = axial2DWindow->getImageDimX() / 3;
    int startY = axial2DWindow->getImageDimY() / 3;
    int startZ = axial2DWindow->getImageDimZ() / 3;
    int stopX  = startX * 2;
    int stopY  = startY * 2;
    int stopZ  = startZ * 2;
    _setROI(startX, startY, startZ, stopX, stopY, stopZ);
  }
}


////////////////////////
// downSizeCallback() //
////////////////////////

void 
ImMap
::downSizeCallback()

{
  //debug gaussian filter

  // make sure two images are loaded
  if (!axial2DWindow->haveValidImage()) 
  {
    _updateStatusBuffer("Error: Translate Register: No Image Loaded");
    return;
  }

  // set up parameters
  int imageIndex = imageChoice->value();
  ImagePointer image = _loadedImages[imageIndex];
  Vector3D<double> scaleFactors(2,2,2);

  //Compute Gaussian filter
  GaussianFilter3D* filter = new GaussianFilter3D();
  filter->SetInput(*image);
  filter->setSigma(1*scaleFactors.x,1*scaleFactors.y,1*scaleFactors.z);
  filter->setFactor((int)scaleFactors.x, (int)scaleFactors.y,
                    (int)scaleFactors.z);
  filter->setKernelSize((int)(2*scaleFactors.x), (int)(2*scaleFactors.y),
                        (int)(2*scaleFactors.z));
  filter->Update();

  //debug output the downsample image
  std::string filename("F:/David prigent/unc/Matlab/TestImage/GaussianImage.raw");  
  Array3DIO::writeRawVolume(filter->GetOutput(),filename.c_str());
  //debug output the downsample image

  //debug gaussian filter



}

///////////////////////////
// downSampleCallback() //
//////////////////////////

void
ImMap
::downSampleCallback()
{
#if 0
  // make sure two images are loaded
  if (!axial2DWindow->haveValidImage()) 
  {
    _updateStatusBuffer("Error: Translate Register: No Image Loaded");
    return;
  }

  // set up parameters
  int imageIndex = imageChoice->value();
  ImagePointer image = _loadedImages[imageIndex];

  /** Downsample the image */
  DownsampleFilter3D filter;
  filter.SetInput(*image);
  int FactorX = static_cast<int>(ScaleFactorValueX->value()+0.5);
  int FactorY = static_cast<int>(ScaleFactorValueY->value()+0.5);
  int FactorZ = static_cast<int>(ScaleFactorValueZ->value()+0.5);
  filter.SetFactor(FactorX,FactorY,FactorZ);
  filter.SetSigma(FactorX,FactorY,FactorZ);
  filter.SetSize(FactorX*2,FactorY*2,FactorZ*2);
  filter.Update();


  /** Create new downsampled image */

  ImagePointer shrinkImage = new ImageType(filter.GetNewSize());
  shrinkImage->setData(filter.GetOutput());

  ImageSizeType imagesize(shrinkImage->getSizeX(),
    shrinkImage->getSizeY(),
    shrinkImage->getSizeZ());
  ImageSizeType spacing = image->getSpacing();
  spacing.scale(FactorX,FactorY,FactorZ);
  //Vector3D<double> product(((Vector3D<double>)image->getSize()) * (image->getSpacing()));
  //Vector3D<double> spacing((product/(shrinkImage->getSize())));
  float origin_x= ((imagesize[0]/2)*spacing.x*(-1));
  float origin_y=((imagesize[1]/2)*spacing.y*(-1));
  float origin_z= ((imagesize[2]/2)*spacing.z*(-1));

  ImageIndexType origin(origin_x, origin_y,origin_z);


  shrinkImage->setOrigin(image->getOrigin());
  shrinkImage->setSpacing(spacing);

  // add a translated surfaces and anastructs
  _updateStatusBuffer("end of downSampling...");  

  //add an empty ImAna vector to the list
  _imAnaVectors.push_back(std::vector<ImAna>());

  std::string imageName(" downsample image");
  _imageNames.push_back(imageName);
  _imageFullFileNames.push_back(imageName);

  _addImageToList(shrinkImage);
#endif
}


//////////////////////////////
// affineRegisterCallback() //
//////////////////////////////

void
ImMap
::affineRegisterCallback()
{

  _updateStatusBuffer("Starting Affine Register...");

  // make sure two images are loaded
  if (!axial2DWindow->haveValidImage()) {
    _updateStatusBuffer("Error: Affine Register: No Image Loaded");
    return;
  }

  if (!axial2DWindow->haveValidOverlay()) {
    _updateStatusBuffer("Error: Affine Register: No Overlay Loaded");
    return;
  }

  // set up parameters and run the registration
  int imageIndex = imageChoice->value();
  int overlayIndex = overlayChoice->value();
  ImagePointer atlas = _loadedImages[imageIndex];
  ImagePointer subject = _loadedImages[overlayIndex];
  int useROI = useROIButton->value();
  bool useIntensityWindowing = (imagePresetIntensityChoice->value() != 0);
  EstimateAffine::OutputMode verbosity = EstimateAffine::VERBOSE;

  if (roiStartX->value()==0 && roiStopX->value()==0 &&
      roiStartY->value()==0 && roiStopY->value()==0 &&
      roiStartZ->value()==0 && roiStopZ->value()==0 &&
      useROI)
  {
    fl_alert("You haven't set the ROI.");
    return;
  }

  EstimateAffine estimateAffine(atlas, subject, useIntensityWindowing,
                                verbosity);

  EstimateAffine::ScheduleType pyramidSchedule = _getPyramidSchedule();
  estimateAffine.SetShrinkSchedule(pyramidSchedule);

  if (useIntensityWindowing) {
    estimateAffine.SetMinMax( _histograms[imageIndex].getRelativeMin(),
                              _histograms[imageIndex].getRelativeMax(),
                              _histograms[overlayIndex].getRelativeMin(),
                              _histograms[overlayIndex].getRelativeMax() );
  }

  if (useROI) estimateAffine.SetROI( _getROI() );

  std::string imageNameString;
  switch (affineChoice->value()) {
  case 0: 
    estimateAffine.RunEstimateTranslation(); 
    imageNameString = "image->overlay translation";
    break;
  case 1: 
    estimateAffine.RunEstimateRotation();
    imageNameString = "image->overlay rigid";
    break;
  case 2: 
    estimateAffine.RunEstimateAffine();
    imageNameString = "image->overlay affine";
    break;
  }

  registrationTransform = estimateAffine.GetTransform();
  _updateTransformDisplay();
  
  // create and add deformed image and anastructs
  _applyAffineTransform(registrationTransform,
                        imageIndex, overlayIndex,
                        imageNameString);
                        
  //
  // old code that did not handel anas properly
  //

  // add image to loaded image list and ImIna vector to its list
  //ImagePointer registeredImage = new ImageType;
  //estimateAffine.CreateRegisteredImage(registeredImage);

  //_imAnaVectors.push_back(std::vector<ImAna>());

  //_imageNames.push_back(imageNameString);
  //_imageFullFileNames.push_back(imageNameString);
  //registeredImage->setDataType( Image<float>::Float );
  //registeredImage->setOrientation(_loadedImages[imageIndex]->getOrientation());
  //_addImageToList(registeredImage);

  _printParameters("AFFINE_REGISTER_ROI_START", "AFFINE_REGISTER_ROI_STOP",
                   "AFFINE_REGISTER_INTENSITY_WINDOW");

}

///////////////////////////////
// fluidParametersCallback() //
///////////////////////////////

void
ImMap
::fluidParametersCallback()
{

  unsigned int paramIndex = fluidParamChoice->value();


  if ((_fluidParamPreset[paramIndex].params[0].numIterations)==0)
  {
    fluidFineParameterGroup->deactivate();
    fluidFineButton->value(0);
  }
  else
  {
    fluidFineParameterGroup->activate();
    fluidFineButton->value(1);
    fluidFineMaxIterations->value(_fluidParamPreset[paramIndex].params[0].numIterations);
    fluidFineAlpha->value(_fluidParamPreset[paramIndex].params[0].alpha);
    fluidFineBeta->value(_fluidParamPreset[paramIndex].params[0].beta);
    fluidFineGamma->value(_fluidParamPreset[paramIndex].params[0].gamma);
    fluidFineMaxPerturbation->value(_fluidParamPreset[paramIndex].params[0].maxPerturbation);
  }

  if ((_fluidParamPreset[paramIndex].params[1].numIterations)==0)
  {
    fluidMediumParameterGroup->deactivate();
    fluidMediumButton->value(0);
  }
  else
  {
    fluidMediumParameterGroup->activate();
    fluidMediumButton->value(1);
    fluidMediumMaxIterations->value(_fluidParamPreset[paramIndex].params[1].numIterations);
    fluidMediumAlpha->value(_fluidParamPreset[paramIndex].params[1].alpha);
    fluidMediumBeta->value(_fluidParamPreset[paramIndex].params[1].beta);
    fluidMediumGamma->value(_fluidParamPreset[paramIndex].params[1].gamma);
    fluidMediumMaxPerturbation->value(_fluidParamPreset[paramIndex].params[1].maxPerturbation);
  }

  if ((_fluidParamPreset[paramIndex].params[2].numIterations)==0)
  {
    fluidCoarseParameterGroup->deactivate();
    fluidCoarseButton->value(0);
  }
  else
  {
    fluidCoarseParameterGroup->activate();
    fluidCoarseButton->value(1);
    fluidCoarseMaxIterations->value(_fluidParamPreset[paramIndex].params[2].numIterations);
    fluidCoarseAlpha->value(_fluidParamPreset[paramIndex].params[2].alpha);
    fluidCoarseBeta->value(_fluidParamPreset[paramIndex].params[2].beta);
    fluidCoarseGamma->value(_fluidParamPreset[paramIndex].params[2].gamma);
    fluidCoarseMaxPerturbation->value(_fluidParamPreset[paramIndex].params[2].maxPerturbation);
  }
}

///////////////////////////////////
// saveFluidParametersCallback() //
///////////////////////////////////
void
ImMap
::saveFluidParametersCallback()
{
  char* presetFilename = (char*)_presetFilename.c_str();
  try
  {
    std::ofstream outputASCII(presetFilename,ofstream::out|ofstream::app);

    if (outputASCII.fail())
    {
      throw std::runtime_error("failed to open file for ascii write");
    }

    bool noName;
    do{
      noName=false;
      fluidParametersNameWindow->show(); 
      while (fluidParametersNameWindow->shown()) Fl::wait();
      //make sure that name was entered
      if (strcmp(fluidParametersNameInput->value(),"")==0)
      {
        noName = true;
        noNameOutput->value("no name entered");
      }
    }while(noName==true);
    noNameOutput->value("");

    outputASCII << "FLUID_PARAM_NAME=\""<<fluidParametersNameInput->value()<< "\"\n";

    outputASCII << "FINE_ALPHA=\"" << fluidFineAlpha->value() << "\"\n";
    outputASCII << "FINE_BETA=\"" << fluidFineBeta->value() << "\"\n";
    outputASCII << "FINE_GAMMA=\"" << fluidFineGamma->value() << "\"\n";
    outputASCII << "FINE_MAXPERT=\"" << fluidFineMaxPerturbation->value() << "\"\n";
    outputASCII << "FINE_NUMBASIS=\"" << 2000 << "\"\n";
    outputASCII << "FINE_NUMITER=\"" << (fluidFineButton->value() == 0 ? 0 : fluidFineMaxIterations->value()) << "\"\n";

    outputASCII << "MEDIUM_ALPHA=\"" << fluidMediumAlpha->value() << "\"\n";
    outputASCII << "MEDIUM_BETA=\"" << fluidMediumBeta->value() << "\"\n";
    outputASCII << "MEDIUM_GAMMA=\"" << fluidMediumGamma->value() << "\"\n";
    outputASCII << "MEDIUM_MAXPERT=\"" << fluidMediumMaxPerturbation->value()<< "\"\n";
    outputASCII << "MEDIUM_NUMBASIS=\"" << 2000 << "\"\n";
    outputASCII << "MEDIUM_NUMITER=\"" << (fluidMediumButton->value() == 0 ? 0 : fluidMediumMaxIterations->value()) << "\"\n";

    outputASCII << "COARSE_ALPHA=\"" << fluidCoarseAlpha->value() << "\"\n";
    outputASCII << "COARSE_BETA=\"" << fluidCoarseBeta->value() << "\"\n";
    outputASCII << "COARSE_GAMMA=\"" << fluidCoarseGamma->value() << "\"\n";
    outputASCII << "COARSE_MAXPERT=\"" << fluidCoarseMaxPerturbation->value() << "\"\n";
    outputASCII << "COARSE_NUMBASIS=\"" << 2000 << "\"\n";
    outputASCII << "COARSE_NUMITER=\"" << (fluidFineButton->value() == 0 ? 0 : fluidCoarseMaxIterations->value()) << "\"\n";
    outputASCII << "\n";


    if (outputASCII.fail())
    {
      throw std::runtime_error("ofstream failed writing ascii header");
    }
    outputASCII.close();


    //add the new preset to the list
    fluidParameters newParameters;
    //fine
    newParameters.params[0].numIterations = 
      static_cast<int>(fluidFineButton->value() == 0 ? 0 :fluidFineMaxIterations->value());
    newParameters.params[0].alpha = fluidFineAlpha->value();
    newParameters.params[0].beta = fluidFineBeta->value();
    newParameters.params[0].gamma = fluidFineGamma->value();
    newParameters.params[0].maxPerturbation=fluidFineMaxPerturbation->value();
    newParameters.params[0].numBasis=2000;

    //medium
    newParameters.params[1].numIterations = 
      static_cast<int>(fluidMediumButton->value() == 0 ? 0 : fluidMediumMaxIterations->value());
    newParameters.params[1].alpha = fluidMediumAlpha->value();
    newParameters.params[1].beta = fluidCoarseBeta->value();
    newParameters.params[1].gamma = fluidMediumGamma->value();
    newParameters.params[1].maxPerturbation=fluidMediumMaxPerturbation->value();
    newParameters.params[1].numBasis=2000;

    //coarse
    newParameters.params[2].numIterations = 
      static_cast<int>(fluidCoarseButton->value() == 0 ? 0 : fluidCoarseMaxIterations->value());
    newParameters.params[2].alpha = fluidCoarseAlpha->value();
    newParameters.params[2].beta = fluidCoarseBeta->value();
    newParameters.params[2].gamma = fluidCoarseGamma->value();
    newParameters.params[2].maxPerturbation=fluidCoarseMaxPerturbation->value();
    newParameters.params[2].numBasis=2000;

    fluidParamChoice->add(fluidParametersNameInput->value());
    _fluidParamPreset.push_back(newParameters);
    fluidParamChoice->value((_fluidParamPreset.size()-1));
    fluidParametersCallback();

    std::cerr<<" "<<fluidParametersNameInput->value()<<" added to the preset file"<<std::endl;
    fluidParametersNameInput->value("");
  }
  catch (...)
  {
    std::cout<<"Failed to save fluid parameters"<<std::endl;
    return;
  }
}

void ImMap::
fluidRegisterCallback()
{

  // What kind of registration are we doing?
  int choice = deformationChoice->value();

  bool overlayNeeded = (choice == 1 || choice == 2);

  // Before we get a filename to output, make sure we can do the
  // deformation:

  // Make sure we have ROI if we plan to use it
  if (_usingROI() && !_roiIsSet())
  {
    fl_alert("ROI is not set.");
    return;
  }

  // make sure all needed images are loaded
  if (!axial2DWindow->haveValidImage()) 
  {
    _updateStatusBuffer("Error: Fluid Register: No Image Loaded");
    return;
  }
  if (overlayNeeded && !axial2DWindow->haveValidOverlay()) 
  {
    _updateStatusBuffer("Error: Fluid Register: No Overlay Loaded");
    return;
  }

  // Get name for hfield, strip extension if necessary, and make sure
  // file is writable

  char* hFieldFilenameCstyle = 
    fl_file_chooser("Select name for deflation hField file.","",0);
  std::string hFieldFilename("");
  if (hFieldFilenameCstyle) {

    hFieldFilename = hFieldFilenameCstyle;

    size_t mhdLoc = hFieldFilename.rfind(".mhd");
    if (mhdLoc != std::string::npos && mhdLoc == hFieldFilename.size() - 4) 
    {
      hFieldFilename.erase(hFieldFilename.size() - 4);
      std::cout << "You don't need to include the '.mhd' in the "
                << "h-field filename." << std::endl;
    }

    if (!_testFileCanWrite(hFieldFilename + ".mhd"))
    {
      std::ostringstream oss;
      oss << "Can't open file for writing: " << hFieldFilename;
      fl_alert(oss.str().c_str());
      _updateStatusBuffer(oss.str().c_str());
      return;      
    }

  }

  _updateStatusBuffer("Starting Fluid Register...");

  // Now set up all the parameters:

  // Depending on the deformation method, the resulting h-field may be
  // in a different place.  hFieldPtr etc will be set to point to the
  // right one.
  HFieldType hField;
  const HFieldType* hFieldPtr = &hField;
  HFieldType hFieldInv;
  const HFieldType* hFieldInvPtr = &hFieldInv;

  int imageIndex = imageChoice->value();
  ImagePointer image = _loadedImages[imageIndex];
  std::string imageFileName = _imageFullFileNames[imageIndex];

  int overlayIndex = overlayChoice->value();
  ImagePointer overlay = _loadedImages[overlayIndex];
  std::string overlayFileName = _imageFullFileNames[overlayIndex];  

  ImageRegionType roi;
  if (_usingROI())
  {
    roi = _getROI();
  }
  else
  {
    roi.setSize(image->getSize());
  }

  Vector3D<double> hFieldOrigin = roi.getStart();
  image->imageIndexToWorldCoordinates(hFieldOrigin, hFieldOrigin);
  Vector3D<double> hFieldSpacing = image->getSpacing();

  const unsigned int MAX_NUM_SCALE_LEVELS = 3;
  unsigned int numScaleLevels = 0;
  FluidWarpParameters params[MAX_NUM_SCALE_LEVELS];
  _getFluidParameters(params, numScaleLevels);

  // for new image in pulldown list
  std::string imageName;

  // Needed for some deformation choices
  MultiScaleFluidWarp fluidWarpInterface;

  // Now actually run the chosen deformation method

  _updateStatusBuffer("Running fluid deformation");
  Timer timer;
  timer.start();

  try 
  {
    if (_deformationChoices[choice] == "Fluid") 
    {
      std::cerr << "starting multiscale fluid" << std::endl;
      MultiscaleFluid::
        registerImages(hField,
                       *image,
                       *overlay,
                       roi,
                       imageValMin->value(),
                       imageValMax->value(),
                       overlayValMin->value(),
                       overlayValMax->value(),
                       numScaleLevels,
                       params,
                       true);
      std::cerr << "DONE starting multiscale fluid" << std::endl;      
      fluidWarpInterface.setHField(hField, roi);
      imageName = "overlay fluid warped into image";
    }
    else if (_deformationChoices[choice] == "Elastic" || 
             _deformationChoices[choice] == "Elastic With Mask")
    {
      if (_deformationChoices[choice] == std::string("Elastic With Mask") &&
          maskChoice->value()==0)
      {
        fl_alert("Please create or select a mask first.");
        return;
      }
      MultiScaleFluidWarp fluidWarpInterface;
      fluidWarpInterface.createWarp(image, overlay, 
                                    overlayFileName.c_str(),
                                    imageFileName.c_str(),
                                    overlayValMin->value(),
                                    overlayValMax->value(),
                                    imageValMin->value(),
                                    imageValMax->value(),
                                    roi, 
                                    numScaleLevels,
                                    params,
                                    resliceButton->value(),
                                    choice,
                                    _mask[maskChoice->value()]);
      hFieldPtr = &fluidWarpInterface.getHField();
      imageName = "overlay fluid warped into image";
    }
    else if (_deformationChoices[choice] == "Shrink")
    {
      MultiscaleFluid::
        deflateImageTwoWay(hField,
                           hFieldInv,
                           *image,
                           roi,
                           imageValMin->value(),
                           imageValMax->value(),
                           numScaleLevels,
                           params,
                           (unsigned int) numDilationsInput->value(),
                           (unsigned int) numErosionsInput->value(),
                           resliceButton->value(),
                           false);
      fluidWarpInterface.setHField(hFieldInv, roi);
      overlayIndex = imageIndex;
      imageName = "deflated image";
    }
    else if (_deformationChoices[choice] == "Elastic Shrink With Mask")
    {
      if(maskChoice->value()==0)
      {
        fl_alert("Please create or select a mask first.");
        return;
      }
      MultiScaleFluidWarp fluidWarpInterface;
      fluidWarpInterface.shrink(image, 
                                imageFileName.c_str(),
                                imageValMin->value(),
                                imageValMax->value(),
                                roi, 
                                numScaleLevels,
                                params,
                                resliceButton->value(),
                                choice,
                                _mask[maskChoice->value()]);
      hFieldPtr = &fluidWarpInterface.getHField();
      imageName = "deflated image";
    }
    else if (_deformationChoices[choice] == "Deflate Reverse" ||
             _deformationChoices[choice] == "Deflate Forward")
    {
      MultiscaleFluid::AlgorithmType deflationType =
        (_deformationChoices[choice] == "Deflate Forward") ?
        MultiscaleFluid::DeflateForwardFluid :
        MultiscaleFluid::DeflateReverseFluid;
      MultiscaleFluid::
        deflateImage(hField,
                     *image,
                     roi,
                     imageValMin->value(),
                     imageValMax->value(),
                     numScaleLevels,
                     params,
                     (unsigned int) numDilationsInput->value(),
                     (unsigned int) numErosionsInput->value(),
                     resliceButton->value(),
                     false,
                     deflationType);
      createOToIImageButton->value(0);
      createOToISurfacesButton->value(0);
      addIToOSurfaceButton->value(0);
    }
    else
    {
      fl_alert("Invalid fluid choice.");
      return;
    }
  }
  catch (...)
  {
    timer.stop();
    std::cout << "Failed after " << timer.getSeconds() << " sec." << std::endl;
    if (overlayNeeded)
    {
      fl_alert("Fluid registration failed.  Make sure images "
               "have same dimensions, spacing, and offset.");      
    }
    else
    {
      fl_alert("Fluid deflation failed.");
    }
    _updateStatusBuffer("Error: Fluid deformation failed.");
    return;
  }

  timer.stop();
  std::cout << "###> Fluid Registration:" << std::endl
            << "\tDimensions (x, y, z) = (" 
            << roi.getSize()[0] << ", " 
            << roi.getSize()[1] << ", " 
            << roi.getSize()[2] << ")" << std::endl
            << "\tTotal Time (sec): " << timer.getSeconds() << std::endl;
  _updateStatusBuffer("Fluid deformation ... done computing warp");  

  // save h field if one was chosen
  if( hFieldFilename != std::string("") ) 
  {
    _updateStatusBuffer("Saving hField...");
    try
    {
        HField3DIO::writeMETA(*hFieldPtr, hFieldOrigin, hFieldSpacing,
                              hFieldFilename);
        if (_deformationChoices[choice] == "Shrink")
        {
          HField3DIO::writeMETA(*hFieldInvPtr, hFieldOrigin, hFieldSpacing,
                                hFieldFilename + "_inv");
        }
    }
    catch (...)
    {
      fl_alert("Failed to write hField.");
      _updateStatusBuffer("Failed to write hField");
    }
    _updateStatusBuffer("DONE");
  }  

  _applyFluidWarp(fluidWarpInterface,
                  imageIndex,
                  overlayIndex,
                  createOToIImageButton->value(),
                  createOToISurfacesButton->value(),
                  addIToOSurfaceButton->value(),
                  imageName);

  overlayChoice->value((int) _loadedImages.size());

  // update everything
  _updateContours();
  _updateSurfaces();
  _redrawImageWindows();      
  //  displayBYUListCallback();
  _printParameters("FLUID_REGISTER_ROI_START", "FLUID_REGISTER_ROI_STOP",
                   "FLUID_REGISTER_INTENSITY_WINDOW");
}

///////////////////////////
// fluidShrinkCallback() //
///////////////////////////

void
ImMap
::fluidShrinkCallback()
{

  // We first check if the ROI is defined
  if (roiStartX->value()==0 && roiStopX->value()==0 &&
      roiStartY->value()==0 && roiStopY->value()==0 &&
      roiStartZ->value()==0 && roiStopZ->value()==0 &&
      useROIButton->value()==1)
  {
    fl_alert("You need to set the ROI.");
    return;
  }

  // get filename for saving hfield
  char *hFieldFileName = 
    fl_file_chooser("Select transformation field file...", "", 0);

  // make sure we can open file before we begin
  if( hFieldFileName ) 
  {
    std::string realfn = std::string(hFieldFileName) + std::string(".mhd");
    std::cout << "Making sure we can open " << realfn << std::endl;
    std::ofstream output(realfn.c_str());
    if (output.fail())
    {
      std::ostringstream oss;
      oss << "Can't open file for writing: " << hFieldFileName;
      fl_alert(oss.str().c_str());
      _updateStatusBuffer(oss.str().c_str());
      return;
    }
    output.close();
  }
  _updateStatusBuffer("Starting Fluid Shrink...");

  // make sure two images are loaded
  if (!axial2DWindow->haveValidImage()) 
  {
    _updateStatusBuffer("Error: Fluid Register: No Image Loaded");
    return;
  }

  // set up parameters
  int imageIndex = imageChoice->value();
  ImagePointer image = _loadedImages[imageIndex];
  std::string imageFileName = _imageFullFileNames[imageIndex];
  VoxelType imageRescaleMinThreshold = imageValMin->value();
  VoxelType imageRescaleMaxThreshold = imageValMax->value();

  unsigned int numScaleLevels = 3;
  FluidWarpParameters params[3];


  if (fluidFineButton->value() == 0)
  {
    params[0].numIterations=0;
  }
  else
  {
    params[0].numIterations = 
      static_cast<unsigned int>(fluidFineMaxIterations->value());
  }
  params[0].alpha=fluidFineAlpha->value();
  params[0].beta=fluidFineBeta->value();
  params[0].gamma=fluidFineGamma->value();
  params[0].maxPerturbation=fluidFineMaxPerturbation->value();
  params[0].numBasis=2000;
  params[0].jacobianScale = false;

  if (fluidMediumButton->value() == 0)
  {
    params[1].numIterations=0;
  }
  else
  {
    params[1].numIterations = 
      static_cast<unsigned int>(fluidMediumMaxIterations->value());
  }
  params[1].alpha=fluidMediumAlpha->value();
  params[1].beta=fluidMediumBeta->value();
  params[1].gamma=fluidMediumGamma->value();
  params[1].maxPerturbation=fluidMediumMaxPerturbation->value();
  params[1].numBasis=2000;
  params[1].jacobianScale = false;

  if (fluidCoarseButton->value() == 0)
  {
    params[2].numIterations=0;
  }
  else
  {
    params[2].numIterations = 
      static_cast<unsigned int>(fluidCoarseMaxIterations->value());
  }
  params[2].alpha=fluidCoarseAlpha->value();
  params[2].beta=fluidCoarseBeta->value();
  params[2].gamma=fluidCoarseGamma->value();
  params[2].maxPerturbation=fluidCoarseMaxPerturbation->value();
  params[2].numBasis=2000;
  params[2].jacobianScale = false;

  // set up roi
  ImageRegionType roi;
  int useROI = useROIButton->value();
  if (useROI)
  {
    roi = _getROI();
  }
  else
  {
    roi.setSize(image->getSize());
  }

  // set up fluid registration and run it
  _updateStatusBuffer("Fluid Shrink ... initializing");
  MultiScaleFluidWarp fluidWarpInterface;
  _updateStatusBuffer("Fluid Shrink ... running");  
  Timer timer;
  timer.start();
  try 
  {
    if (deformationChoice->value() == 4)
    {
      if(maskChoice->value()==0)
        {
          fl_alert("Please create or select a mask first.");
          return;
        }
    }
    fluidWarpInterface.shrink(image,
                              imageFileName.c_str(),
                              imageRescaleMinThreshold, 
                              imageRescaleMaxThreshold,
                              roi, 
                              numScaleLevels,
                              params,
                              resliceButton->value(),
                              deformationChoice->value(),
                              _mask[maskChoice->value()]);
  }
  catch (...)
  {
    fl_alert("Fluid registration failed.  Make sure images "
             "have same dimensions, spacing, and offset.");      
    _updateStatusBuffer("Error: Fluid registration failed.");
    return;
  }

  timer.stop();
  std::cerr << "###> Fluid Shrink:" << std::endl
            << "\tDimensions (x, y, z) = (" 
            << roi.getSize()[0] << ", " 
            << roi.getSize()[1] << ", " 
            << roi.getSize()[2] << ")" << std::endl
            << "\tTotal Time (sec): " << timer.getSeconds() << std::endl;
  _updateStatusBuffer("Fluid Shrink ... done computing warp");  

  // save h field if one was chosen
  if( hFieldFileName != (char*)0 ) 
  {
    _updateStatusBuffer("Saving hField...");
    try
    {
      fluidWarpInterface.writeHFieldMETA(hFieldFileName);
    }
    catch (...)
    {
      fl_alert("Failed to write hField.");
      _updateStatusBuffer("Failed to write hField");
    }
    _updateStatusBuffer("DONE");
  }  

  std::string imageName("shrunken image");

  try
  {
    // apply to atlas and add to loaded image list
    _updateStatusBuffer("Warping image ...");
    ImagePointer warpedImage =
      fluidWarpInterface.apply(_loadedImages[imageIndex]);

    //add an empty ImAna vector to the list
    _imAnaVectors.push_back(std::vector<ImAna>());

    _imageNames.push_back(imageName);
    _imageFullFileNames.push_back(imageName);
    warpedImage->setDataType( Image<float>::Float );
    warpedImage->setOrientation(_loadedImages[imageIndex]->getOrientation());
    _addImageToList(warpedImage);
    _updateStatusBuffer("Warping image ... DONE");
  }
  catch(...)
  {
    fl_alert("Failed to warp image.");
    _updateStatusBuffer("Failed to warp image.");
    return;
  }


  // update everything
  _updateContours();
  _updateSurfaces();
  _redrawImageWindows();      
  displayBYUListCallback();
}

void 
ImMap
::fluidDeflateCallback()
{
  _updateStatusBuffer("Fluid Deflate: initializing...");

  // check that roi is set if using roi
  if (_usingROI() && !_roiIsSet())
    {
      fl_alert("ROI is not set.");
      return;
    }

  // get name for deflation hfield s, make sure file is writable
  char* sFieldFilename = 
    fl_file_chooser("Select name for deflation hField file.","",0);
  if (sFieldFilename && !_testFileCanWrite(sFieldFilename))
    {
      std::ostringstream oss;
      oss << "Can't open file for writing: " << sFieldFilename;
      fl_alert(oss.str().c_str());
      _updateStatusBuffer(oss.str().c_str());
      return;      
    }

  // make sure a valid image is loaded
  if (!_imageIsLoaded())
    {
      _updateStatusBuffer("Error: Fluid Deflate: No image loaded.");
      return;
    }

  // get parameters
  int imageIndex = _getCurrentImageIndex();
  ImagePointer imagePtr = _loadedImages[imageIndex];
  unsigned int numScaleLevels = 3;
  FluidWarpParameters* params = new FluidWarpParameters[numScaleLevels];
  _getFluidParameters(params, numScaleLevels);

  // set up region of interest
  ImageRegionType roi;
  if (_usingROI())
    {
      roi = _getROI();
    }
  else
    {
      roi.setSize(imagePtr->getSize());
    }

  // continue setting up parameters
  MultiscaleFluid::writeDebugImages(false);
  MultiscaleFluid::outputFluidVerbose(true);
  MultiscaleFluid::doFFTWMeasure(true);
  MultiscaleFluid::AlgorithmType deflationType;
  switch (deformationChoice->value())
    {
    case (6):
      deflationType = MultiscaleFluid::DeflateForwardFluid;
      break;
    case (5):
    default:
      deflationType = MultiscaleFluid::DeflateReverseFluid;
      break;
    }
  
  // run deflation algorithm
  _updateStatusBuffer("Fluid Deflate: running...");
  HFieldType sField;
  Vector3D<double> sFieldOrigin = roi.getStart();
  imagePtr->imageIndexToWorldCoordinates(sFieldOrigin, sFieldOrigin);
  Vector3D<double> sFieldSpacing = imagePtr->getSpacing();
  try
    {
      MultiscaleFluid::
        deflateImage(sField,
                     *imagePtr,
                     roi,
                     imageValMin->value(),
                     imageValMax->value(),
                     numScaleLevels,
                     params,
                     (unsigned int) numDilationsInput->value(),
                     (unsigned int) numErosionsInput->value(),
                     resliceButton->value(),
                     false,
                     deflationType);

      // clean up memory
      delete [] params;
    }
  catch (...)
    {
      std::string msg = "Fluid Deflate: algorithm failed to finish.";
      fl_alert(msg.c_str());
      _updateStatusBuffer(msg.c_str());

      // clean up memory
      delete [] params;

      return;
    }
  _updateStatusBuffer("Fluid Deflate: running...done");  

  // write sField
  if (sFieldFilename != (char*) 0)
    {
      _updateStatusBuffer("Fluid Deflate: writing sField...");
      try 
        {
          HField3DIO::writeMETA(sField, 
                                sFieldOrigin, 
                                sFieldSpacing, 
                                sFieldFilename);
        }
      catch (...)
        {
          std::string msg = "Fluid Deflate: error writing sField.";
          fl_alert(msg.c_str());
          _updateStatusBuffer(msg.c_str());
        }
      _updateStatusBuffer("Fluid Deflate: writing sField...done");
    }

  // add new image to list 
  _updateStatusBuffer("Fluid Deflate: adding image to list...");
  try {
    std::string newImageName;
    ImagePointer deflatedImage = new ImageType(*imagePtr);
    switch (deformationChoice->value())
      {
      case (6):
        newImageName = "Deflated Forward";
        HField3DUtils::forwardApply(*imagePtr, sField, *deflatedImage,
                                    roi.getStart().x, 
                                    roi.getStart().y,
                                    roi.getStart().z);
        break;
      case (5):
      default:
        newImageName = "Deflated Reverse";
        HField3DUtils::apply(*imagePtr, sField, *deflatedImage,
                             roi.getStart().x, 
                             roi.getStart().y,
                             roi.getStart().z);
        break;
      }
    
    _imAnaVectors.push_back(std::vector<ImAna>());
    _imageNames.push_back(newImageName);
    _addImageToList(deflatedImage);
    _updateStatusBuffer("Fluid Deflate: adding image to list...done");
  }
  catch (...) {
    fl_alert("Failed to add image to list!");
    _updateStatusBuffer("Fluid Deflate: adding image to list...failed");    
  }

  _updateContours();
  _updateSurfaces();
  _redrawImageWindows();      
  displayBYUListCallback();
}

//////////////////////////////////////
// applyTranslationVectorCallback() //
//////////////////////////////////////

void 
ImMap
::applyTranslationVectorCallback()
{
  // update status bar
  _updateStatusBuffer("Starting Translate By Vector...");

  // make sure two images are loaded
  if (!axial2DWindow->haveValidImage()) 
  {
    _updateStatusBuffer("Error: Translate By Vector: No Image Loaded");
    return;
  }
  if (!axial2DWindow->haveValidOverlay()) 
  {
    _updateStatusBuffer("Error: Translate By Vector: No Overlay Loaded");
    return;
  }

  // set up parameters
  int imageIndex = imageChoice->value();
  int overlayIndex = overlayChoice->value();

  double tx = translateVectorX->value();
  double ty = translateVectorY->value();
  double tz = translateVectorZ->value();

  std::string imageName("overlay translated by vector");
  _applyTranslation(tx, ty, tz, imageIndex, overlayIndex, imageName);
}

///////////////////////////
// applyMatrixCallback() //
///////////////////////////

void 
ImMap
::applyMatrixCallback()
{

  char* matrixFileName = fl_file_chooser("Choose matrix file...","","", 0);
  if( matrixFileName == (char*)0 ) 
  {
    std::cout << "[ImMap::applyMatrixCallback] file chooser failed to get file";
    std::cout << std::endl;
    return;
  }

  AffineTransform3D<double> transform;
  try 
  {
    transform.readPLUNCStyle(matrixFileName);
  }
  catch (...)
  {
    std::cout<<"Failed to read matrix"<<std::endl;
    return;
  }

  // set up parameters
  int imageIndex = imageChoice->value();
  int overlayIndex = overlayChoice->value();

  std::string imageName("overlay transformed by matrix");
  _applyAffineTransform(transform, imageIndex, overlayIndex, imageName);

}

//////////////////////////
// saveMatrixCallback() //
//////////////////////////

void 
ImMap
::saveMatrixCallback()
{

  char* matrixFileName = fl_file_chooser("Choose matrix file...","","", 0);

  if( matrixFileName == (char*)0 ) 
  {
    std::cout << "[ImMap::saveMatrixCallback] file chooser failed to get file";
    std::cout << std::endl;
    return;
  }
  try
  {
    registrationTransform.writePLUNCStyle(matrixFileName);
  }
  catch (...)
  {
    std::cout<<"Failed to save matrix"<<std::endl;
    return;
  }

}

//////////////////////////////
// alignCentroidsCallback() //
//////////////////////////////

void
ImMap
::alignCentroidsCallback()
{
  // update status bar
  _updateStatusBuffer("Starting Align Centroids...");

  // make sure two images are loaded
  if (!axial2DWindow->haveValidImage()) 
  {
    _updateStatusBuffer("Error: Align Centroids: No Image Loaded");
    return;
  }
  if (!axial2DWindow->haveValidOverlay()) 
  {
    _updateStatusBuffer("Error: Align Centroids: No Overlay Loaded");
    return;
  }

  // set up parameters
  int imageIndex = imageChoice->value();
  int overlayIndex = overlayChoice->value();
  ImagePointer atlas = _loadedImages[imageIndex];
  ImagePointer subject = _loadedImages[overlayIndex];

  // find centroids
  _updateStatusBuffer("Computing centroids ...");
  Vector3D<double> atlasCentroidVec = ImageUtils::computeCentroid(*atlas);
  Vector3D<double> subjectCentroidVec = ImageUtils::computeCentroid(*subject);
  float atlasCentroid[3];
  float subjectCentroid[3];
  for (unsigned int i = 0; i < 3; ++i) {
      atlasCentroid[i] = atlasCentroidVec[i];
      subjectCentroid[i] = subjectCentroidVec[i];
  }
  _updateStatusBuffer("Computing centroids ... DONE");

  double tx = subjectCentroid[0] - atlasCentroid[0];
  double ty = subjectCentroid[1] - atlasCentroid[1];
  double tz = subjectCentroid[2] - atlasCentroid[2];

  registrationTransform.eye();
  registrationTransform.vector = Vector3D<double>(tx, ty, tz);
  _updateTransformDisplay();

  // apply centroid difference
  std::string imageName("overlay translated by centroid difference");
  _applyTranslation(tx, ty, tz,
    imageIndex,
    overlayIndex,
    imageName);
}

///////////////////////////
// applyHFieldCallback() //
///////////////////////////

void
ImMap
::applyHFieldCallback()
{
  // update status bar
  _updateStatusBuffer("Applying h Field...");

  int createOToIImage    = createOToIImageButton->value();
  int createOToISurfaces = createOToISurfacesButton->value();
  int addIToOSurfaces    = addIToOSurfaceButton->value();

  if (createOToIImage && !axial2DWindow->haveValidOverlay())
  {
    fl_alert("Can't warp overlay image, no overlay loaded.");
    _updateStatusBuffer("Error: Apply h Field: No Overlay Loaded");
    createOToIImage = false;
  }
  if (addIToOSurfaces && !axial2DWindow->haveValidImage())
  {
    fl_alert("Can't warp image surfaces, no image loaded.");
    _updateStatusBuffer("Error: Apply h Field: No Image Loaded");
    addIToOSurfaces = false;      
  }
  if (addIToOSurfaces && !axial2DWindow->haveValidOverlay())
  {
    fl_alert("Can't warp image surfaces, no overlay loaded.");
    _updateStatusBuffer("Error: Apply h Field: No Overlay Loaded");
    addIToOSurfaces = false;      
  }

  // we should only create surfaces this direction if we are creating
  // an image
  if (createOToISurfaces)
  {
    createOToISurfaces = createOToIImage;
  }

  if (!(createOToIImage || addIToOSurfaces))
  {
    fl_alert("Nothing to do.");
    _updateStatusBuffer("Apply h Field ... DONE");
    return;
  }

  // get hfield filename
  char *hFieldFileName = 
    fl_file_chooser("Select transformation field file...", "", 0);
  if( hFieldFileName == 0 )
  {
    std::cerr << "[ImMap:applyHFieldCallback] no hfield selected"
      << std::endl;
    return;
  }

  string extension=hFieldFileName;
  int pos = extension.find_last_of(".");
  extension.erase(extension.begin(),extension.begin() + pos + 1 );
  bool fileIsMETA = (extension.compare("mhd")==0);


  if (fileIsMETA && !axial2DWindow->haveValidImage())
  {
    fl_alert("Can't apply META hfield -- no image loaded.");
    _updateStatusBuffer("Error: Apply h Field: No image loaded");
    return;
  }

  if (fileIsMETA && !axial2DWindow->haveValidOverlay())
  {
    fl_alert("Can't apply META hfield -- no overlay loaded.");
    _updateStatusBuffer("Error: Apply h Field: No overlay Loaded");
    return;
  }

  // load hfield
  MultiScaleFluidWarp fluidWarpInterface;

  // need image and overlay in case we're using META
  int imageIndex = imageChoice->value();
  int overlayIndex = overlayChoice->value();

  try 
  {
    if (fileIsMETA) {
      ImageRegionType roi = 
        fluidWarpInterface.loadHFieldMETA(hFieldFileName,
                                          _loadedImages[imageIndex],
                                          _loadedImages[overlayIndex]);
      surface3DWindow->createROI();
      _setROI(roi.getStartX(), roi.getStartY(), roi.getStartZ(), 
              roi.getStopX(), roi.getStopY(), roi.getStopZ());
      ROIvisibleCheckButton->value(true);
      _ROIcreated = true;
      roiPropertyChangedCallback(); 
    } else {
      fluidWarpInterface.loadHField(hFieldFileName);
    }
  }
  catch (...)
  {
    fl_alert("Apply h Field: error loading h field");      
    _updateStatusBuffer("Apply h Field: error loading h field");      
    return;
  }

  std::string fileName("overlay fluid warped by h field");
  _applyFluidWarp(fluidWarpInterface,
    imageIndex,
    overlayIndex,
    createOToIImage,
    createOToISurfaces,
    addIToOSurfaces,
    fileName);    

  _updateContours();
  _updateSurfaces();
  _redrawImageWindows();      
  displayBYUListCallback();
}

////////////////////////
// resampleCallback() //
////////////////////////

void
ImMap
::resampleCallback()
{
  // update status bar
  _updateStatusBuffer("Resampling...");

  // make sure two images are loaded
  if (!axial2DWindow->haveValidImage()) 
  {
    _updateStatusBuffer("Error: Resampling: No Image Loaded");
    return;
  }
  if (!axial2DWindow->haveValidOverlay()) 
  {
    _updateStatusBuffer("Error: Resampling: No Overlay Loaded");
    return;
  }

  // set up parameters
  int imageIndex = imageChoice->value();
  int overlayIndex = overlayChoice->value();

  registrationTransform.eye();
  _updateTransformDisplay();

  std::string imageName("overlay resampled into image");
  _applyTranslation(0,
    0,
    0,
    imageIndex,
    overlayIndex,
    imageName);
}

///////////////////////////////
// anastructChoiceCallback() //
///////////////////////////////

void
ImMap
::anastructChoiceCallback()
{

}

/////////////////////////////////////////
// anastructPropertyChangedCallback() //
/////////////////////////////////////////

void
ImMap
::anastructPropertyChangedCallback()
{

}


////////////////////////
// createMaskCallback //
////////////////////////

void
ImMap
::createMaskCallback()
{
  int imageIndex = imageChoice->value();
  if (imageIndex == 0)
  {
    fl_alert("Please select an image first!");
    return;
  }
  ImagePointer atlas = _loadedImages[imageIndex];
  ImageSizeType size(atlas->getSize());
  MaskType newMask(size);

  for(unsigned int z = 0 ; z < size.z ; z++){
    for(unsigned int y = 0 ; y < size.y ; y++){
      for(unsigned int x = 0 ; x < size.x ; x++){
        if(atlas->get(x,y,z) > imageValMin->value())
        {
          newMask.set(x,y,z,1.0);
        }
        else
        {
          newMask.set(x,y,z,0.0);
        }
      }
    }
  }
  static unsigned int maskIndex;
  char label[100];//add the imageIndex to the label
  sprintf(label,"%s  (%d)",imagePresetIntensityChoice->text(imagePresetIntensityChoice->value()),maskIndex);
  maskChoice->add(label);
  _mask.push_back(newMask);
  maskIndex++;

}

// To set all the histograms to the values saved in the histogram class.
// When you make appear an image previously loaded, the histograms came 
// back to their last positions for this very image.

void
ImMap
::histogramImageCallback()
{
  int histoIndex = imageChoice->value();

  if (histoIndex!=0) {  

    float minRel = _histograms[histoIndex].getRelativeMin();
    float maxRel = _histograms[histoIndex].getRelativeMax();
    float minAbs = _histograms[histoIndex].getAbsoluteMin();
    float maxAbs = _histograms[histoIndex].getAbsoluteMax();

   std::cout<<"min rel: "<<minRel<<"  Max Rel: "<<maxRel<<"  minAbs: "<<minAbs<<"  maxAbs: "<<maxAbs<<std::endl;

    // Set the histogram Window  
    HistogramDisplay->setWindow(_histograms[histoIndex].getHistogram(),
                                minRel, maxRel, minAbs, maxAbs); 

    // Set the control panel

    imageValMin->value(minRel);
    imageValMax->value(maxRel);

    _presetIntensity[0].relativeMin = minAbs;
    _presetIntensity[0].relativeMax = maxAbs;

    HistogramDisplay->redraw();

    // Set the histogram slider of the main Window

    updateHistogramSliderCallback();

  } else {
    _updateStatusBuffer("No image selected");
  }
}


// We update the display (histogram window), the histogram class,
// the control panel and the histogram slider


//////////////////////////////////
// transmitHistoImageCallback() //
//////////////////////////////////

void 
ImMap
:: transmitHistoImageCallback(){

  // if the relative value of the min to be transmitted is inferior
  // to the absolute min of the other histogram (overlay), we set the
  // new relative min value of the overlay histogram to its min.

  if ((imageValMin->value())<=(overlayValMin->minimum())){
    overlayValMin->value(overlayValMin->minimum());
  }
  else{
    overlayValMin->value(imageValMin->value());
  }
  overlayValMin->do_callback();

  // if the relative value of the max to be transmitted is greater than
  // the absolute max of the other histogram (overlay), we set the
  // new relative max value of the overlay histogram to its max.

  if (imageValMax->value()>=(overlayValMax->maximum())){
    overlayValMax->value(overlayValMax->maximum());
  }
  else {
    overlayValMax->value(imageValMax->value());
  }
  overlayValMax->do_callback();
}



// Callback to open the overlay histogram Window

////////////////////////////////
// histogramOverlayCallback() //
////////////////////////////////

void
ImMap
::histogramOverlayCallback(){

  int histoIndex = overlayChoice->value();

  if (histoIndex!=0){  

    float minRel = _histograms[histoIndex].getRelativeMin();
    float maxRel = _histograms[histoIndex].getRelativeMax();
    float minAbs = _histograms[histoIndex].getAbsoluteMin();
    float maxAbs = _histograms[histoIndex].getAbsoluteMax();

    HistogramOverlayDisplay->setWindow(_histograms[histoIndex].getHistogram(),
      minRel,maxRel,minAbs,maxAbs);

 
    imageValMin->value(minRel);
    overlayValMax->value(maxRel);

    _presetIntensity[0].relativeMin = minAbs;
    _presetIntensity[0].relativeMax = maxAbs;


    HistogramOverlayDisplay->redraw();

  }

}

////////////////////////////////////
// transmitHistoOverlayCallback() //
////////////////////////////////////

void 
ImMap
:: transmitHistoOverlayCallback(){

  // if the relative value of the min to be transmitted is inferior
  // to the absolute min of the other histogram (image), we set the
  // new relative min value of the image histogram to its min.

  if ((overlayValMin->value())<=(imageValMin->minimum())){
    std::cout << "Hitting this case 1" << std::endl;
    imageValMin->value(imageValMin->minimum());
  }
  else{
    imageValMin->value(overlayValMin->value());
  }

  // if the relative value of the max to be transmitted is greater than
  // the absolute max of the other histogram (image), we set the
  // new relative max value of the image histogram to its max.

  if (overlayValMax->value()>=(imageValMax->maximum())){
    imageValMax->value(imageValMax->maximum());
  }
  else {
    imageValMax->value(overlayValMax->value());
  }
  imageValMin->do_callback();
  imageValMax->do_callback();
}

/////////////////////////
// infoImageCallback() //
/////////////////////////

void
ImMap
:: infoImageCallback(){

  int  indexImage = imageChoice->value();
  char temp[1024];

  if (indexImage==0){
    _updateStatusBuffer("No image selected");
  }
  else{

    // Dimensions

    ImageSizeType Dimensions(_loadedImages[indexImage]->getSize());;
    // std::cout << "Dimensions = " << Dimensions[0] << " " << Dimensions[1]
    //          << " " << Dimensions[2] << std::endl;
    sprintf(temp,"%d x %d x %d ",(int)Dimensions[0],(int)Dimensions[1],
            (int)Dimensions[2]);
    DimensionsInfoImage->value(temp);

    // PixelSize cm/voxel

    ImageSizeType PixelSize = _loadedImages[indexImage]->getSpacing();
    // std::cout<<"PixelSize = "<<PixelSize[0]<<" "<<PixelSize[1]<<" "<<PixelSize[2]<< std::endl;
    sprintf(temp,"%.3f x %.3f x %.3f",PixelSize[0],PixelSize[1],PixelSize[2]);
    PixelSizeInfoImage->value(temp);

    // Orientation

    OrientationInfoImage->value(_loadedImages[indexImage]->getOrientationStr().c_str());

    // Origin cm/voxel

    ImageSizeType Origin = _loadedImages[indexImage]->getOrigin();
    // std::cout<<"Origin = "<<Origin[0]<<" "<<Origin[1]<<" "<<Origin[2]<<std::endl;
    sprintf(temp,"%.3f  %.3f  %.3f",Origin[0],Origin[1],Origin[2]);
    OriginInfoImage->value(temp);

    XOrigin->value(Origin[0]);
    YOrigin->value(Origin[1]);
    ZOrigin->value(Origin[2]);


    // Min/Max

    sprintf(temp,"%.1f  / %.1f",_histograms[indexImage].getAbsoluteMin(),
      _histograms[indexImage].getAbsoluteMax());
    MinMaxInfoImage->value(temp);

    // DataType

    DataTypeInfoImage->value(_loadedImages[indexImage]->getDataTypeStr().c_str());   

  }

}

/////////////////////////
// infoOverlayCallback() //
/////////////////////////

void
ImMap
:: infoOverlayCallback(){

  int  indexImage = overlayChoice->value();
  char temp[1024];

  if (indexImage==0){
    _updateStatusBuffer("No overlay selected");
  }
  else{

    // Dimensions

    ImageSizeType Dimensions(_loadedImages[indexImage]->getSize());;
    std::cout<<"Dimensions = "<<Dimensions[0]<<" "<<Dimensions[1]<<" "<<Dimensions[2]<<std::endl;
    sprintf(temp,"%d x %d x %d",(int)Dimensions[0],(int)Dimensions[1],(int)Dimensions[2]);
    DimensionsInfoOverlay->value(temp);

    // PixelSize

    ImageSizeType PixelSize = _loadedImages[indexImage]->getSpacing();
    std::cout<<"PixelSize = "<<PixelSize[0]<<" "<<PixelSize[1]<<" "<<PixelSize[2]<<std::endl;
    sprintf(temp,"%.3f x %.3f x %.3f",PixelSize[0],PixelSize[1],PixelSize[2]);
    PixelSizeInfoOverlay->value(temp);

    // Orientation

    OrientationInfoOverlay->value(_loadedImages[indexImage]->getOrientationStr().c_str());

    // Origin

    ImageSizeType Origin = _loadedImages[indexImage]->getOrigin();
    std::cout<<"Origin = "<<Origin[0]<<" "<<Origin[1]<<" "<<Origin[2]<<std::endl;
    sprintf(temp,"%.3f  %.3f  %.3f",Origin[0],Origin[1],Origin[2]);
    OriginInfoOverlay->value(temp);

    XOrigin->value(Origin[0]);
    YOrigin->value(Origin[1]);
    ZOrigin->value(Origin[2]);


    // Min/Max

    sprintf(temp,"%.1f  / %.1f",_histograms[indexImage].getAbsoluteMin(),
      _histograms[indexImage].getAbsoluteMax());
    MinMaxInfoOverlay->value(temp);

    // DataType

    DataTypeInfoOverlay->value(_loadedImages[indexImage]->getDataTypeStr().c_str());   

  }

}

/////////////////////
// DICOMCallback() //
/////////////////////

void 
ImMap
::DICOMCallback(){

  int selected_DICOM;
  selected_DICOM=choiceDICOM->value();

  DICOMimage DICOM = dicomWindow->getDicom();

  DICOM.SelectImage(selected_DICOM);

  char temp[1024];
  sprintf(temp,"   %d*%d*%d",DICOM.getsizeX(),
    DICOM.getsizeY(),DICOM.getsizeZ());
  dimensionDICOM->value((const char*) temp);
  sprintf(temp,"%.3f*%.3f*%.3f mm/voxel",DICOM.PixsizeX(),
    DICOM.PixsizeY(),DICOM.PixsizeZ());
  resolutionDICOM->value((const char*) temp);

  dicomWindow->redraw();

}


// General Callback for the histogram slider of the main Window
// Update all the other histogram

///////////////////////////////
// histogramSliderCallback() //
///////////////////////////////

void
ImMap
::histogramSliderCallback(){

  int imageHisto = imageChoice->value();
  int overlayHisto = overlayChoice->value();

  // The histogram goes from 0 to 450

  int mini, maxi;
  mini =   histogramSlider->GetMin();
  maxi =   histogramSlider->GetMax();

  float minHisto,maxHisto;
  float newMin,newMax;

  if (_histogramImage){

    minHisto = imageValMin->minimum();
    maxHisto = imageValMin->maximum();

    newMin = minHisto + (mini ) * (maxHisto - minHisto)/double(HISTOGRAM_HEIGHT-1.0);
    newMax = minHisto + (maxi ) * (maxHisto - minHisto)/double(HISTOGRAM_HEIGHT-1.0);

    // Update Control Panel
    imageValMin->value(newMin);
    imageValMax->value(newMax);

    // Update the display of the histogram and the histogram class for this image

    HistogramDisplay->updateRelativeMin(imageValMin->value());
    _histograms[imageHisto].setRelativeMin(imageValMin->value());

    HistogramDisplay->updateRelativeMax(imageValMax->value());
    _histograms[imageHisto].setRelativeMax(imageValMax->value());

    // To redraw the image
    imageIWChangedCallback();

    if (_histogramLinked){
      transmitHistoImageCallback();
    }

  }
  else {

    minHisto = overlayValMin->minimum();
    maxHisto = overlayValMin->maximum();

    newMin = minHisto + (mini) * (maxHisto - minHisto)/double(HISTOGRAM_HEIGHT-1.0);
    newMax = minHisto + (maxi) * (maxHisto - minHisto)/double(HISTOGRAM_HEIGHT-1.0);

    // Update Control Panel
    overlayValMin->value(newMin);
    overlayValMax->value(newMax);

    // Update Overlay Histogram Window
    imageValMin->value(newMin);
    overlayValMax->value(newMax);

    // Update the display of the histogram and the histogram class for this image 
    //(overlay)

    HistogramOverlayDisplay->updateRelativeMin(imageValMin->value());
    _histograms[overlayHisto].setRelativeMin(imageValMin->value());

    HistogramOverlayDisplay->updateRelativeMax(overlayValMax->value());
    _histograms[overlayHisto].setRelativeMax(overlayValMax->value());

    // To redraw the overlay
    overlayIWChangedCallback();

    if(_histogramLinked){
      transmitHistoOverlayCallback();
    }

  }
}

// Update the slider histogram of the main window

/////////////////////////////////////
// updateHistogramSliderCallback() //
/////////////////////////////////////

void 
ImMap
:: updateHistogramSliderCallback(){

  float mini_val, maxi_val;
  float minHisto,maxHisto;


  if (_histogramImage){

    mini_val =  imageValMin->value();
    maxi_val =  imageValMax->value();

    minHisto = imageValMin->minimum();
    maxHisto = imageValMin->maximum();
  }
  else{
    mini_val =  overlayValMin->value();
    maxi_val =  overlayValMax->value();

    minHisto = overlayValMin->minimum();
    maxHisto = overlayValMin->maximum();
  }


  double minForSlider,maxForSlider;

  minForSlider =double(double(HISTOGRAM_HEIGHT-1.0)*(mini_val-minHisto)/(maxHisto-minHisto));
  maxForSlider =double(double(HISTOGRAM_HEIGHT-1.0)*(maxi_val-minHisto)/(maxHisto-minHisto));
/*
  if(maxi_val<1)
  {
	mini_val*=100;
	maxi_val*=100;
	minHisto*=100;
	maxHisto*=100;
  }
*/
//  std::cout<<"mini_val : "<<mini_val<<"  maxi_val : "<<maxi_val<<"  minHisto : "<<minHisto<<"  maxHisto : "<<maxHisto<<"  minForSlider : "<<minForSlider<<"  maxForSlider : "<<maxForSlider<<std::endl;

  histogramSlider->SetMin(minForSlider);
  histogramSlider->SetMax(maxForSlider);
  histogramSlider->redraw();
}

// Callback for the image histogram slider of the control panel
// Update all the other histograms

//////////////////////////////
// imageValMinMaxCallback() //
//////////////////////////////

void 
ImMap
::imageValMinMaxCallback(){

  int imageHisto = imageChoice->value();

  if (imageValMin->value() < _histograms[imageHisto].getAbsoluteMin() )
  {
    imageValMin->value(_histograms[imageHisto].getAbsoluteMin());
  }

  // For the Min
  /////////////////
  // Update Image Histogram Window 

  HistogramDisplay->updateRelativeMin(imageValMin->value());
  _histograms[imageHisto].setRelativeMin(imageValMin->value());

  // For the Max
  /////////////////

  // Update Image Histogram Window

  if (imageValMax->value() > _histograms[imageHisto].getAbsoluteMax() )
  {
    imageValMax->value(_histograms[imageHisto].getAbsoluteMax());
  }

  HistogramDisplay->updateRelativeMax(imageValMax->value());
  _histograms[imageHisto].setRelativeMax(imageValMax->value());

  // Update the histogram Slider on the main window

  updateHistogramSliderCallback();

  // To redraw the image
  // Update the control panel and do the callback (imagePropertyChangedCallback)
  // to redisplay the image
  imageIWChangedCallback();

  //std::cout<<"imageValMin->value() : "<<imageValMin->value()<<" _histograms[imageHisto].getAbsoluteMin() : "<<_histograms[imageHisto].getAbsoluteMin()<<" imageValMax->value() : "<<imageValMax->value()<<" _histograms[imageHisto].getAbsoluteMax() : "<<_histograms[imageHisto].getAbsoluteMax()<<"   : "<<std::endl;

  if ((lockedIntensity->value() == 1) && (overlayChoice->value()!=0))
  {
    overlayValMin->value(imageValMin->value());
    overlayValMax->value(imageValMax->value());
    overlayValMinMaxCallback();
  }
}

//////////////////////////////
//  presetIntensityCallback //
//////////////////////////////
void 
ImMap
:: presetIntensityCallback(int imageIntensity)
{
  std::cout << "presetIntensityC" << std::endl;
  float mini, maxi;
  int IntensityChoice;
  if(imageIntensity == 1)//image
  {
    IntensityChoice = imagePresetIntensityChoice->value();
    mini = _presetIntensity[IntensityChoice].relativeMin;
    maxi = _presetIntensity[IntensityChoice].relativeMax;
    imageValMin->value(mini);
    imageValMax->value(maxi);
    imageValMinMaxCallback();

    if ((lockedIntensity->value() == 1))
    {
      overlayPresetIntensityChoice->value(IntensityChoice);
    }
  }
  else//overlay
  {
    IntensityChoice = overlayPresetIntensityChoice->value();
    mini = _presetIntensity[IntensityChoice].relativeMin;
    maxi = _presetIntensity[IntensityChoice].relativeMax;
    overlayValMin->value(mini);
    overlayValMax->value(maxi);
    overlayValMinMaxCallback();
  }
}

bool
ImMap
::_usingROI() const
{
  return useROIButton->value() == 1;
}

bool
ImMap
::_roiIsSet() const
{
  return !(roiStartX->value()==0 && roiStopX->value()==0 &&
           roiStartY->value()==0 && roiStopY->value()==0 &&
           roiStartZ->value()==0 && roiStopZ->value()==0);
}

bool
ImMap
::_testFileCanWrite(const char* filename)
{
  std::ofstream output(filename);
  bool canWrite = !output.fail();
  output.close();
  return canWrite;
}

bool
ImMap
::_imageIsLoaded() const
{
  return axial2DWindow->haveValidImage();
}

bool 
ImMap
::_overlayIsLoaded() const
{
  return axial2DWindow->haveValidOverlay();
}

int 
ImMap 
::_getCurrentImageIndex() const
{
  return imageChoice->value();
}

int
ImMap
::_getCurrentOverlayIndex() const
{
  return overlayChoice->value();
}

void
ImMap
::_getFluidParameters(FluidWarpParameters* params,
                      unsigned int& numScaleLevels) const
{

  // How many scale levels are there?
  if (fluidMediumButton->value() == 0 && fluidCoarseButton->value() == 0) 
  {
    numScaleLevels = 1;
  }
  else if (fluidCoarseButton->value() == 0) 
  {
    numScaleLevels = 2;
  }
  else
  {
    numScaleLevels = 3;
  }

  // Now load the information
  if (fluidFineButton->value() == 0)
    {
      params[0].numIterations=0;
    }
  else
    {
      params[0].numIterations = 
        static_cast<unsigned int>(fluidFineMaxIterations->value());
    }
  params[0].alpha=fluidFineAlpha->value();
  params[0].beta=fluidFineBeta->value();
  params[0].gamma=fluidFineGamma->value();
  params[0].maxPerturbation=fluidFineMaxPerturbation->value();
  params[0].numBasis=2000;
  params[0].jacobianScale = false;
  
  if (fluidMediumButton->value() == 0)
    {
      params[1].numIterations=0;
    }
  else
    {
      params[1].numIterations = 
        static_cast<unsigned int>(fluidMediumMaxIterations->value());
    }
  params[1].alpha=fluidMediumAlpha->value();
  params[1].beta=fluidMediumBeta->value();
  params[1].gamma=fluidMediumGamma->value();
  params[1].maxPerturbation=fluidMediumMaxPerturbation->value();
  params[1].numBasis=2000;
  params[1].jacobianScale = false;
  
  if (fluidCoarseButton->value() == 0)
    {
      params[2].numIterations=0;
    }
  else
    {
      params[2].numIterations = 
        static_cast<unsigned int>(fluidCoarseMaxIterations->value());
    }
  params[2].alpha=fluidCoarseAlpha->value();
  params[2].beta=fluidCoarseBeta->value();
  params[2].gamma=fluidCoarseGamma->value();
  params[2].maxPerturbation=fluidCoarseMaxPerturbation->value();
  params[2].numBasis=2000;
  params[2].jacobianScale = false;  
}

/////////////////
// _parseValue //
/////////////////

std::string
ImMap
::_parseValue(const std::string& str)
{
  //
  // return info between first and last quotes
  //
  size_t firstIndex = str.find_first_of('\"');
  size_t lastIndex  = str.find_last_of('\"');

  if ((firstIndex == std::string::npos) || 
    (lastIndex  == std::string::npos) ||
    (firstIndex == lastIndex))
  {
    throw std::invalid_argument("str does not contain a quote pair");
  }

  std::string result = str.substr(firstIndex + 1, lastIndex - firstIndex - 1);
  // std::cerr << "parse result" << result << std::endl;
  return result;
}

///////////////////////
//  _createPresetFile //
///////////////////////
void 
ImMap
:: _createPresetFile()
{
  char* presetFilename = (char*)_presetFilename.c_str();
  // first write ascii header
  std::ofstream outputASCII(presetFilename);

  if (outputASCII.fail())
  {
    throw std::runtime_error("failed to open file for ascii write");
  }

  outputASCII << "PRESET_ORGAN_INTENSITY_NAME=\""<<"Bone"<< "\"\n";

  outputASCII << "RELATIVE_MIN=\"" << 1150 << "\"\n";
  outputASCII << "RELATIVE_MAX=\"" << 1360 << "\"\n";

  outputASCII << "\n";

  outputASCII << "PRESET_ORGAN_INTENSITY_NAME=\""<<"BoneMVCT"<< "\"\n";

  outputASCII << "RELATIVE_MIN=\"" << 1140 << "\"\n";
  outputASCII << "RELATIVE_MAX=\"" << 1160 << "\"\n";

  outputASCII << "\n";

  outputASCII << "PRESET_ORGAN_INTENSITY_NAME=\""<<"Prostate"<< "\"\n";

  outputASCII << "RELATIVE_MIN=\"" << 900 << "\"\n";
  outputASCII << "RELATIVE_MAX=\"" << 1100 << "\"\n";

  outputASCII << "\n";

  outputASCII << "PRESET_ORGAN_INTENSITY_NAME=\""<<"Prostate2"<< "\"\n";

  outputASCII << "RELATIVE_MIN=\"" << 800 << "\"\n";
  outputASCII << "RELATIVE_MAX=\"" << 1200 << "\"\n";

  outputASCII << "\n";

  outputASCII << "PRESET_ORGAN_INTENSITY_NAME=\""<<"Prostate2MV"<< "\"\n";

  outputASCII << "RELATIVE_MIN=\"" << 855 << "\"\n";
  outputASCII << "RELATIVE_MAX=\"" << 1160 << "\"\n";

  outputASCII << "\n";

  outputASCII << "PRESET_ORGAN_INTENSITY_NAME=\""<<"Prostate3"<< "\"\n";

  outputASCII << "RELATIVE_MIN=\"" << 650 << "\"\n";
  outputASCII << "RELATIVE_MAX=\"" << 1250 << "\"\n";

  outputASCII << "\n";

  outputASCII << "PRESET_ORGAN_INTENSITY_NAME=\""<<"Lung"<< "\"\n";

  outputASCII << "RELATIVE_MIN=\"" << 0 << "\"\n";
  outputASCII << "RELATIVE_MAX=\"" << 1456 << "\"\n";

  outputASCII << "\n";

  outputASCII << "PRESET_ORGAN_INTENSITY_NAME=\""<<"Gas"<< "\"\n";

  outputASCII << "RELATIVE_MIN=\"" << 750 << "\"\n";
  outputASCII << "RELATIVE_MAX=\"" << 751 << "\"\n";

  outputASCII << "PRESET_COLOR_NAME=\""<<"RedGreen"<< "\"\n";

  outputASCII << "IMAGE_RED=\"" << 1.0 << "\"\n";
  outputASCII << "IMAGE_GREEN=\"" << 0.0 << "\"\n";
  outputASCII << "IMAGE_BLUE=\"" << 0.0 << "\"\n";

  outputASCII << "OVERLAY_RED=\"" << 0.0 << "\"\n";
  outputASCII << "OVERLAY_GREEN=\"" << 1.0 << "\"\n";
  outputASCII << "OVERLAY_BLUE=\"" << 0.0 << "\"\n";

  outputASCII << "\n";

  outputASCII << "PRESET_COLOR_NAME=\""<<"GreenBlue"<< "\"\n";

  outputASCII << "IMAGE_RED=\"" << 0.0 << "\"\n";
  outputASCII << "IMAGE_GREEN=\"" << 1.0 << "\"\n";
  outputASCII << "IMAGE_BLUE=\"" << 0.0 << "\"\n";

  outputASCII << "OVERLAY_RED=\"" << 0.0 << "\"\n";
  outputASCII << "OVERLAY_GREEN=\"" << 0.0 << "\"\n";
  outputASCII << "OVERLAY_BLUE=\"" << 1.0 << "\"\n";

  outputASCII << "\n";

  outputASCII << "PRESET_COLOR_NAME=\""<<"BlueRed"<< "\"\n";

  outputASCII << "IMAGE_RED=\"" << 0.0 << "\"\n";
  outputASCII << "IMAGE_GREEN=\"" << 0.0 << "\"\n";
  outputASCII << "IMAGE_BLUE=\"" << 1.0 << "\"\n";

  outputASCII << "OVERLAY_RED=\"" << 1.0 << "\"\n";
  outputASCII << "OVERLAY_GREEN=\"" << 0.0 << "\"\n";
  outputASCII << "OVERLAY_BLUE=\"" << 0.0 << "\"\n";

  outputASCII << "\n";

  outputASCII << "PRESET_COLOR_NAME=\""<<"MagentaYellow"<< "\"\n";

  outputASCII << "IMAGE_RED=\"" << 1.0 << "\"\n";
  outputASCII << "IMAGE_GREEN=\"" << 0.0 << "\"\n";
  outputASCII << "IMAGE_BLUE=\"" << 1.0 << "\"\n";

  outputASCII << "OVERLAY_RED=\"" << 1.0 << "\"\n";
  outputASCII << "OVERLAY_GREEN=\"" << 1.0 << "\"\n";
  outputASCII << "OVERLAY_BLUE=\"" << 0.0 << "\"\n";

  outputASCII << "\n";

  outputASCII << "FLUID_PARAM_NAME=\""<<"Prostate Shrinking"<< "\"\n";

  outputASCII << "FINE_ALPHA=\"" << 0.02 << "\"\n";
  outputASCII << "FINE_BETA=\"" << 0.02 << "\"\n";
  outputASCII << "FINE_GAMMA=\"" << 0.0001 << "\"\n";
  outputASCII << "FINE_MAXPERT=\"" << 0.5 << "\"\n";
  outputASCII << "FINE_NUMBASIS=\"" << 2000 << "\"\n";
  outputASCII << "FINE_NUMITER=\"" << 5 << "\"\n";

  outputASCII << "MEDIUM_ALPHA=\"" << 0.01 << "\"\n";
  outputASCII << "MEDIUM_BETA=\"" << 0.01 << "\"\n";
  outputASCII << "MEDIUM_GAMMA=\"" << 0.001 << "\"\n";
  outputASCII << "MEDIUM_MAXPERT=\"" << 0.5 << "\"\n";
  outputASCII << "MEDIUM_NUMBASIS=\"" << 2000 << "\"\n";
  outputASCII << "MEDIUM_NUMITER=\"" << 50 << "\"\n";

  outputASCII << "COARSE_ALPHA=\"" << 0.0 << "\"\n";
  outputASCII << "COARSE_BETA=\"" << 0.0 << "\"\n";
  outputASCII << "COARSE_GAMMA=\"" << 0.0 << "\"\n";
  outputASCII << "COARSE_MAXPERT=\"" << 0.0 << "\"\n";
  outputASCII << "COARSE_NUMBASIS=\"" << 0 << "\"\n";
  outputASCII << "COARSE_NUMITER=\"" << 0 << "\"\n";
  outputASCII << "\n";

  outputASCII << "FLUID_PARAM_NAME=\""<<"Prostate Warping"<< "\"\n";

  outputASCII << "FINE_ALPHA=\"" << 0.02 << "\"\n";
  outputASCII << "FINE_BETA=\"" << 0.02 << "\"\n";
  outputASCII << "FINE_GAMMA=\"" << 0.0001 << "\"\n";
  outputASCII << "FINE_MAXPERT=\"" << 0.5 << "\"\n";
  outputASCII << "FINE_NUMBASIS=\"" << 2000 << "\"\n";
  outputASCII << "FINE_NUMITER=\"" << 50 << "\"\n";

  outputASCII << "MEDIUM_ALPHA=\"" << 0.01 << "\"\n";
  outputASCII << "MEDIUM_BETA=\"" << 0.01 << "\"\n";
  outputASCII << "MEDIUM_GAMMA=\"" << 0.001 << "\"\n";
  outputASCII << "MEDIUM_MAXPERT=\"" << 0.5 << "\"\n";
  outputASCII << "MEDIUM_NUMBASIS=\"" << 2000 << "\"\n";
  outputASCII << "MEDIUM_NUMITER=\"" << 50 << "\"\n";

  outputASCII << "COARSE_ALPHA=\"" << 0.0 << "\"\n";
  outputASCII << "COARSE_BETA=\"" << 0.0 << "\"\n";
  outputASCII << "COARSE_GAMMA=\"" << 0.0 << "\"\n";
  outputASCII << "COARSE_MAXPERT=\"" << 0.0 << "\"\n";
  outputASCII << "COARSE_NUMBASIS=\"" << 0 << "\"\n";
  outputASCII << "COARSE_NUMITER=\"" << 0 << "\"\n";
  outputASCII << "\n";



  if (outputASCII.fail())
  {
    throw std::runtime_error("ofstream failed writing ascii header");
  }
  outputASCII.close();

}

/////////////////////////////
//  loadPresetFileCallback //
/////////////////////////////
void 
ImMap
:: loadPresetFileCallback()
{
  std::string name,colorName,fluidParamName;
  float mini = 0.0F, maxi = 0.0F;
  float imageRed = 0.0F, imageGreen = 0.0F, imageBlue = 0.0F, 
    overlayRed = 0.0F, overlayGreen = 0.0F, overlayBlue = 0.0F;
  int presetIndex = 0;
  fluidParameters newParameters;

  char* presetFilename = (char*)_presetFilename.c_str();
  //
  // first read ascii header
  //
  std::ifstream inputASCII(presetFilename);
  if (inputASCII.fail())
  {
    std::cout << "creating preset file" << std::endl;
    try{
      _createPresetFile();
      loadPresetFileCallback();
    }catch(...){
      std::cout << "failed to create preset file" << std::endl;
      return;
    }
  }      
  else
  {
    bool foundName   = false;
    bool foundRelativeMin = false;
    bool foundRelativeMax   = false;
    bool foundColorName = false;
    bool foundImageRed = false;
    bool foundImageGreen = false;
    bool foundImageBlue = false;
    bool foundOverlayRed = false;
    bool foundOverlayGreen = false;
    bool foundOverlayBlue = false;
    bool foundFluidParamName = false;
    bool foundFineAlpha = false;
    bool foundFineBeta = false;
    bool foundFineGamma = false;
    bool foundFineMaxPert = false;
    bool foundFineNumBasis = false;
    bool foundFineNumIter = false;
    bool foundMediumAlpha = false;
    bool foundMediumBeta = false;
    bool foundMediumGamma = false;
    bool foundMediumMaxPert = false;
    bool foundMediumNumBasis = false;
    bool foundMediumNumIter = false;
    bool foundCoarseAlpha = false;
    bool foundCoarseBeta = false;
    bool foundCoarseGamma = false;
    bool foundCoarseMaxPert = false;
    bool foundCoarseNumBasis = false;
    bool foundCoarseNumIter = false;
    bool foundAllIntensityData = false;
    bool foundAllColorData = false;
    bool foundAllFluidParams = false;

    Intensity newOrgan;
    ColorPreset newColor;

    do{
      foundName   = false;
      foundRelativeMin = false;
      foundRelativeMax   = false;
      foundColorName = false;
      foundImageRed = false;
      foundImageGreen = false;
      foundImageBlue = false;
      foundOverlayRed = false;
      foundOverlayGreen = false;
      foundOverlayBlue = false;
      foundAllIntensityData = false;
      foundAllColorData = false;
      foundAllFluidParams = false;
      foundFluidParamName = false;
      foundFineAlpha = false;
      foundFineBeta = false;
      foundFineGamma = false;
      foundFineMaxPert = false;
      foundFineNumBasis = false;
      foundFineNumIter = false;
      foundMediumAlpha = false;
      foundMediumBeta = false;
      foundMediumGamma = false;
      foundMediumMaxPert = false;
      foundMediumNumBasis = false;
      foundMediumNumIter = false;
      foundCoarseAlpha = false;
      foundCoarseBeta = false;
      foundCoarseGamma = false;
      foundCoarseMaxPert = false;
      foundCoarseNumBasis = false;
      foundCoarseNumIter = false;
      while (true)
      { 
        std::string key;
        std::getline(inputASCII, key, '=');
        if (inputASCII.eof())
        {
          presetIndex = 3;//to break inside the second switch
          break;
        }

        if (key.find('\n') != std::string::npos)
        {
          key.erase(0, key.find('\n') + 1);
        }
        else if (inputASCII.fail())
        {
          std::cerr << "died on: " << key << std::endl;
          throw std::runtime_error("ifstream failed reading header");
        } 


        if (strcmp(key.c_str(),"PRESET_ORGAN_INTENSITY_NAME") == 0)
        {
          presetIndex = 0;
        }
        else 
        {
          if (strcmp(key.c_str(),"PRESET_COLOR_NAME") == 0)
          {
            presetIndex = 1;
          }
          else
          {
            if (strcmp(key.c_str(),"FLUID_PARAM_NAME") == 0)
            {
              presetIndex = 2;
            }
          }
        }

        std::string value;
        std::getline(inputASCII, value);

        if (inputASCII.fail())
        {
          std::cerr << "died on: " << value << std::endl;
          throw std::runtime_error("ifstream failed reading header");
        }     

        switch(presetIndex)
        {
        case 0:
          {
            if (key.find("PRESET_ORGAN_INTENSITY_NAME") == 0)
            {
              name = _parseValue(value);
              foundName = true;
            }
            else if (key.find("RELATIVE_MIN") == 0)
            {
              mini = atof(_parseValue(value).c_str());
              foundRelativeMin = true;
            }
            else if (key.find("RELATIVE_MAX") == 0)
            {
              maxi = atof(_parseValue(value).c_str());
              foundRelativeMax = true;
              foundAllIntensityData = true;
              break;
            }
            break;
          }
        case 1:
          {
            if (key.find("PRESET_COLOR_NAME") == 0)
            {
              colorName = _parseValue(value);
              foundColorName = true;
            }
            else if (key.find("IMAGE_RED") == 0)
            {
              imageRed = atof(_parseValue(value).c_str());
              foundImageRed = true;
            }
            else if (key.find("IMAGE_GREEN") == 0)
            {
              imageGreen = atof(_parseValue(value).c_str());
              foundImageGreen = true;
            }
            else if (key.find("IMAGE_BLUE") == 0)
            {
              imageBlue = atof(_parseValue(value).c_str());
              foundImageBlue = true;
            }
            else if (key.find("OVERLAY_RED") == 0)
            {
              overlayRed = atof(_parseValue(value).c_str());
              foundOverlayRed = true;
            }
            else if (key.find("OVERLAY_GREEN") == 0)
            {
              overlayGreen = atof(_parseValue(value).c_str());
              foundOverlayGreen = true;
            }
            else if (key.find("OVERLAY_BLUE") == 0)
            {
              overlayBlue = atof(_parseValue(value).c_str());
              foundOverlayBlue = true;
              foundAllColorData = true;
              break;
            }
            break;
          }
        case 2:
          {
            if (key.find("FLUID_PARAM_NAME") == 0)
            {
              fluidParamName = _parseValue(value);
              foundFluidParamName = true;
            }
            else if (key.find("FINE_ALPHA") == 0)
            {
              newParameters.params[0].alpha = atof(_parseValue(value).c_str());
              foundFineAlpha = true;
            }
            else if (key.find("FINE_BETA") == 0)
            {
              newParameters.params[0].beta = atof(_parseValue(value).c_str());
              foundFineBeta = true;
            }
            else if (key.find("FINE_GAMMA") == 0)
            {
              newParameters.params[0].gamma = atof(_parseValue(value).c_str());
              foundFineGamma = true;
            }
            else if (key.find("FINE_MAXPERT") == 0)
            {
              newParameters.params[0].maxPerturbation = atof(_parseValue(value).c_str());
              foundFineMaxPert = true;
            }
            else if (key.find("FINE_NUMBASIS") == 0)
            {
              newParameters.params[0].numBasis = atoi(_parseValue(value).c_str());
              foundFineNumBasis = true;
            }
            else if (key.find("FINE_NUMITER") == 0)
            {
              newParameters.params[0].numIterations = atoi(_parseValue(value).c_str());
              foundFineNumIter = true;
            }
            else if (key.find("MEDIUM_ALPHA") == 0)
            {
              newParameters.params[1].alpha = atof(_parseValue(value).c_str());
              foundMediumAlpha = true;
            }
            else if (key.find("MEDIUM_BETA") == 0)
            {
              newParameters.params[1].beta = atof(_parseValue(value).c_str());
              foundMediumBeta = true;
            }
            else if (key.find("MEDIUM_GAMMA") == 0)
            {
              newParameters.params[1].gamma = atof(_parseValue(value).c_str());
              foundMediumGamma = true;
            }
            else if (key.find("MEDIUM_MAXPERT") == 0)
            {
              newParameters.params[1].maxPerturbation = atof(_parseValue(value).c_str());
              foundMediumMaxPert = true;
            }
            else if (key.find("MEDIUM_NUMBASIS") == 0)
            {
              newParameters.params[1].numBasis = atoi(_parseValue(value).c_str());
              foundMediumNumBasis = true;
            }
            else if (key.find("MEDIUM_NUMITER") == 0)
            {
              newParameters.params[1].numIterations = atoi(_parseValue(value).c_str());
              foundMediumNumIter = true;
            }
            else if (key.find("COARSE_ALPHA") == 0)
            {
              newParameters.params[2].alpha = atof(_parseValue(value).c_str());
              foundCoarseAlpha = true;
            }
            else if (key.find("COARSE_BETA") == 0)
            {
              newParameters.params[2].beta = atof(_parseValue(value).c_str());
              foundCoarseBeta = true;
            }
            else if (key.find("COARSE_GAMMA") == 0)
            {
              newParameters.params[2].gamma = atof(_parseValue(value).c_str());
              foundCoarseGamma = true;
            }
            else if (key.find("COARSE_MAXPERT") == 0)
            {
              newParameters.params[2].maxPerturbation = atof(_parseValue(value).c_str());
              foundCoarseMaxPert = true;
            }
            else if (key.find("COARSE_NUMBASIS") == 0)
            {
              newParameters.params[2].numBasis = atoi(_parseValue(value).c_str());
              foundCoarseNumBasis = true;
            }
            else if (key.find("COARSE_NUMITER") == 0)
            {
              newParameters.params[2].numIterations = atoi(_parseValue(value).c_str());
              foundCoarseNumIter = true;
              foundAllFluidParams = true;
              break;
            }
            break;
          }     
        case 3:
          {
            break;
          }
        }

        if (foundAllIntensityData || foundAllColorData || foundAllFluidParams )
        {
          break;
        }
      }
      switch(presetIndex)
      {
      case 0 :
        {
          /* organ intensity preset */

          if (foundName &&
            foundRelativeMin &&
            foundRelativeMax ) 
          {
            //push the new preset in the file chooser

            newOrgan.relativeMin = mini;
            newOrgan.relativeMax = maxi;
            //image
            imagePresetIntensityChoice->add(name.c_str());
            _presetIntensity.push_back(newOrgan);
            //overlay
            overlayPresetIntensityChoice->add(name.c_str());
          }
          else
          {
            // make sure we got all the info we need
            if (!foundName)
            {
              std::cerr << "read preset file: could not find FILENAME" << std::endl;
              foundAllIntensityData = false;
            }
            if (!foundRelativeMin)
            {
              std::cerr << "read preset file: could not find RELATIVE_MIN" << std::endl;
              foundAllIntensityData = false;
            }
            if (!foundRelativeMax)
            {
              std::cerr << "read preset file: could not find RELATIVE_MAX" << std::endl;
              foundAllIntensityData = false;
            }
            if (!foundAllIntensityData)
            {
              fl_alert("preset file contains insufficient data");
            }
          }
          break;
        }

      case 1:
        {

          /* color preset */

          if (foundColorName &&
            foundImageRed &&
            foundImageGreen &&
            foundImageBlue &&
            foundOverlayRed &&
            foundOverlayGreen &&
            foundOverlayBlue )
          {
            newColor.image.red = imageRed;
            newColor.image.green = imageGreen;
            newColor.image.blue = imageBlue;
            newColor.overlay.red = overlayRed;
            newColor.overlay.green = overlayGreen;
            newColor.overlay.blue = overlayBlue;
            colorPresetChoice->add(colorName.c_str());
            _colorPresetIntensity.push_back(newColor);
          }
          else
          {
            // make sure we got all the info we need
            if (!foundColorName)
            {
              std::cerr << "read preset file: could not find COLORNAME" << std::endl;
              foundAllColorData = false;
            }
            if (!foundImageRed)
            {
              std::cerr << "read preset file: could not find IMAGE_RED" << std::endl;
              foundAllColorData = false;
            }
            if (!foundImageGreen)
            {
              std::cerr << "read preset file: could not find IMAGE_GREEN" << std::endl;
              foundAllColorData = false;
            }
            if (!foundImageBlue)
            {
              std::cerr << "read preset file: could not find IMAGE_BLUE" << std::endl;
              foundAllColorData = false;
            }
            if (!foundOverlayRed)
            {
              std::cerr << "read preset file: could not find OVERLAY_RED" << std::endl;
              foundAllColorData = false;
            }
            if (!foundOverlayGreen)
            {
              std::cerr << "read preset file: could not find OVERLAY_GREEN" << std::endl;
              foundAllColorData = false;
            }
            if (!foundOverlayBlue)
            {
              std::cerr << "read preset file: could not find OVERLAY_BLUE" << std::endl;
              foundAllColorData = false;
            }
            if (!foundAllColorData)
            {
              fl_alert("preset file contains insufficient data");
            }

          }
          break;
        }

      case 2:
        {

          /* fluid parameters preset */

          if (foundFluidParamName && foundFineAlpha && foundFineBeta &&
            foundFineGamma && foundFineMaxPert && foundFineNumBasis &&
            foundFineNumIter && foundMediumAlpha && foundMediumBeta &&
            foundMediumGamma && foundMediumMaxPert && foundMediumNumBasis &&
            foundMediumNumIter && foundCoarseAlpha && foundCoarseBeta &&
            foundCoarseGamma && foundCoarseMaxPert && foundCoarseNumBasis &&
            foundCoarseNumIter)
          {
            fluidParamChoice->add(fluidParamName.c_str());
            _fluidParamPreset.push_back(newParameters);
          }
          else
          {
            // make sure we got all the info we need
            if (!foundFluidParamName)
            {
              std::cerr << "read preset file: could not find FLUID_PARAM_NAME" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundFineAlpha)
            {
              std::cerr << "read preset file: could not find FINE_ALPHA" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundFineBeta)
            {
              std::cerr << "read preset file: could not find FINE_BETA" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundFineGamma)
            {
              std::cerr << "read preset file: could not find FINE_GAMMA" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundFineMaxPert)
            {
              std::cerr << "read preset file: could not find FINE_MAXPERT" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundFineNumBasis)
            {
              std::cerr << "read preset file: could not find FINE_NUMBASIS" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundFineNumIter)
            {
              std::cerr << "read preset file: could not find FINE_NUMITER" << std::endl;
              foundAllColorData = false;
            }
            if (!foundMediumAlpha)
            {
              std::cerr << "read preset file: could not find MEDIUM_ALPHA" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundMediumBeta)
            {
              std::cerr << "read preset file: could not find MEDIUM_BETA" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundMediumGamma)
            {
              std::cerr << "read preset file: could not find MEDIUM_GAMMA" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundMediumMaxPert)
            {
              std::cerr << "read preset file: could not find MEDIUM_MAXPERT" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundMediumNumBasis)
            {
              std::cerr << "read preset file: could not find MEDIUM_NUMBASIS" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundMediumNumIter)
            {
              std::cerr << "read preset file: could not find MEDIUM_NUMITER" << std::endl;
              foundAllColorData = false;
            }
            if (!foundCoarseAlpha)
            {
              std::cerr << "read preset file: could not find COARSE_ALPHA" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundCoarseBeta)
            {
              std::cerr << "read preset file: could not find COARSE_BETA" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundCoarseGamma)
            {
              std::cerr << "read preset file: could not find COARSE_GAMMA" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundCoarseMaxPert)
            {
              std::cerr << "read preset file: could not find COARSE_MAXPERT" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundCoarseNumBasis)
            {
              std::cerr << "read preset file: could not find COARSE_NUMBASIS" << std::endl;
              foundAllFluidParams = false;
            }
            if (!foundCoarseNumIter)
            {
              std::cerr << "read preset file: could not find COARSE_NUMITER" << std::endl;
              foundAllColorData = false;
            }
            if (!foundAllFluidParams)
            {
              fl_alert("preset file contains insufficient data");
            }

          }
          break;
        }
      case 3:
        {
          break;
        }
      }

    }while(!inputASCII.eof());

    //set to the origin 
    imagePresetIntensityChoice->value(0);
    overlayPresetIntensityChoice->value(0);
    colorPresetChoice->value(0);
    fluidParamChoice->value(0);
    fluidParametersCallback();

    inputASCII.close();
  }
}

//////////////////////////////
//  lockedIntensityCallback //
//////////////////////////////
void 
ImMap
:: lockedIntensityCallback()
{
  if (lockedIntensity->value())
  {
    overlayPresetIntensityChoice->deactivate();
    transmitImage->deactivate();
    transmitOverlay->deactivate();
    overlayValMin->deactivate();
    overlayValMax->deactivate();
  }
  else
  {
    overlayPresetIntensityChoice->activate();
    transmitImage->activate();
    transmitOverlay->activate();
    overlayValMin->activate();
    overlayValMax->activate();
  }
}

////////////////////////////////
// overlayValMinMaxCallback() //
////////////////////////////////

void 
ImMap
:: overlayValMinMaxCallback(){

  int overlayHisto = overlayChoice->value();


  if (overlayValMin->value() < _histograms[overlayHisto].getAbsoluteMin() )
  {
    overlayValMin->value(_histograms[overlayHisto].getAbsoluteMin());
  }

  // For the Min
  /////////////////

  // Update Image Histogram Window

  HistogramOverlayDisplay->updateRelativeMin(overlayValMin->value());
  _histograms[overlayHisto].setRelativeMin(overlayValMin->value());

  if (overlayValMax->value() > _histograms[overlayHisto].getAbsoluteMax() )
  {
    overlayValMax->value(_histograms[overlayHisto].getAbsoluteMax());
  }

  // For the Max
  /////////////////

  // Update Overlay Histogram Window

  HistogramOverlayDisplay->updateRelativeMax(overlayValMax->value());
  _histograms[overlayHisto].setRelativeMax(overlayValMax->value());

  // Update the histogram Slider

  updateHistogramSliderCallback();

  // To redraw the image
  overlayIWChangedCallback();

}




// Histogram to toggle from image histogram to overlay histogram
//     label I -> image histogram
//     label O -> overlay histogram 

///////////////////////////////
// histogramButtonCallback() //
///////////////////////////////

void
ImMap
::  histogramButtonCallback(){

  if (_histogramImage){
    _histogramImage = false;
    histogramButton->label("O");
    histogramButton->tooltip("Overlay");
  }
  else{
    _histogramImage = true;
    histogramButton->label("I");
    histogramButton->tooltip("Image");
  }
  updateHistogramSliderCallback();
}

/////////////////////////////
// histogramLinkCallback() //
/////////////////////////////

void
ImMap
::  histogramLinkCallback(){

  if(_histogramLinked){
    _histogramLinked = false;
    histogramLinkButton->label("U");
    histogramLinkButton->tooltip("Histograms are Unlinked");
  }
  else{
    _histogramLinked = true;
    histogramLinkButton->label("L");
    histogramLinkButton->tooltip("Histograms are Linked");

    if (_histogramImage){
      transmitHistoImageCallback();
    }
    else{
      transmitHistoOverlayCallback();
    }
  }
}

//////////////////////////
// zoomButtonCallback() //
//////////////////////////

void 
ImMap
:: zoomButtonCallback(float val){

  _zoomValue += val;
  if (_zoomValue<0.1){
    _zoomValue=0.1;
  }
  if (_zoomValue>25){
    _zoomValue=25;
  }
  axial2DWindow->setWindowZoom(_zoomValue);
  coronal2DWindow->setWindowZoom(_zoomValue);
  sagittal2DWindow->setWindowZoom(_zoomValue);

  imageZoomVal->value(_zoomValue);

  _redrawImageWindows();
}

/////////////////////////
// setOriginCallback() //
/////////////////////////

void  
ImMap
:: setOriginCallback(){

  int imageIndex=imageSaveChoice->value();

  ImageIndexType origin;
  origin[0]=XOrigin->value();
  origin[1]=YOrigin->value();
  origin[2]=ZOrigin->value();

  _loadedImages[imageIndex]->setOrigin(origin);
  if ((imageIndex==imageChoice->value()) ||
    (imageIndex==overlayChoice->value()) ){
    _redrawImageWindows();
  }
  infoImageCallback();
}

//////////////////////////////
// viewCrosshairsCallback() //
//////////////////////////////

void ImMap
::viewCrosshairsCallback()
{
  axial2DWindow->setViewCrosshairs(viewCrosshairsButton->value());
  coronal2DWindow->setViewCrosshairs(viewCrosshairsButton->value());
  sagittal2DWindow->setViewCrosshairs(viewCrosshairsButton->value());
  _redrawImageWindows();
}

/////////////////////////////
// viewImageInfoCallback() //
/////////////////////////////

void ImMap
::viewImageInfoCallback()
{
  axial2DWindow->setViewImageInfo(viewImageInfoButton->value());
  coronal2DWindow->setViewImageInfo(viewImageInfoButton->value());
  sagittal2DWindow->setViewImageInfo(viewImageInfoButton->value());
  _redrawImageWindows();
}

//////////////////////////////
// lineWidthCallback(width) //
//////////////////////////////

void ImMap
::lineWidthCallback(const double& width)
{
  axial2DWindow->setLineWidth(width);
  coronal2DWindow->setLineWidth(width);
  sagittal2DWindow->setLineWidth(width);
  _redrawImageWindows();
}

void ImMap
::window3DBGColorCallback()
{
  // get old background color
  double r,g,b;
  surface3DWindow->getBackgroundColor(r,g,b);
  fl_color_chooser("Select Background Color for 3D Window...",
                   r,g,b);
  surface3DWindow->setBackgroundColor(r,g,b);
  surface3DWindow->updateDisplay();
}

void
ImMap::
_printParameters(const char* startStr, const char* stopStr,
                 const char* intensityStr)
{
  std::cout << "\n" << startStr << " = "
            << roiStartX->value() << " "
            << roiStartY->value() << " "
            << roiStartZ->value() << "\n"
            << stopStr << " = "
            << roiStopX->value() << " "
            << roiStopY->value() << " "
            << roiStopZ->value() << "\n"
            << intensityStr << " = "
            << imageValMin->value() << " "
            << imageValMax->value() << std::endl;
}

void
ImMap::
printParametersCallback()
{
  _printParameters("ROI_START", "ROI_STOP", "INTENSITY");
}



///////////////////////////////
//  Wizard Related Callback  //
///////////////////////////////

// To display info about the reference image (on the wizard Window),
// once this image has been loaded (added to the list);

void 
ImMap
::  referenceInfoCallback(){

  char temp[1024];

  int indexImage = _loadedImages.size()-1;

  // Name

  RefNameInfo->value(_imageNames[indexImage].c_str());
  RefImageName->label(_imageNames[indexImage].c_str());

  // Dimensions

  ImageSizeType Dimensions(_loadedImages[indexImage]->getSize());
  //Dimensions = (_loadedImages[indexImage]->GetLargestPossibleRegion()).GetSize();
  sprintf(temp,"%d x %d x %d",(int)Dimensions[0],(int)Dimensions[1],(int)Dimensions[2]);
  RefDimensionsInfo->value(temp);

  // PixelSize

  ImageSizeType PixelSize = _loadedImages[indexImage]->getSpacing();
  sprintf(temp,"%.3f x %.3f x %.3f cm/voxel",PixelSize[0],PixelSize[1],PixelSize[2]);
  RefPixelSizeInfo->value(temp);

  // Orientation

  RefOrientationInfo->value("   -   ");

  // Origin

  ImageIndexType Origin = _loadedImages[indexImage]->getOrigin();
  sprintf(temp,"%.3f  %.3f  %.3f",Origin[0],Origin[1],Origin[2]);
  RefOriginInfo->value(temp);

  // Min/Max

  sprintf(temp,"%.1f  / %.1f",_histograms[indexImage].getAbsoluteMin(),
    _histograms[indexImage].getAbsoluteMax());
  RefMinMaxInfo->value(temp);

  // DataType

  RefDataTypeInfo->value("    -    ");   

  // To display our reference image
  imageChoice->value(indexImage);
  imageChoiceCallback();

  //cout<<"ctime() "<<ctime(time())<<endl;

}

// To display info about the Daily image, overlay, (on the wizard Window)
// once this image has been loaded (added to the list);

/////////////////////////
// dailyInfoCallback() //
/////////////////////////

void
ImMap
::  dailyInfoCallback(){

  char temp[1024];

  int indexImage = _loadedImages.size()-1;

  // Name

  DailyNameInfo->value(_imageNames[indexImage].c_str());
  DailyImageName->label(_imageNames[indexImage].c_str());

  // Dimensions

  ImageSizeType Dimensions(_loadedImages[indexImage]->getSize());
  sprintf(temp,"%d x %d x %d",(int)Dimensions[0],(int)Dimensions[1],(int)Dimensions[2]);
  DailyDimensionsInfo->value(temp);

  // PixelSize

  ImageSizeType PixelSize = _loadedImages[indexImage]->getSpacing();
  sprintf(temp,"%.3f x %.3f x %.3f cm/voxel",PixelSize[0],PixelSize[1],PixelSize[2]);
  DailyPixelSizeInfo->value(temp);

  // Orientation

  DailyOrientationInfo->value("   -   ");

  // Origin

  ImageIndexType Origin = _loadedImages[indexImage]->getOrigin();
  sprintf(temp,"%.3f  %.3f  %.3f",Origin[0],Origin[1],Origin[2]);
  DailyOriginInfo->value(temp);

  // Min/Max

  sprintf(temp,"%.1f  / %.1f",_histograms[indexImage].getAbsoluteMin(),
    _histograms[indexImage].getAbsoluteMax());
  DailyMinMaxInfo->value(temp);

  // DataType

  DailyDataTypeInfo->value("    -    ");  

  // To display our daily image (overlay)
  overlayChoice->value(indexImage);
  overlayChoiceCallback();
}

// To save the result of the translation

//////////////////////////////////
// saveTranslationResCallback() //
//////////////////////////////////

void
ImMap
::  saveTranslationResCallback(){

  // The resulting image is the last one added to the list
  int indexImage = _loadedImages.size()-1;

  imageSaveChoice->value(indexImage);
  saveImageCallback();

}

// Save the translation to a file

///////////////////////////////
// saveTranslationCallback() //
///////////////////////////////

void
ImMap
::  saveTranslationCallback(){

  char * filename =fl_file_chooser("Select a filename :","",0);

  if ( filename ){

    std::ofstream outfile;
    outfile.open(filename);

    time_t timer;
    timer=time(NULL);

    outfile << "# Date : " << asctime(localtime(&timer)) << std::endl;
    outfile << "# Reference Image : " << ReferenceFile->value() << std::endl;
    outfile << "# Target Image : " << dailyFile->value() << std::endl;
    outfile << "# ROI : " << WroiStartX->value() << " "
            << WroiStartY->value() << " "
            << WroiStartZ->value() << " "
            << WroiStopX->value() << " "
            << WroiStopY->value() << " "
            << WroiStopZ->value() << std::endl;
    outfile << "# Transformation : " << registrationTransform << std::endl;
  }

}

//////////////////////////
// initWizardCallback() //
//////////////////////////

void
ImMap
::initWizardCallback(){

  step2Button->value(0);
  step3Button->value(0);
  step4Button->value(0);
  step5Button->value(0);
  step6Button->value(0);
  step7Button->value(0);
  step8Button->value(0);

  RefImageName->label("None");
  DailyImageName->label("None");

  setROIGroup->hide();
  estimateTranslationGroup->hide();
  selectDailyGroup->hide();
  loadAnastructsGroup->hide();
  setROIRegistrationGroup->hide();
  registrationGroup->hide();

  selectReferenceGroup->show();

  DailyNameInfo->value(NULL);
  DailyDimensionsInfo->value(NULL);
  DailyPixelSizeInfo->value(NULL);
  DailyOriginInfo->value(NULL);
  DailyMinMaxInfo->value(NULL);

  RefNameInfo->value(NULL);
  RefDimensionsInfo->value(NULL);
  RefPixelSizeInfo->value(NULL);
  RefOriginInfo->value(NULL);
  RefMinMaxInfo->value(NULL);

  NextSelRef->deactivate();
  NextSelDaily->deactivate();
  NextTrans->deactivate();
  FinishWizard->deactivate();
  nextStepChoice->add("no defined");
  nextStepChoice->add("prostate ");
  nextStepChoice->add("lung");
  nextStepChoice->value(0);

  registrationStatus->value("");
  Xtranslation->value(0);
  Ytranslation->value(0);
  Ztranslation->value(0);

}

/////////////////////////////////////////
// wizardTranslationRegisterCallback() //
/////////////////////////////////////////

void
ImMap
::wizardTranslateRegisterCallback()
{
  translateRegisterCallback();

  // display last image in list, this is result of 
  // registration
  int indexImage = _loadedImages.size()-1;
  overlayChoice->value(indexImage);
  overlayChoiceCallback();
  NextTrans->activate();
}

/////////////////////////////////////////
// wizardRegistrationCallback() //
/////////////////////////////////////////

void
ImMap
::wizardRegistrationCallback()
{
  registrationStatus->value("Registration : running...");
  fluidRegisterCallback();
  int indexImage = _loadedImages.size()-2;
  imageChoice->value(indexImage);
  overlayChoice->value(0);
  overlayChoiceCallback();
  imageChoiceCallback();
  step8Button->value(1);
  FinishWizard->activate();
  registrationStatus->value("Registration DONE");
}

/////////////////////////////////////////
// wizardLoadAnastructCallback() //
/////////////////////////////////////////

void
ImMap
::wizardLoadAnastructsCallback()
{
  int next = nextStepChoice->value();
  if (next == 1)
  {
    imagePresetIntensityChoice->value(2);
    presetIntensityCallback(1);

    overlayPresetIntensityChoice->value(2);
    presetIntensityCallback(2);
  }
  else
  {
    imagePresetIntensityChoice->value(3);
    presetIntensityCallback(1);
    overlayPresetIntensityChoice->value(3);
    presetIntensityCallback(2);
  }
  BYUImageList->value(_loadedImages.size()-3);
  displayBYUListCallback();

}


//////////////////////////////
// displayBYUListCallback() //
//////////////////////////////

void
ImMap
:: displayBYUListCallback()
{
  unsigned int row = 0;

  //save the rows selected at this point
  for (row = 0 ; row < BYUTableDisplay->getNbRows() ; row++)
  {
    _lastRowsSelected[row] = BYUTableDisplay->row_selected(row);
  }

  BYUTableDisplay->clearTable();

  for (unsigned int imageIndex = 1 ; 
       imageIndex <= (unsigned int) BYUImageList->size() ; 
       imageIndex++)
  {
    if (BYUImageList->selected(imageIndex) == 1)
    {
      for (unsigned int imAnaIndex = 0 ; 
           imAnaIndex < _imAnaVectors[imageIndex].size() ; 
           ++imAnaIndex)
      {
        // get the name of the byu 
        std::string nameBYU(
          _imAnaVectors[imageIndex][imAnaIndex].anastruct.label);
        int pos = nameBYU.find_first_of(" ");
        if (pos != -1)
        {
          nameBYU.erase(nameBYU.begin() + pos, nameBYU.end());
        }
        char label[100];//add the imageIndex to the label
        sprintf(label,"%s  (%d)",nameBYU.c_str(),imageIndex);
        //add to the table properties of the anastruct
        //and also inageIndex and imAnaIndex to keep correspondance
        //between row and inageIndex or imAnaIndex
        BYUTableDisplay->addObjectToTable(
          imageIndex, 
          imAnaIndex, 
          label, 
          _imAnaVectors[imageIndex][imAnaIndex].visible, 
          _imAnaVectors[imageIndex][imAnaIndex].color, 
          _imAnaVectors[imageIndex][imAnaIndex].aspect, 
          _imAnaVectors[imageIndex][imAnaIndex].opacity);
      }
    }
  }
  //reselect the rows selected before
  for (row = 0 ; row < BYUTableDisplay->getNbRows() ; row++)
  {
    BYUTableDisplay->select_row(row, _lastRowsSelected[row]) ;
  }
}

///////////////////////////////
// BYUTableChangedCallback() //
///////////////////////////////

void 
ImMap
::BYUTableChangedCallback(Fl_Widget* w, void* obj)
{
  ImMap *immap = (ImMap*)obj;
  int selectedImage = 0;
  int selectedImAna = 0;
  unsigned int row = 0;
  if (immap->BYUTableDisplay->callback_context() == Fl_Table::CONTEXT_CELL)
  {
    if(immap->BYUTableDisplay->getNbRows() > 0)
    {
      row = immap->BYUTableDisplay->callback_row();
      //get Image and ImAna index from the row
      selectedImage = immap->BYUTableDisplay->getImageIndex(row);
      selectedImAna = immap->BYUTableDisplay->getImAnaIndex(row);
    }
    immap->viewSelectedObjectButton->value((immap->_imAnaVectors[selectedImage][selectedImAna].visible ? 1 : 0));
    switch(immap->_imAnaVectors[selectedImage][selectedImAna].aspect)
    {
    case ImAna::surface_representation:
      {
        immap->aspectSelectedObjectChooser->value(0);
        break;
      }
    case ImAna::wireframe_representation:
      {
        immap->aspectSelectedObjectChooser->value(1);
        break;
      }
    case ImAna::contours_representation:
      {
        immap->aspectSelectedObjectChooser->value(2);
        break;
      }     
    }
    immap->alphaSelectedObjectInput->value(immap->_imAnaVectors[selectedImage][selectedImAna].opacity);
  }
}


///////////////////////////////
// BYUTableCallback(int val) //
///////////////////////////////

void
ImMap
::BYUTableCallback(int val)
{
  switch(val)
  {
  case 0://selected all rows 
    {
      BYUTableDisplay->select_all_rows();
      break;
    }
  case 1://visibility
    {
      int selectedImage = 0;
      int selectedImAna = 0;
      for (unsigned int row = 0 ; row < BYUTableDisplay->getNbRows() ; row++)
      {
        if (BYUTableDisplay->row_selected(row) == 1)
        {
          //get Image and ImAna index from the row
          selectedImage = BYUTableDisplay->getImageIndex(row);
          selectedImAna = BYUTableDisplay->getImAnaIndex(row);

          bool visible = (viewSelectedObjectButton->value() == 1 ? true : false);
          _imAnaVectors[selectedImage][selectedImAna].visible=visible;

          if(selectedImage == imageChoice->value())
          {
            axial2DWindow->setImageContourVisibility(selectedImAna, visible);
            sagittal2DWindow->setImageContourVisibility(selectedImAna, visible);
            coronal2DWindow->setImageContourVisibility(selectedImAna, visible);
            int surfaceIndex = surface3DWindow->getSurfaceIndex(selectedImage,selectedImAna);
            surface3DWindow->setSurfaceVisibility(surfaceIndex, visible);
          }
          if(selectedImage == overlayChoice->value())
          {
            axial2DWindow->setOverlayContourVisibility(selectedImAna, visible);
            sagittal2DWindow->setOverlayContourVisibility(selectedImAna, visible);
            coronal2DWindow->setOverlayContourVisibility(selectedImAna, visible);
            int surfaceIndex = surface3DWindow->getSurfaceIndex(selectedImage,selectedImAna);
            surface3DWindow->setSurfaceVisibility(surfaceIndex, visible);
          }
        }
      }
      surface3DWindow->updateDisplay();
      _redrawImageWindows(); 
      displayBYUListCallback();


      break;
    }
  case 2://color
    {
      bool colorPanelCalled = false;//flag to avoid the color panel called in case of multiselection 
      Vector3D<double> newColor;
      for (unsigned int row = 0 ; row < BYUTableDisplay->getNbRows() ; row++)
      {
        if (BYUTableDisplay->row_selected(row) == 1)
        {
          //get Image and ImAna index from the row
          int selectedImage = BYUTableDisplay->getImageIndex(row);
          int selectedImAna = BYUTableDisplay->getImAnaIndex(row);

          switch(colorSelectedObjectMenu->value())
          {
          case 0://red
            {
              newColor.set(1,0,0);
              break;
            }
          case 1://green
            {
              newColor.set(0,1,0);
              break;
            }
          case 2://blue
            {
              newColor.set(0,0,1);
              break;
            }
          case 3://yellow
            {
              newColor.set(1,1,0);
              break;
            }
          case 4://yellow75
            {
              newColor.set(0.75,0.75,0);
              break;
            }
          case 5://magenta
            {
              newColor.set(1,0,1);
              break;
            }
          case 6://magenta75
            {
              newColor.set(0.75,0,0.75);
              break;
            }
          case 7://cyan
            {
              newColor.set(0,1,1);
              break;
            }
          case 8://cyan75
            {
              newColor.set(0,0.75,0.75);
              break;
            }
          case 9://white
            {
              newColor.set(1,1,1);
              break;
            }
          case 10://black
            {
              newColor.set(0,0,0);
              break;
            }
          case 11://silver
            {
              newColor.set(0.75,0.75,0.75);
              break;
            }
          case 12://gray
            {
              newColor.set(0.5,0.5,0.5);
              break;
            }
          case 13://maroon
            {
              newColor.set(0.5,0,0);
              break;
            }
          case 14://green
            {
              newColor.set(0,0.5,0);
              break;
            }
          case 15://navy
            {
              newColor.set(0,0,0.5);
              break;
            }
          case 16://purple
            {
              newColor.set(0.5,0,0.5);
              break;
            }
          case 17://olive
            {
              newColor.set(0.5,0.5,0);
              break;
            }
          case 18://teal
            {
              newColor.set(0,0.5,0.5);
              break;
            }
          case 19://color panel
            {
              if (colorPanelCalled == false)
              {
                newColor = _imAnaVectors[selectedImage][selectedImAna].color;
                fl_color_chooser("color chooser for anastruct ",
                                 newColor[0],
                                 newColor[1],
                                 newColor[2]);
                newColor.set(newColor[0],newColor[1],newColor[2]);
                colorPanelCalled = true; 
              }
              break;
            }

          }
          _imAnaVectors[selectedImage][selectedImAna].color=newColor;

          if(selectedImage == imageChoice->value())
          {
            axial2DWindow->setImageAnastructColor(selectedImAna, newColor[0],newColor[1],newColor[2]);
            int surfaceIndex = surface3DWindow->getSurfaceIndex(selectedImage,selectedImAna);
            surface3DWindow->setSurfaceColor(surfaceIndex, newColor[0],newColor[1],newColor[2]);
            surface3DWindow->setAnastructColor(surfaceIndex, newColor[0],newColor[1],newColor[2]);
          }
          if(selectedImage == overlayChoice->value())
          {
            axial2DWindow->setOverlayAnastructColor(selectedImAna, newColor[0],newColor[1],newColor[2]);
            int surfaceIndex = surface3DWindow->getSurfaceIndex(selectedImage,selectedImAna);
            surface3DWindow->setSurfaceColor(surfaceIndex, newColor[0],newColor[1],newColor[2]);
            surface3DWindow->setAnastructColor(surfaceIndex, newColor[0],newColor[1],newColor[2]);
          }
        }
      }
      _updateContours();
      surface3DWindow->updateDisplay();
      _redrawImageWindows(); 
      displayBYUListCallback();
      break;

    }
  case 3://aspect
    {
      for (unsigned int row = 0 ; row < BYUTableDisplay->getNbRows() ; row++)
      {
        if (BYUTableDisplay->row_selected(row) == 1)
        {
          //get Image and ImAna index from the row
          int selectedImage = BYUTableDisplay->getImageIndex(row);
          int selectedImAna = BYUTableDisplay->getImAnaIndex(row);

          ImAna::SurfaceRepresentationType newAspect = 
            ImAna::surface_representation;
          switch(aspectSelectedObjectChooser->value())
          {
          case 0://surface
            {
              newAspect = ImAna::surface_representation;
              break;
            }
          case 1:
            {
              newAspect = ImAna::wireframe_representation;
              break;
            }
          case 2:
            {
              newAspect = ImAna::contours_representation;
              break;
            }     
          }
          _imAnaVectors[selectedImage][selectedImAna].aspect=newAspect;

          if((selectedImage == imageChoice->value())||(selectedImage == overlayChoice->value()))
          {
            int surfaceIndex = surface3DWindow->getSurfaceIndex(selectedImage,selectedImAna);
            surface3DWindow->setSurfaceRepresentation(surfaceIndex, (SurfaceViewWindow::SurfaceRepresentationType)newAspect);
          }
        }
      }
      surface3DWindow->updateDisplay();
      displayBYUListCallback();
      break;
    }
  case 4://opacity
    {
      for (unsigned int row = 0 ; row < BYUTableDisplay->getNbRows() ; row++)
      {
        if (BYUTableDisplay->row_selected(row) == 1)
        {
          //get Image and ImAna index from the row
          int selectedImage = BYUTableDisplay->getImageIndex(row);
          int selectedImAna = BYUTableDisplay->getImAnaIndex(row);

          _imAnaVectors[selectedImage][selectedImAna].opacity=alphaSelectedObjectInput->value();

          if((selectedImage == imageChoice->value())||(selectedImage == overlayChoice->value()))
          {
            int surfaceIndex = surface3DWindow->getSurfaceIndex(selectedImage,selectedImAna);
            surface3DWindow->setSurfaceOpacity(surfaceIndex, alphaSelectedObjectInput->value());
          }
        }
      }

      surface3DWindow->updateDisplay();
      displayBYUListCallback();
      break;
    }

}


}

/////////////////////////////
// loadAnastructCallback() //
/////////////////////////////

void 
ImMap 
::loadAnastructCallback()
{
  unsigned int imageIndex = BYUImageList->value();
  int moreThanOneImageSelected = 0;

  for (unsigned int index = 0 ; 
       index < (unsigned int) BYUImageList->size() ; 
       index++)
  {
    if (BYUImageList->selected(index))
    {
      moreThanOneImageSelected++;
    }
  }
  if ((moreThanOneImageSelected>1)||(imageIndex == 0))
  {
    fl_alert("Please select an image first (only one).");
    return;
  }

  char* fileName = fl_file_chooser(" Choose an Anastruct File ","","");
  if (fileName == NULL)
  {
    return;
  }
  _loadAnastruct(fileName, imageIndex);


}

//////////////////////////////////////////
// _loadAnastruct(filename, imageIndex) //
//////////////////////////////////////////
void
ImMap
::_loadAnastruct(const std::string& fileName, const unsigned int& imageIndex)
{
  //
  // load the anastruct (world coordinates)
  //
  Anastruct anastruct;

  try 
    {
      AnastructUtils::readPLUNCAnastruct(anastruct, (char*)fileName.c_str());
    }
  catch(...)
    {
      _updateStatusBuffer("Failed to read anastruct.");
      fl_alert("Failed to read anastruct.");
      return;      
    }

  //
  // create a surface and push it back
  //

  AnastructUtils::capContour(anastruct);
  Surface surface;
  AnastructUtils::anastructToSurfacePowerCrust(anastruct, surface);
  if (refineButton->value())
    {
      SurfaceUtils::refineSurface(surface);
    }

  AnastructUtils::
    worldToImageIndex(anastruct,
                      _loadedImages[imageIndex]->getOrigin(),
                      _loadedImages[imageIndex]->getSpacing());

  // add the new ImAna in the list
  ImAna loadedImAna(anastruct, surface);

  _imAnaVectors[imageIndex].push_back(loadedImAna);

  if (imageIndex == (unsigned int)(imageChoice->value()) ||
      imageIndex == (unsigned int)(overlayChoice->value()) )
  {
    _updateContours();
    _updateSurfaces();
    _redrawImageWindows();      
  }

  displayBYUListCallback();
}

///////////////////////
// loadBYUCallback() //
///////////////////////

void 
ImMap 
::loadBYUCallback()
{
  int imageIndex = BYUImageList->value();
  int moreThanOneImageSelected = 0;

  for (unsigned int index = 0 ; 
       index < (unsigned int) BYUImageList->size() ; index++)
  {
    if (BYUImageList->selected(index))
    {
      moreThanOneImageSelected++;
    }
  }
  if ((moreThanOneImageSelected>1)||(imageIndex==0))
  {
    fl_alert("Please select an image first (only one).");
    return;
  }

  char* BYUFile = fl_file_chooser(" Choose a BYU File ","","");
  if (BYUFile == NULL)
  {
    return;
  }

  _loadBYU(BYUFile,imageIndex);
}

////////////////////////////////////
// _loadBYU(filename, imageIndex) //
////////////////////////////////////
void
ImMap
::_loadBYU(const std::string& fileName, const unsigned int& imageIndex)
{
  // get the name of the byu (the filename without the path)
  std::string  nameBYU(fileName);
  int pos = nameBYU.find_last_of("/");
  nameBYU.erase(nameBYU.begin(), nameBYU.begin() + pos + 1);
  pos = nameBYU.find_last_of(".");
  std::string Extension = nameBYU.substr(pos+1);

  // load the surface
  Surface loadedSurface;
  try 
  {
    if (Extension == "off") {
      loadedSurface.readOFF(fileName.c_str());
    } else {
      loadedSurface.readBYU((char*)fileName.c_str());
    }
  }
  catch(std::exception &e)
  {
    _updateStatusBuffer(e.what());
    _updateStatusBuffer("Failed to read byu surface.");
    fl_alert("Failed to read byu surface.");
    return;
  }
  catch(...)
  {
    _updateStatusBuffer("Failed to read byu surface.");
    fl_alert("Failed to read byu surface.");
    return;
  }

  //
  // create an anastruct and push it back
  //

  Anastruct loadedAnastruct;
  _createAnastruct(loadedSurface,
    _loadedImages[imageIndex]->getOrigin(),
    _loadedImages[imageIndex]->getSpacing(),
    nameBYU,
    loadedAnastruct);

  //add the new ImAna in the list
  ImAna loadedImAna(loadedAnastruct, loadedSurface);

  _imAnaVectors[imageIndex].push_back(loadedImAna);

  if ( imageIndex == (unsigned int)(imageChoice->value()) ||
       imageIndex == (unsigned int)(overlayChoice->value()) )
  {
    _updateContours();
    _updateSurfaces();
    _redrawImageWindows();      
  }

  displayBYUListCallback();
}


//////////////////////////
// unloadBYUCallback() //
/////////////////////////
void 
ImMap 
::unloadBYUCallback()
{
  int imageIndex = BYUImageList->value();
  bool noBYUSelected = true;

  if (imageIndex == 0)
  {
    fl_alert("Please select an image first!");
    return;
  }
  int rowIndex = BYUTableDisplay->getNbRows()-1;
  while(rowIndex>=0)
  {
    if (BYUTableDisplay->row_selected(rowIndex) == 1)
    {
      //get Image and ImAna index from the row
      unsigned int index = rowIndex;
          int selectedImage = BYUTableDisplay->getImageIndex(index);
          int selectedImAna = BYUTableDisplay->getImAnaIndex(index);
      _imAnaVectors[selectedImage].erase(_imAnaVectors[selectedImage].begin()+ selectedImAna);
      noBYUSelected = false;
    }
    rowIndex--;
  }
  if (noBYUSelected)
  {
    fl_alert("Please select a surface first!");
    return;
  }
  //
  // removing name from list is handled by displayBYUListCallback()
  //

  if (imageIndex == imageChoice->value() ||
    imageIndex == overlayChoice->value())
  {
    _updateContours();
    _updateSurfaces();
    _redrawImageWindows();      
  }

  displayBYUListCallback();
}

///////////////////////
// saveBYUCallback() //
///////////////////////

void ImMap :: saveBYUCallback()
{

  int imageIndex = BYUImageList->value();
  bool noBYUSelected = true;

  if (imageIndex == 0)
  {
    fl_alert("You have to select an image first.");
    return;
  }

  for (unsigned int rowIndex = 0 ; rowIndex < BYUTableDisplay->getNbRows();
       rowIndex++)
  {
    if (BYUTableDisplay->row_selected(rowIndex) == 1)
    {
      char* BYUFile = fl_file_chooser(" Choose a BYU Name ","","");
      if (BYUFile == NULL)
      {
        return;
      }
      noBYUSelected = false;
      //get Image and ImAna index from the row
      int selectedImage = BYUTableDisplay->getImageIndex(rowIndex);
      int selectedImAna = BYUTableDisplay->getImAnaIndex(rowIndex);
      _updateStatusBuffer("Writing BYU surface...");
      _imAnaVectors[selectedImage][selectedImAna].surface.writeBYU(BYUFile);
      _updateStatusBuffer("Writing BYU surface...DONE");  
    }

  }
  if (noBYUSelected)
  {
    fl_alert("Please select a surface first!");
    return;
  }

}

/////////////////////////////
// saveAnastructCallback() //
////////////////////////////

void ImMap :: saveAnastructCallback(){

  int imageIndex = BYUImageList->value();
  bool noBYUSelected = true;

  if (imageIndex  == 0)
  {
    fl_alert("Please select an image first!");
    return;
  }
  for (unsigned int rowIndex = 0 ; rowIndex < BYUTableDisplay->getNbRows() ; rowIndex++)
  {
    if (BYUTableDisplay->row_selected(rowIndex) == 1)
    {
      char* AnastructFile = fl_file_chooser(" Choose a anastruct Name ","","");
      if ( AnastructFile == NULL)
      {
        return;
      }
      //get Image and ImAna index from the row
          int selectedImage = BYUTableDisplay->getImageIndex(rowIndex);
          int selectedImAna = BYUTableDisplay->getImAnaIndex(rowIndex);
      _updateStatusBuffer("Writing Anastruct...");
      Anastruct savedAnastruct(_imAnaVectors[selectedImage][selectedImAna].anastruct);
      AnastructUtils::imageIndexToWorldXY(savedAnastruct,
        _loadedImages[selectedImage]->getOrigin()[0],
        _loadedImages[selectedImage]->getOrigin()[1],
        _loadedImages[selectedImage]->getSpacing()[0],
        _loadedImages[selectedImage]->getSpacing()[1]);
      AnastructUtils::writePLUNCAnastruct(savedAnastruct,AnastructFile);

      _updateStatusBuffer("Writing Anastruct...DONE");  
      noBYUSelected = false;
    }
  }
  if (noBYUSelected)
  {
    fl_alert("Please select a surface first!");
    return;
  }
}


/////////////////////////////////
// getVolume(const Surface&) //
////////////////////////////////
double
ImMap
::getVolume(const Surface& s)
{
  double volume = SurfaceUtils::computeVolume(s);
  return volume;
}


//////////////////////////////////
// getCentroid(const Surface&) //
/////////////////////////////////
Vector3D<double>
ImMap
::getCentroid(const Surface& s)
{ 
  Vector3D<double> centroid = SurfaceUtils::computeCentroid(s);
  return centroid;  
}


///////////////////////
// infoBYUCallback() //
///////////////////////
void
ImMap
::infoBYUCallback()
{
  Surface surfaceForInfo;
  std::string surfacename;
  bool noBYUSelected = true;
  int nbBYUselected=0;
  int rowIndex = BYUTableDisplay->getNbRows()-1;
  while(rowIndex>=0)
    {
      if (BYUTableDisplay->row_selected(rowIndex) == 1)
        {
          unsigned int index = rowIndex;
          int selectedImage = BYUTableDisplay->getImageIndex(index);
          int selectedImAna = BYUTableDisplay->getImAnaIndex(index);
          surfaceForInfo=_imAnaVectors[selectedImage][selectedImAna].surface;
          surfacename=_imAnaVectors[selectedImage][selectedImAna].anastruct.label;
          noBYUSelected = false;
          nbBYUselected++;
        }
      rowIndex--;
    }
  if ((noBYUSelected)||(nbBYUselected>1))
    {
      fl_alert("Please select a surface first (only one)!");
      return;
    }

  //
  // compute surface information
  //
  double volume = -getVolume(surfaceForInfo);

  Vector3D<double> centroid = getCentroid(surfaceForInfo);  
  Vector3D<double> min;
  Vector3D<double> max;
  surfaceForInfo.getAxisExtrema(min, max);  
  int facets = surfaceForInfo.numFacets();
  int vertices = surfaceForInfo.numVertices();

  surfacename_value->value(surfacename.c_str());

  std::ostringstream oss;
  oss << volume;
  volumeTextBox1->value(oss.str().c_str());

  oss.str(""); // clear oss
  oss << "(x,y,z) = (" << centroid.x << ", " << centroid.y << ", " << centroid.z << ")";
  centroidTextBox1->value(oss.str().c_str());

  oss.str(""); 
  oss << "(minx, miny, minz) = (" << min.x << ", " << min.y << ", " << min.z << ")";
  minimaTextBox1->value(oss.str().c_str());

  oss.str(""); 
  oss << "(maxx, maxy, maxz) = (" << max.x << ", " << max.y << ", " << max.z << ")";
  maximaTextBox1->value(oss.str().c_str());

  oss.str("");
  oss << facets;
  facetsTextBox1->value(oss.str().c_str());

  oss.str("");
  oss << vertices;
  verticesTextBox1->value(oss.str().c_str());

  BYUinfowindow->show();
  while ( BYUinfowindow->shown()) Fl::wait();
}


////////////////////////////
// closeinfoBYUCallback() //
///////////////////////////
void
ImMap
::closeinfoBYUCallback()
{
  BYUinfowindow->hide();
}

////////////////////////////////////////////////////////////////////////////////////
//void
// ImMap
// ::SaveBYUInfoCallback()
// {
//   char* filename = fl_file_chooser("Choose a name to save the datas...","","", 0);
//   if( filename == (char*)0 ) 
//     {
//       std::cout << "save datas file chooser failed ";
//       std::cout << std::endl;
//       return;
//     }
//   SaveBYUInfo(filename)
// }

// void
// ImMap
// ::SaveBYUInfo(char* saveBYUinfofilename)
//{
//   try
//   {
//     std::ofstream outputASCII(saveBYUinfoFilename,ofstream::out|ofstream::app);

//     if (outputASCII.fail())
//       {
//      throw std::runtime_error("failed to open file for ascii write");
//       }    

//     outputASCII <<surfacename_value->value()<< ",";    
//     outputASCII <<volumeTextBox1->value() << ",";
//     outputASCII <<verticesTextBox1->value() << ",";
//     outputASCII <<facetsTextBox1->value() << ",";
//     outputASCII <<centroidTextBox1->value() << ",";
//     outputASCII <<minimaTextBox1->value() << ",";  
//     outputASCII << "\n";    

//     if (outputASCII.fail())
//     {
//       throw std::runtime_error("ofstream failed writing ascii header");
//     }
//     outputASCII.close();    
//}


///////////////////////////
// refineBYUCallback()  //
/////////////////////////
void
ImMap
::refineBYUCallback()
{   
  Surface surface;
  int imageIndex = BYUImageList->value();
  bool noBYUSelected = true;
  int rowIndex = BYUTableDisplay->getNbRows()-1;
  std::string refinename;
  while(rowIndex>=0)
    {
      if (BYUTableDisplay->row_selected(rowIndex) == 1)
        {
          unsigned int index = rowIndex;
          int selectedImage = BYUTableDisplay->getImageIndex(index);
          int selectedImAna = BYUTableDisplay->getImAnaIndex(index);
          surface=_imAnaVectors[selectedImage][selectedImAna].surface;
          refinename=_imAnaVectors[selectedImage][selectedImAna].anastruct.label;
          noBYUSelected = false;
        }
      rowIndex--;
    }
  if (noBYUSelected)
    {
      fl_alert("Please select a surface first!");
      return;
    }

  SurfaceUtils::refineSurface(surface);   
  std::string refineBYUname;
  refineBYUname=refinename+"_after_refine";
  Anastruct loadedAnastruct;
  _createAnastruct(surface,
                   _loadedImages[imageIndex]->getOrigin(),
                   _loadedImages[imageIndex]->getSpacing(),
                   refineBYUname ,
                   loadedAnastruct);

  //add the new ImAna in the list
  ImAna loadedImAna(loadedAnastruct, surface);
  _imAnaVectors[imageIndex].push_back(loadedImAna);

  if ( imageIndex == imageChoice->value() ||
       imageIndex == overlayChoice->value() )
    {
      _updateContours();
      _updateSurfaces();
      _redrawImageWindows();      
    }

  displayBYUListCallback();
}


/////////////////////////////////////
// createOToIImageButtonCallback() //
/////////////////////////////////////

void
ImMap
::createOToIImageButtonCallback()
{
  if (createOToIImageButton->value())
    {
      createOToISurfacesButton->activate();
    }
  else
    {
      createOToISurfacesButton->deactivate();
    }
}


void
ImMap
::_addImageToList(ImagePointer imagePointer){

  std::cout << "_addImageToList()" << std::endl;

  // add image to list
  _loadedImages.push_back(imagePointer);
  _totalImagesLoaded++;

  std::ostringstream newImageNameOstr;

  newImageNameOstr << _imageNames.back().c_str()
                   << "(" << _totalImagesLoaded << ")";
  std::string NewImageNameStr = newImageNameOstr.str();
  const char* NewImageName = NewImageNameStr.c_str();

  // add filename to selection lists
  imageChoice->add(NewImageName);
  overlayChoice->add(NewImageName);
  imageSaveChoice->add(NewImageName);
  BYUImageList->add(NewImageName);
  BYUImageList->select(BYUImageList->size());

  // create and set the histogram
  HistogramDat histogram;
  float mini,maxi;
  mini=maxi=0; // make compiler happy
  Array3DUtils::getMinMax(*imagePointer,mini,maxi);
  std::cerr << "min: " << mini << ", max: " << maxi << std::endl;
  
  histogram.setAbsoluteMinMax(mini, maxi);
  _presetIntensity[0].relativeMin = mini;
  _presetIntensity[0].relativeMax = maxi;
  std::vector<double> histo = histogram.getHistogram();

  VoxelType *imageBufferPtr = imagePointer->getDataPointer();

  unsigned int numVoxels = imagePointer->getNumElements();
  if(maxi>2)
  {
  for (unsigned int i = 0; i < numVoxels; ++i) 
    {
      ++histo[static_cast<double>(*imageBufferPtr++ - mini)];
    }
  }
  else
  {
    for (unsigned int i = 0; i < numVoxels; ++i)
    {
      ++histo[static_cast<int>((*imageBufferPtr++ - mini) * 100)];
    }
  }

  histogram.setHistogram(histo);

  // add image to list
  _histograms.push_back(histogram);

  surface3DWindow->updateDisplay();

  std::cerr << "done" << std::endl;
}

///////////////////////////
// _redrawImageWindows() //
///////////////////////////

void
ImMap
::_redrawImageWindows()
{
  axial2DWindow->invalidate();
  axial2DWindow->redraw();

  coronal2DWindow->invalidate();
  coronal2DWindow->redraw();

  sagittal2DWindow->invalidate();
  sagittal2DWindow->redraw();
}

void ImMap::
_updateTransformDisplay()
{
  std::ostringstream matrixStream;
  matrixStream << registrationTransform << std::endl;
  matrixDisplay->insert( matrixStream.str().c_str() );
}

/////////////////////////
// _updateImagePanel() //
/////////////////////////

void
ImMap
::_updateImagePanel()
{
  // intensity windowing
  if (axial2DWindow->haveValidImage())
    {
      imageValMin->range(axial2DWindow->getImageDataMin(),
                         axial2DWindow->getImageDataMax());
      imageValMax->range(axial2DWindow->getImageDataMin(),
      axial2DWindow->getImageDataMax());
  }
  if (axial2DWindow->haveValidOverlay())
  {
    overlayValMin->range(axial2DWindow->getOverlayDataMin(),
      axial2DWindow->getOverlayDataMax());
    overlayValMax->range(axial2DWindow->getOverlayDataMin(),
      axial2DWindow->getOverlayDataMax());
  }

  // To set the sliders to their previous value
  int indexImageHisto=imageChoice->value();
  imageValMin->value(_histograms[indexImageHisto].getRelativeMin());
  imageValMax->value(_histograms[indexImageHisto].getRelativeMax());

  imageValMinMaxCallback();

  int indexOverlayHisto=overlayChoice->value();

  overlayValMin->value(_histograms[indexOverlayHisto].getRelativeMin());
  overlayValMax->value(_histograms[indexOverlayHisto].getRelativeMax());

  overlayValMinMaxCallback();


  imageZoomVal->value(axial2DWindow->getWindowZoom());

  overlayAlphaSlider->value(axial2DWindow->getOverlayOpacity());
}

///////////////////////
// _updateROIPanel() //
///////////////////////

void
ImMap
::_updateROIPanel()
{
  if (axial2DWindow->haveValidImage() || axial2DWindow->haveValidOverlay())
  {
    roiStartX->range(0, axial2DWindow->getImageDimX() - 1);
    roiStartY->range(0, axial2DWindow->getImageDimY() - 1);
    roiStartZ->range(0, axial2DWindow->getImageDimZ() - 1);
    roiStopX->range(0, axial2DWindow->getImageDimX() - 1);
    roiStopY->range(0, axial2DWindow->getImageDimY() - 1);
    roiStopZ->range(0, axial2DWindow->getImageDimZ() - 1);
  }
}

//////////////////////
// _updateROIInfo() //
//////////////////////

void
ImMap
::_updateROIInfo()
{
  if (axial2DWindow->haveValidImage() || axial2DWindow->haveValidOverlay())
  {
    roiSizeX->value(roiStopX->value() - roiStartX->value() + 1);
    roiSizeY->value(roiStopY->value() - roiStartY->value() + 1);
    roiSizeZ->value(roiStopZ->value() - roiStartZ->value() + 1);
    roiAreaXY->value(int(roiSizeX->value()*roiSizeY->value()));
    roiAreaXZ->value(int(roiSizeX->value()*roiSizeZ->value()));
    roiAreaYZ->value(int(roiSizeY->value()*roiSizeZ->value()));
    std::ostringstream oss;
    oss << int(roiSizeX->value()*roiSizeY->value()*roiSizeZ->value());
    roiVolumeXYZoutput->value(oss.str().c_str());
  }
}

//////////////////////////////
// _updatePositionSliders() //
//////////////////////////////

void
ImMap
::_updatePositionSliders()
{
  axialScrollBar->range(0,axial2DWindow->getNumSlices() - 1);
  coronalScrollBar->range(0,coronal2DWindow->getNumSlices() - 1);
  sagittalScrollBar->range(0,sagittal2DWindow->getNumSlices() - 1);

  axialScrollBar->Fl_Valuator::value(axial2DWindow->getSliceIndex());
  coronalScrollBar->Fl_Valuator::value(coronal2DWindow->getSliceIndex());
  sagittalScrollBar->Fl_Valuator::value(sagittal2DWindow->getSliceIndex());  

  // this makes arrows increment by one
  axialScrollBar->linesize(1);
  coronalScrollBar->linesize(1);
  sagittalScrollBar->linesize(1);
}

///////////////////////////////
// _centerImage(int,int,int) //
///////////////////////////////

void
ImMap
::_centerImage(float xIndex, float yIndex, float zIndex)
{
  axial2DWindow->setDataCenter(xIndex, yIndex, zIndex);
  coronal2DWindow->setDataCenter(xIndex, yIndex, zIndex);
  sagittal2DWindow->setDataCenter(xIndex, yIndex, zIndex);

  axial2DWindow->update();
  coronal2DWindow->update();
  sagittal2DWindow->update();

  axialScrollBar->Fl_Valuator::value((int)zIndex);
  coronalScrollBar->Fl_Valuator::value((int)yIndex);
  sagittalScrollBar->Fl_Valuator::value((int)xIndex);  
}

//////////////////////////////////////
// _updateStatusBuffer(const char*) //
//////////////////////////////////////

void
ImMap
::_updateStatusBuffer(const char *message)
{
  std::cerr << "{ " << message << " }" << std::endl;
  statusText->value(message);
  Fl::flush(); 
}


/////////////////////////
// DicomLoaderCallback //
/////////////////////////

void ImMap
::DicomLoaderCallback()
{

  // delete all the varaiables
  DicomFileSelector->clear();
  DicomFileSelected->clear();
  filenames.clear();
  filenamesSelected.clear();
  ZpositionsSelected.clear();
  Zpositions.clear();

  DirectoryName = DicomLoaderWindow->getFileName();
  //display the directory
  std :: string dirRef(DirectoryName);
  int pos = dirRef.find_last_of("/");
  dirRef.erase(dirRef.begin() + pos+1, dirRef.end() );
  Directory_Name->value(dirRef.c_str());

  _getFileListDicom((char * )DirectoryName.c_str());
  for (int i = 0 ; i < num_files ; i ++ ){

    char tmpFilePath[1024];
    sprintf(tmpFilePath, "%s",infile[i]);

    std :: string FileNameStr(tmpFilePath);

    size_t loc = FileNameStr.find(';');
    if (loc!= std::string::npos){
      FileNameStr.insert(FileNameStr.begin()+loc,'\\');
    }

    DICOMimage tmpDicomImage;
    tmpDicomImage.OpenDICOMFile((char *)FileNameStr.c_str());
    tmpDicomImage.SelectImage(0);

    Zpositions.push_back(tmpDicomImage.get_Zoffset());
    filenames.push_back(infile[i]);

    _addDicomFileSelector(infile[i],tmpDicomImage.get_Zoffset());

    PreviewWindow->SetFilename(infile[0]);
  }
  DicomLoaderWindow->show();
}

///////////////////////////
// _addDicomFileSelector //
///////////////////////////

void  ImMap
::_addDicomFileSelector(char * SliceName, float Zposition)
{

  char tmpFilenameZposition[512];
  sprintf(tmpFilenameZposition,"%s     z = %.1f",SliceName, Zposition);

  std::string DisplayFilename(tmpFilenameZposition); 

  int pos = DisplayFilename.find_first_of(".");
  DisplayFilename.erase(DisplayFilename.begin(),DisplayFilename.begin() + pos + 1 );

  DicomFileSelector->add(DisplayFilename.c_str());

}

///////////////////////////
// _addDicomFileSelected //
///////////////////////////

void  ImMap
::_addDicomFileSelected(char * SliceName, float Zposition)
{

  char tmpFilenameZposition[512];
  sprintf(tmpFilenameZposition,"%s     z = %.1f",SliceName, Zposition);

  std::string DisplayFilename(tmpFilenameZposition); 

  int pos = DisplayFilename.find_first_of(".");
  DisplayFilename.erase(DisplayFilename.begin(),
                        DisplayFilename.begin() + pos + 1 );

  DicomFileSelected->add(DisplayFilename.c_str());

}

///////////////////////
// _getFileListDicom //
///////////////////////

void ImMap
::_getFileListDicom(char *basename)
{

  char  cmd[1000];
  char  tmpfile[1024];
  int fileCount=0;

  infile = NULL;
#ifdef WIN32
  char *p = basename;
  while (*p){
    if (*p == '/') *p = '\\';
    *p++;
  }
  strcpy(tmpfile, "filelist.dispimage.txt");
  // 'dir /b /s' is bare-format (no dir header info) and subdir's
  // (forces dirname prepended to filename)
  sprintf(cmd, "dir /b /s /O:N \"%s\" > %s\n", basename, tmpfile);
  printf("_getFileList::%s\n",cmd);
#else
  tmpnam(tmpfile);
  //  mkstemp(tmpfile);
  sprintf(cmd, "/bin/ls -1 %s > %s", basename, tmpfile);
#endif
#ifdef WIN32
#else
  system(cmd);
#endif
  std::ifstream fileListInput(tmpfile);
  if (fileListInput.fail())
  {
    return ;
  }

  while (!fileListInput.eof())
  {
    infile = (char **)realloc((char*)infile, (fileCount+1)*sizeof(char *));
    infile[fileCount] = (char*) malloc(1000);
    fileListInput.getline(infile[fileCount],1000);
    if (fileListInput.gcount() > 0) fileCount++;
  }
  fileListInput.close();


  remove(tmpfile);

  /* strip newline from end */
  num_files = fileCount;


}

/////////////////////////
// SelectFilesCallback //
/////////////////////////

void ImMap
::SelectFilesCallback()
{

  bool alreadySelected ;

  for (unsigned int i = 0 ; i < filenames.size() ; i ++){

    alreadySelected =  false;

    if (DicomFileSelector->selected(i+1)){

      //Check if the file is already selected
      for (unsigned int j = 0 ; j < filenamesSelected.size() ; j ++){
        if (filenamesSelected[j]==filenames[i]){
          alreadySelected = true;
        }
      }

      if (!alreadySelected){
        filenamesSelected.push_back(filenames[i]);
        ZpositionsSelected.push_back(Zpositions[i]);
        _addDicomFileSelected((char *) filenames[i].c_str(),Zpositions[i] );
      }

    } 
  }

  // We deselect the previous selected files
  DicomFileSelector->deselect();

}

///////////////////////////
// DeselectFilesCallback //
///////////////////////////

void ImMap
::DeselectFilesCallback()
{


  for (unsigned int i = 0 ; i < filenamesSelected.size() ; i ++){
    if (DicomFileSelected->selected(i+1)){
      DicomFileSelected->remove(i+1);
      ZpositionsSelected.erase(ZpositionsSelected.begin()+i);
      filenamesSelected.erase(filenamesSelected.begin()+i);
      i--;
    } 
  }

}

///////////////////////////
// SliceSelectedCallback //
///////////////////////////

void ImMap 
::SliceSelectedCallback()
{
  int val = ((DicomFileSelector->value())-1);
  PreviewWindow->SetFilename((char *) filenames[val].c_str());
  PreviewWindow->redraw();
}

///////////////
// _sortList //
///////////////

void  ImMap
:: _sortList(int val)
{

  float min ;
  int minPos = 0 ;


  if (val == 1){

    if (filenames.size()==0){
      return ;
    } 

    std :: vector<float> TempZpositions(Zpositions);
    std :: vector<std :: string> Tempfilenames(filenames);

    filenames.clear();
    Zpositions.clear();

    do {
      min = TempZpositions[minPos];
      for (unsigned int i = 0 ; i < TempZpositions.size() ; i ++){
        if (TempZpositions[i]<=min) {
          min=TempZpositions[i];
          minPos=i;
        }
      }
      if (min != 10000){

        filenames.push_back(Tempfilenames[minPos]);
        Zpositions.push_back(TempZpositions[minPos]);
        TempZpositions[minPos]=10000;
      }
    } 
    while ( min != 10000);

  }

  else{

    if (filenamesSelected.size()==0){
      return ;
    } 

    std :: vector<float> TempZpositions(ZpositionsSelected);
    std :: vector<std :: string> Tempfilenames(filenamesSelected);

    filenamesSelected.clear();
    ZpositionsSelected.clear();


    do {

      min = TempZpositions[minPos];
      for (unsigned int i = 0 ; i < TempZpositions.size() ; i ++){
        if (TempZpositions[i]<=min) {
          min=TempZpositions[i];
          minPos=i;
        }
      }
      if (min != 10000){
        filenamesSelected.push_back(Tempfilenames[minPos]);
        ZpositionsSelected.push_back(TempZpositions[minPos]);
        TempZpositions[minPos]=10000;
      }
    } 
    while ( min != 10000);
  }


}

///////////////////////////
// SortSelectionCallback //
///////////////////////////

void ImMap
:: SortSelectionCallback(int val )
{


  _sortList(val);


  // To display the new list

  if (val == 1){

    DicomFileSelector->clear();
    for (unsigned int i = 0 ; i < filenames.size() ; i ++){
      _addDicomFileSelector((char *) filenames[i].c_str(), Zpositions[i] );
    }

  }
  else{

    DicomFileSelected->clear();
    for (unsigned int i = 0 ; i < filenamesSelected.size() ; i ++){
      _addDicomFileSelected( (char *)filenamesSelected[i].c_str() , 
        ZpositionsSelected[i]);
    }
  }

}

////////////////////////
// InvertSortCallback //
////////////////////////

void ImMap
:: InvertSortCallback(int val)
{

  int nbElement;

  // 1. We sort

  _sortList(val);

  if (val == 1){

    nbElement = filenames.size();

    // 2. We invert the list

    std :: vector<std :: string> tempFilenames(filenames);
    std :: vector<float> tempZpositions(Zpositions);
    for (int i = 0 ; i < nbElement ; i ++){
      filenames[i]=tempFilenames[nbElement-1-i];
      Zpositions[i]=tempZpositions[nbElement-1-i];
    }

    // 3. We display

    DicomFileSelector->clear();
    for (unsigned int j = 0 ; j < filenames.size() ; j ++){
      _addDicomFileSelector( (char *)filenames[j].c_str(),
        Zpositions[j]);
    }

  }
  else {

    nbElement = filenamesSelected.size();

    //2.(bis) We invert the list

    std :: vector<std :: string> tempFilenamesSelected(filenamesSelected);
    std :: vector<float> tempZpositionsSelected(ZpositionsSelected);
    for (int i = 0 ; i < nbElement ; i ++){
      filenamesSelected[i]=tempFilenamesSelected[nbElement-1-i];
      ZpositionsSelected[i]=tempZpositionsSelected[nbElement-1-i];
    }

    // 3.(bis) We display

    DicomFileSelected->clear();
    for (unsigned int k = 0 ; k < filenamesSelected.size() ; k ++){
      _addDicomFileSelected( (char *)filenamesSelected[k].c_str(),
        ZpositionsSelected[k]);
    }

  }

}

////////////////////
// AddAllCallback //
////////////////////

void  ImMap
:: AddAllCallback()
{

  for (unsigned int i = 0 ; i < filenames.size(); i ++){
    DicomFileSelector->select(i+1);  
  }
  SelectFilesCallback();

}

///////////////////////
// RemoveAllCallback //
///////////////////////

void  ImMap
:: RemoveAllCallback()
{

  for (unsigned int i = 0 ; i < filenames.size(); i ++){
    DicomFileSelected->select(i+1);  
  }
  DeselectFilesCallback();

}


/////////////////////////
// CreateImageCallback //
/////////////////////////

void  ImMap
::CreateImageCallback()
{

  DICOMimage dicomImage;

  int numVol = dicomImage.OpenDICOMSelectedFiles(DirectoryName,filenamesSelected);
  std :: cout <<"numVol "<<numVol<<std :: endl;

  for (int i = 0 ; i < numVol; i ++)
  {
    dicomImage.SelectImage(i);
    std :: cout<<"Z Size : "<<dicomImage.getsizeZ()<<std :: endl;
    std :: cout<<"Z PixelSize "<<dicomImage.PixsizeZ()<<std :: endl;
  }
  dicomWindow->setDicom(dicomImage);
  DicomLoaderWindow->hide();
  if (numVol < 1)
  {
    return;
  }
  else 
  {
    choiceDICOM->clear();
    char temp[1024]="Dicom Selected";
    choiceDICOM->add((const char *)temp);
    choiceDICOM->value(0);
    DICOMCallback();
    // delete all the varaiables
    DicomFileSelector->clear();
    DicomFileSelected->clear();
    filenames.clear();
    filenamesSelected.clear();
    ZpositionsSelected.clear();
    Zpositions.clear();
  } 

}


////////////////////////////
// _loadImageDICOM(char*) //
////////////////////////////

ImMap::ImagePointer
ImMap
::_loadImageDICOM(char *fileName)
{
  ImageIO imageIO;
  DICOMimage DICOM_;
  ImagePointer image3D;
  std :: string name(fileName);

  imageIO.LoadDicom(name);

  int NumDicom = imageIO.get_nb_DICOM();
  DICOM_=imageIO.get_DICOM();

  if( NumDicom < 1 ) 
  {
    std::cerr << "[Dicom::Load] no volumes loaded!" << std::endl;
  }
  std::cerr << "[Dicom::Load] loaded " << NumDicom << " volumes." 
    << std::endl;

  dicomWindow->setDicom(DICOM_);
  // Check if the Preload was successful
  if (NumDicom < 1)
  {
    return 0;
  }
  else 
  {
    choiceDICOM->clear();

    for (int i=0; i < NumDicom ; i++)
    {
      char temp[1024];
      DICOM_.SelectImage(i);
      sprintf(temp,"%s %d",DICOM_.get_name(),i);
      choiceDICOM->add((const char *)temp);
    }
    DicomLoaderWindow->setFileName(fileName);
    DicomLoaderWindow->setDicom(DICOM_);
    choiceDICOM->value(0);
    DICOMCallback();

    dicomWindow->show(); 
    while (dicomWindow->shown()) Fl::wait();
    DICOM_=dicomWindow->getDicom();

    // load image with imageIO
    image3D = new ImageType;
    imageIO.setDICOM(DICOM_);

    //check the spacing

    bool uniformSpacing = imageIO.CheckSpacingSelectedDicom(choiceDICOM->value());
    DICOM_.SelectImage(choiceDICOM->value());
    if (uniformSpacing == true)
    {
      imageIO.LoadSelectedDicom(*image3D);
    }
    else{
      bool sameSign;
      do{
        sameSign=true;
        NewSpacingWindow->show(); 
        float previousSpacing =  DICOM_.PixsizeZ();
        previousSpacingOutput->value(previousSpacing);
        while (NewSpacingWindow->shown()) Fl::wait();
        if (((newSpacing->value())*(previousSpacing))<=0)
        {
          sameSign = false;
          sameSignOutput->value("new and previous spacing must have the same sign");
        }
      }while(sameSign==false);
      sameSignOutput->value("");
      imageIO.LoadSelectedDicom(*image3D,newSpacing->value());
    }

    // add name of the dicom to the list
    _imageNames.push_back(DICOM_.get_name());
    _imageFullFileNames.push_back(fileName);    
    std::cout<<"Name : "<<DICOM_.get_name()<<std::endl;

    std::vector<ImAna> vectorImAna;

    // if selected, we add the anastruct of this dicom to the list
    if (LoadContourButton->value()){

      DICOMcontour dcont(DICOM_.getObject());
      dcont.OpenDICOMFile();

      int totalAnastruct = dcont.get_num_anastruct();
      for (int nbAnastruct = 0 ; 
      nbAnastruct < totalAnastruct ; 
      ++nbAnastruct)
      {
        //
        // add the anastruct to the list
        //
        Anastruct currentAnastruct(dcont.get_AnaStruct(nbAnastruct));
        //      vectorAnastruct.push_back(currentAnastruct);

        std::cerr << "#### got anastruct: "
          << currentAnastruct.label 
          << std::endl
          << "\t num contours: " 
          << currentAnastruct.contours.size()
          << std::endl;

        for (unsigned int i = 0; i < currentAnastruct.contours.size(); ++i)
        {
          std::cerr << "\t num vertices: " 
            << currentAnastruct.contours[i].vertices.size()
            << std::endl;
        }

        //
        // add a surface to the list
        //
        AnastructUtils::capContour(currentAnastruct);
        Anastruct worldCoordsAnastruct(currentAnastruct);

        AnastructUtils::imageIndexToWorldXY(worldCoordsAnastruct,
          image3D->getOrigin().x,
          image3D->getOrigin().y,
          image3D->getSpacing().x,
          image3D->getSpacing().y);
        Surface tmpSurface;
        AnastructUtils::anastructToSurfacePowerCrust(worldCoordsAnastruct, 
          tmpSurface);
        _createAnastruct(tmpSurface,
          image3D->getOrigin(),
          image3D->getSpacing(),
          currentAnastruct.label,
          currentAnastruct);
        ImAna currentImAna(currentAnastruct, tmpSurface);
        vectorImAna.push_back(currentImAna);       
      }
    }

    _imAnaVectors.push_back(vectorImAna);
    }
    return image3D;
}

////////////////////////////
// _loadOtherImage(char*) //
////////////////////////////

ImMap::ImagePointer
ImMap
::_loadOtherImage(char *fileName)
{
  // load image with imageIO
  ImagePointer image3D;
  image3D = new ImageType;
  ImageIO imageIO;
  ImageIO::ImageType imageType = imageIO.GuessImageFormat(fileName);

  // add name of the planIm to the list
  std::string name(fileName); 

  //
  // first try to load an image using itk file readers
  //
  if (ApplicationUtils::ITKHasImageFileReader(fileName))
  {
    ApplicationUtils::LoadImageITK(fileName, *image3D);
    int pos = name.find_last_of("/");
    name.erase(name.begin(), name.begin() + pos + 1 );
    pos = name.find_last_of(".");
    name.erase(name.begin() + pos ,name.end());
    _imageNames.push_back(name.c_str());
    _imageFullFileNames.push_back(fileName); 
    std::cout<<"FullFileName : "<<fileName<<std::endl; 
    std::cout<<"Name : "<<name.c_str()<<std::endl; 
  }
  //
  // next try to use one of the ImageIO file readers
  //
  else if (imageType == ImageIO::planIM ||
           imageType == ImageIO::PLUNC_dose_grid )
  {
    imageIO.LoadThisImage(fileName, *image3D);
    size_t pos = name.find_last_of("/");
    if (pos != string::npos)
    {
      name.erase(name.begin() + pos, name.end() );
    }
    else
    {
      char cwdBuffer[1024];
      bool bufferIsBigEnough = (getcwd(cwdBuffer, 1024) != NULL);
      if (bufferIsBigEnough)
      {
        name = std::string(cwdBuffer);
      }
      else
      {
        name = std::string(fileName);
      }
    }
    pos = name.find_last_of("/");
    name.erase(name.begin(),name.begin() + pos + 1);
    _imageNames.push_back(name);
    _imageFullFileNames.push_back(fileName);    
    std::cout << "Name : " << name.c_str() << std::endl;
    std::cout << "_imageFullFileName : " << fileName << std::endl;
  } 
  // finally, give the user the opportunity to enter the dimensions,
  // origin, spacing, etc, of the image, and save the results into a
  // .mhd file.  Then load that file, which loads the image.
  else 
  {
    _header_saved=false;    
    std::string Imagename_str( fileName);   

    std::string Name (fileName);

    string::size_type pos = Imagename_str.find_last_of("/");
    Imagename_str.erase(Imagename_str.begin()+pos+1  ,Imagename_str.end());  
    Name.erase(Name.begin(), Name.begin() +pos+1);    

    pos = Name.find_last_of(".");
    if (pos != string::npos)
    {    
      Name.erase(Name.begin() + pos, Name.end());      
      Name=Name+".mhd";
    }
    else
      Name= Name+".mhd";

    Imagename_str= Imagename_str+Name;   

    //to display in the fltk window the name of the image to save
    string::size_type len;
    len = name.length(); 
    pos =  name.rfind("/",len);
    name.erase ( name.begin(), name.begin()+pos+1);
    const char* name_image=name.c_str();
    ElementDataFile_value->value(name_image);


    defineheader_windows->show();
    while (defineheader_windows->shown()) Fl::wait();
    if(_header_saved)
    {
      const char* filename=  Imagename_str.c_str();        
      //we copy all the data in a .mhd
      FILE *file;

      file=fopen(filename,"w+");
      if (file==NULL) perror ("Error opening file");

      fprintf(file,"ObjectType = ");
      fprintf(file,"%s\n", _ObjectType.c_str());

      fprintf(file,"NDims = ");
      fprintf(file,"%i\n",_NDims);

      fprintf(file,"BinaryData = ");
      fprintf(file,"%s\n",_BinaryData.c_str());

      fprintf(file,"BinaryDataByteOrderMSB = "); 
      fprintf(file,"%s\n",_BinaryDataByteOrderMSB.c_str());

      fprintf(file,"Offset = ");   
      fprintf(file,"%f %f %f\n",_Offset[0],_Offset[1],_Offset[2]); 

      fprintf(file,"ElementSpacing = ");    
      fprintf(file,"%f %f %f\n",_ElementSpacing[0],_ElementSpacing[1],_ElementSpacing[2]);   
      fprintf(file,"DimSize = ");      
      fprintf(file,"%i %i %i\n" ,_DimSize[0],_DimSize[1],_DimSize[2]);

      fprintf(file,"ElementType = ");
      fprintf(file,"%s\n",_ElementType.c_str());

      std::string ElementDataFilestr(ElementDataFile_value->value());   
      fprintf(file,"ElementDataFile = ");
      fprintf(file,"%s\n",_ElementDataFile.c_str());

      int fermeture=fclose(file);
      if (fermeture != 0)
        perror ("Error closing file");


      //  to have the value to update the vectors
      const char* NewFileName=Imagename_str.c_str(); 
      std::string NewImageName=Imagename_str;
      pos = NewImageName.find_last_of("/");
      NewImageName.erase( NewImageName.begin(),
                          NewImageName.begin() + pos + 1 );
      pos =  NewImageName.find_last_of(".");
      NewImageName.erase( NewImageName.begin() + pos , NewImageName.end());

      strcpy(fileName,NewFileName);                     
      _imageNames.push_back(NewImageName.c_str());
      _imageFullFileNames.push_back(fileName); 

      std::cout<<"FullFileName : "<<NewFileName<<std::endl; 
      std::cout<<"Name : "<<NewImageName.c_str()<<std::endl; 

      imageIO.LoadThisImage(fileName, *image3D);
    } else {
      _imageNames.push_back("NONE");
      _imageFullFileNames.push_back("NONE");
      imageIO.LoadThisImage(fileName, *image3D);
    }
  }

  //add an empty ImAna vector to the list
  _imAnaVectors.push_back(std::vector<ImAna>());


  std::cout << "Done." << std::endl;

  return image3D;
}


//////////////////////////
// saveheaderCallback()//
/////////////////////////

void
ImMap
::saveheaderCallback()
{
  _ObjectType=ObjectType_value->value();

  _NDims=(int)NDims_value->value();

  if( BinaryData_value->value()==1)  
    _BinaryData= "True";
  else
    _BinaryData="False";

  if( BinaryDataByteOrderMSB_value->value()==1)  
    _BinaryDataByteOrderMSB= "True";
  else
    _BinaryDataByteOrderMSB= "False";

  _Offset.push_back(Offset_X_value->value());
  _Offset.push_back(Offset_Y_value->value());
  _Offset.push_back(Offset_Z_value->value());

  _ElementSpacing.push_back( ElementSpacing_X_value->value());
  _ElementSpacing.push_back(ElementSpacing_Y_value->value());
  _ElementSpacing.push_back(ElementSpacing_Z_value->value());

  _DimSize.push_back((int)DimSize_X_value->value());
   _DimSize.push_back((int) DimSize_Y_value->value());
   _DimSize.push_back((int)DimSize_Z_value->value());

  if(ElementType_value->value()==0)
    _ElementType = "MET_CHAR";
  if(ElementType_value->value()==1)
    _ElementType = "MET_UCHAR";
  if(ElementType_value->value()==2)
    _ElementType = "MET_SHORT";
  if(ElementType_value->value()==3)
    _ElementType = "MET_USHORT";
  if(ElementType_value->value()==4)
    _ElementType = "MET_INT";
  if(ElementType_value->value()==5)
    _ElementType = "MET_UINT";
  if(ElementType_value->value()==6)
    _ElementType = "MET_DOUBLE";
  if(ElementType_value->value()==7)
    _ElementType =" MET_FLOAT";

_ElementDataFile=ElementDataFile_value->value();
 _header_saved=true;    
defineheader_windows->hide();

}

//////////////////////////////////////
// _saveImage(ImagePointer, char *) //
//////////////////////////////////////

void
ImMap
::_saveImage(ImagePointer image, char *fileName)
{
  std::cout << "Writing " << fileName << "...";
  ApplicationUtils::SaveImageITK(fileName, *image);
  std::cout<< "DONE"<<std::endl;
}

//////////////////////////////////////////
// _setROI(int, int, int,int, int, int) //
//////////////////////////////////////////

void
ImMap
::_setROI(int startX, int startY, int startZ,
          int stopX, int stopY, int stopZ)
{
  axial2DWindow->setROI(startX, startY, startZ, stopX, stopY, stopZ);
  coronal2DWindow->setROI(startX, startY, startZ, stopX, stopY, stopZ);
  sagittal2DWindow->setROI(startX, startY, startZ, stopX, stopY, stopZ);
  _redrawImageWindows();

  roiStartX->value(startX);
  roiStartY->value(startY);
  roiStartZ->value(startZ);
  roiStopX->value(stopX);
  roiStopY->value(stopY);
  roiStopZ->value(stopZ);

  // To set the wizard

  WroiStartX->value(startX);
  WroiStartY->value(startY);
  WroiStartZ->value(startZ);
  WroiStopX->value(stopX);
  WroiStopY->value(stopY);
  WroiStopZ->value(stopZ);

  WroiStartX2->value(startX);
  WroiStartY2->value(startY);
  WroiStartZ2->value(startZ);
  WroiStopX2->value(stopX);
  WroiStopY2->value(stopY);
  WroiStopZ2->value(stopZ);

  _updateROIInfo();
  _updateROIin3DWindow();
}

///////////////
// _getROI() //
///////////////

ImMap::ImageRegionType
ImMap
::_getROI()
{
  int startX = static_cast<int>(roiStartX->value());
  int startY = static_cast<int>(roiStartY->value());
  int startZ = static_cast<int>(roiStartZ->value());
  int stopX = static_cast<int>(roiStopX->value());
  int stopY = static_cast<int>(roiStopY->value());
  int stopZ = static_cast<int>(roiStopZ->value());

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

  ImageIndexType roiStartIndex(startX,startY,startZ);

  ImageSizeType roiSize(stopX - startX + 1,
    stopY - startY + 1 ,
    stopZ - startZ + 1);
  ImageRegionType roi(roiStartIndex, roiSize);
  return roi;
}

///////////////////////////
// _getPyramidSchedule() //
///////////////////////////

ImMap::ScheduleType
ImMap
::_getPyramidSchedule()
{

  // get values from gui
  std::vector< Vector3D<double> > pyramidScheduleAll(4);
  pyramidScheduleAll[0][0] = registrationPyramidX1->value();
  pyramidScheduleAll[0][1] = registrationPyramidY1->value();
  pyramidScheduleAll[0][2] = registrationPyramidZ1->value();  
  pyramidScheduleAll[1][0] = registrationPyramidX2->value();
  pyramidScheduleAll[1][1] = registrationPyramidY2->value();
  pyramidScheduleAll[1][2] = registrationPyramidZ2->value();  
  pyramidScheduleAll[2][0] = registrationPyramidX3->value();
  pyramidScheduleAll[2][1] = registrationPyramidY3->value();
  pyramidScheduleAll[2][2] = registrationPyramidZ3->value();  
  pyramidScheduleAll[3][0] = registrationPyramidX4->value();
  pyramidScheduleAll[3][1] = registrationPyramidY4->value();
  pyramidScheduleAll[3][2] = registrationPyramidZ4->value();  

  std::vector< Vector3D<double> > scheduleVec;

  // Make sure that each level (a) has no zeros, (b) is different from
  // the next coarser level, and (b) is consistently finer than the
  // next coarser level.
  for( int i = 3; i >= 0; --i ) {
    Vector3D<double>& scale = pyramidScheduleAll[i];
    if ( scale[0] > 0 && scale[1] > 0 && scale[2] > 0 &&
         (scheduleVec.empty() ||
          (scale != scheduleVec.back() &&
           scale[0] >= scheduleVec.back()[0] &&
           scale[1] >= scheduleVec.back()[1] &&
           scale[2] >= scheduleVec.back()[2]))) {
      scheduleVec.push_back(scale);
    }
  }
  if (scheduleVec.empty()) scheduleVec.push_back(Vector3D<double>(1,1,1));

  ScheduleType schedule( scheduleVec.size(), 3 );
  for (unsigned int i = 0; i < schedule.getSizeX(); ++i) {
    for (unsigned int j = 0; j < 3; ++j) {
      schedule(i, j) = scheduleVec[scheduleVec.size()-i-1][j];
    }
  }
  return schedule;
}

///////////////////////
// _updateSurfaces() //
///////////////////////

void
ImMap
::_updateSurfaces()
{
  bool surfaceWasLoaded = (surface3DWindow->getNumSurfaces() > 0);
  surface3DWindow->clearSurfaces();
  surface3DWindow->clearAnastructs();

  int imageIndex = imageChoice->value();
  int overlayIndex = overlayChoice->value();
  bool surfaceIsLoaded = false;
  unsigned int surfaceIndex = 0;

  // unused var//unsigned int anastructIndex = 0;


  for (surfaceIndex = 0; 
  surfaceIndex < _imAnaVectors[imageIndex].size(); 
  ++surfaceIndex)
  {
    Vector3D<double> imageColor(_imAnaVectors[imageIndex][surfaceIndex].color);
    surface3DWindow->addSurface(
      imageIndex,
      surfaceIndex,
      _imAnaVectors[imageIndex][surfaceIndex].surface,
      _imAnaVectors[imageIndex][surfaceIndex].visible,
      imageColor[0],imageColor[1],imageColor[2],
      (SurfaceViewWindow::SurfaceRepresentationType)_imAnaVectors[imageIndex][surfaceIndex].aspect,
      _imAnaVectors[imageIndex][surfaceIndex].opacity);

    // convert ana to world coords
    Anastruct tmpAna(_imAnaVectors[imageIndex][surfaceIndex].anastruct);
    AnastructUtils::imageIndexToWorldXY(tmpAna,
      _loadedImages[imageIndex]->getOrigin()[0],
      _loadedImages[imageIndex]->getOrigin()[1],
      _loadedImages[imageIndex]->getSpacing()[0],
      _loadedImages[imageIndex]->getSpacing()[1]);
    surface3DWindow->addAnastruct(tmpAna,
      imageColor[0],imageColor[1],imageColor[2]);
    surfaceIsLoaded = true;
  }     

  for (surfaceIndex = 0; 
  surfaceIndex < _imAnaVectors[overlayIndex].size(); 
  ++surfaceIndex)
  {
    Vector3D<double> overlayColor(_imAnaVectors[overlayIndex][surfaceIndex].color);
    std::cerr << "adding overlay surface to 3d display..." << std::endl;
    surface3DWindow->addSurface(overlayIndex,surfaceIndex,
      _imAnaVectors[overlayIndex][surfaceIndex].surface,
      _imAnaVectors[overlayIndex][surfaceIndex].visible,
      overlayColor[0],overlayColor[1],overlayColor[2],
      (SurfaceViewWindow::SurfaceRepresentationType)_imAnaVectors[overlayIndex][surfaceIndex].aspect,
      _imAnaVectors[overlayIndex][surfaceIndex].opacity);
    // convert ana to world coords
    Anastruct tmpAna(_imAnaVectors[overlayIndex][surfaceIndex].anastruct);
    AnastructUtils::imageIndexToWorldXY(tmpAna,
      _loadedImages[overlayIndex]->getOrigin()[0],
      _loadedImages[overlayIndex]->getOrigin()[1],
      _loadedImages[overlayIndex]->getSpacing()[0],
      _loadedImages[overlayIndex]->getSpacing()[1]);
    surface3DWindow->addAnastruct(tmpAna,
      overlayColor[0],overlayColor[1],overlayColor[2]);
    surfaceIsLoaded = true;
  }     

  if (surfaceIsLoaded && (!surfaceWasLoaded))
  {
    surface3DWindow->centerActors();
  }
  surface3DWindow->updateDisplay();
}  


////////////////////////////
// _updateROIin3DWindow() //
////////////////////////////

void
ImMap
::_updateROIin3DWindow()
{ 
  surface3DWindow->clearROI();
  Vector3D<double> start, stop;

  int imageIndex = imageChoice->value();
  if (imageIndex == 0)
  {
    imageIndex = overlayChoice->value();
    if (imageIndex == 0)
    {
      fl_alert("Please select an image first!");
      return;
    }
  }
  ImagePointer image = _loadedImages[imageIndex];
  Vector3D<double> spacing(image->getSpacing());
  Vector3D<double> origin(image->getOrigin());

  start.x = roiStartX->value()*spacing.x + origin.x;
  start.y = roiStartY->value()*spacing.y + origin.y;
  start.z = roiStartZ->value()*spacing.z + origin.z;

  stop.x = roiStopX->value()*spacing.x + origin.x;
  stop.y = roiStopY->value()*spacing.y + origin.y;
  stop.z = roiStopZ->value()*spacing.z + origin.z;

  if (start.x > stop.x)
  {
    double tmp = start.x;
    start.x = stop.x;
    stop.x = tmp;
  }
  if (start.y > stop.y)
  {
    double tmp = start.y;
    start.y = stop.y;
    stop.y = tmp;
  }
  if (start.z > stop.z)
  {
    double tmp = start.z;
    start.z = stop.z;
    stop.z = tmp;
  }
  surface3DWindow->updateROI(start,stop);
  surface3DWindow->centerActors();
  surface3DWindow->updateDisplay();
}


///////////////////////
// _updateContours() //
///////////////////////

void
ImMap
::_updateContours()
{
  int imageIndex = imageChoice->value();
  int overlayIndex = overlayChoice->value();
  axial2DWindow->clearImageAnastructs();
  axial2DWindow->clearOverlayAnastructs();
  sagittal2DWindow->clearImageAnastructs();
  sagittal2DWindow->clearOverlayAnastructs();
  coronal2DWindow->clearImageAnastructs();
  coronal2DWindow->clearOverlayAnastructs();

  // std::cerr << "imageIndex: " << imageIndex << std::endl;
  // std::cerr << "overlayIndex: " << overlayIndex << std::endl;

  unsigned int anastructIndex;
  for (anastructIndex = 0; 
       anastructIndex < _imAnaVectors[imageIndex].size(); 
       ++anastructIndex)
  {
      Vector3D<double> imageColor(_imAnaVectors[imageIndex][anastructIndex].color);

      axial2DWindow->
          addImageAnastruct(_imAnaVectors[imageIndex][anastructIndex].anastruct,
                            _imAnaVectors[imageIndex][anastructIndex].visible, 
                            imageColor[0],imageColor[1],imageColor[2]);

      sagittal2DWindow->
          addImageAnastruct(_imAnaVectors[imageIndex][anastructIndex].anastruct,
                            _imAnaVectors[imageIndex][anastructIndex].visible,
                            imageColor[0],imageColor[1],imageColor[2]);

      coronal2DWindow->
          addImageAnastruct(_imAnaVectors[imageIndex][anastructIndex].anastruct,
                            _imAnaVectors[imageIndex][anastructIndex].visible,
                            imageColor[0],imageColor[1],imageColor[2]);
  }   


  for (anastructIndex = 0; 
       anastructIndex < _imAnaVectors[overlayIndex].size(); 
       ++anastructIndex)
    {
      Vector3D<double> 
        overlayColor(_imAnaVectors[overlayIndex][anastructIndex].color);

      axial2DWindow->
        addOverlayAnastruct(_imAnaVectors[overlayIndex][anastructIndex].anastruct,
                            _imAnaVectors[overlayIndex][anastructIndex].visible,
                            overlayColor[0],overlayColor[1],overlayColor[2]);

      sagittal2DWindow->
        addOverlayAnastruct(_imAnaVectors[overlayIndex][anastructIndex].anastruct,
                            _imAnaVectors[overlayIndex][anastructIndex].visible,
                            overlayColor[0],overlayColor[1],overlayColor[2]);

      coronal2DWindow->
        addOverlayAnastruct(_imAnaVectors[overlayIndex][anastructIndex].anastruct,
                            _imAnaVectors[overlayIndex][anastructIndex].visible,
                            overlayColor[0],overlayColor[1],overlayColor[2]);
    }   
  axial2DWindow->redraw();
  sagittal2DWindow->redraw();
  coronal2DWindow->redraw();
}

void
ImMap
::_createAnastruct(const Surface& surface,
                   ImageIndexType const imageOrigin,
                   ImageSizeType const imageSpacing,
                   const std::string& name,
                   Anastruct& anastruct)
{
  //
  // decide contour z positions
  //
  Vector3D<double> surfaceMax, surfaceMin;
  surface.getAxisExtrema(surfaceMin, surfaceMax);

  // get slice index for min and max
  int minSliceIndex = 
    static_cast<int>(ceil((surfaceMin.z - 
    imageOrigin[2])
    / imageSpacing[2]));
  int maxSliceIndex = 
    static_cast<int>(floor((surfaceMax.z - 
    imageOrigin[2])
    / imageSpacing[2]));

  //when the spacing is negative
  if (minSliceIndex > maxSliceIndex)
  {
    int tmp = maxSliceIndex;
    maxSliceIndex = minSliceIndex;
    minSliceIndex = tmp;
  }

  // include all intervening slices
  unsigned int numContours = maxSliceIndex - minSliceIndex + 1;
  double *contourZPositions = new double[numContours];
  int    *sliceNumbers      = new int[numContours];
  for (int contourSliceIndex = minSliceIndex, positionIndex = 0;
  contourSliceIndex <= maxSliceIndex;
  ++contourSliceIndex, ++positionIndex)
  {
    contourZPositions[positionIndex] = 
      imageOrigin[2] +
      imageSpacing[2] * contourSliceIndex;
    sliceNumbers[positionIndex] = contourSliceIndex;
  }

  // create the anastruct from the surface
  AnastructUtils::surfaceToAnastruct(surface, 
    anastruct,
    numContours, 
    sliceNumbers,
    contourZPositions);

  // make anastruct x and y in image index coords
  AnastructUtils::worldToImageIndexXY(anastruct,
    imageOrigin[0],
    imageOrigin[1],
    imageSpacing[0],
    imageSpacing[1]);
  delete [] contourZPositions;
  delete [] sliceNumbers;

  // copy over the anastruct name
  anastruct.label = name;
}

void 
ImMap
::_applyTranslation(const float& tx, 
                    const float& ty,
                    const float& tz,
                    unsigned int atlasIndex,
                    unsigned int subjectIndex,
                    std::string& imageName)
{
  AffineTransform3D<double> transform;
  transform.vector = Vector3D<double>(tx, ty, tz);
  _applyAffineTransform(transform, atlasIndex, subjectIndex, imageName);
}

void 
ImMap
::_applyAffineTransform(const AffineTransform3D<double>& transform,
                        unsigned int atlasIndex,
                        unsigned int subjectIndex,
                        std::string& imageName)
{

  _updateStatusBuffer("Creating Transformed Image...");
  ImagePointer registeredImage = new ImageType(*_loadedImages[atlasIndex]);

  EstimateAffine::ApplyTransformation(_loadedImages[atlasIndex],
                                      _loadedImages[subjectIndex],
                                      registeredImage, transform);
  _updateStatusBuffer("Creating Transformed Image...DONE");  

  // add translated surfaces and anastructs
  AffineTransform3D<double> invertedTransform = transform;
  if (!invertedTransform.invert()) {
    std::cerr << "Transform not invertible: Transformed "
              << "surfaces will not be created." << std::endl;
  } else {
    _updateStatusBuffer("Creating Transformed Surfaces...");  
    std::vector< ImAna > transformedImAnas;
    for (unsigned int surfaceIndex = 0; 
         surfaceIndex < _imAnaVectors[subjectIndex].size();
         ++surfaceIndex)
    {
      std::cerr << "###### translating surface ######" << std::endl;
      Surface transformedSurface(
        _imAnaVectors[subjectIndex][surfaceIndex].surface);
      transformedSurface.applyAffineTransform(invertedTransform);

      std::cerr << "######  creating anastruct ######" << std::endl;      
      Anastruct transformedAnastruct;
      _createAnastruct(transformedSurface,
                       _loadedImages[atlasIndex]->getOrigin(),
                       _loadedImages[atlasIndex]->getSpacing(),
                       _imAnaVectors[subjectIndex][surfaceIndex].anastruct.label,
                       transformedAnastruct);

      ImAna transformedImAna(transformedAnastruct,
                             transformedSurface,
                             _imAnaVectors[subjectIndex][surfaceIndex].visible,
                             _imAnaVectors[subjectIndex][surfaceIndex].color,
                             _imAnaVectors[subjectIndex][surfaceIndex].aspect,
                             _imAnaVectors[subjectIndex][surfaceIndex].opacity);
      transformedImAnas.push_back(transformedImAna);
    }
    _updateStatusBuffer("Creating Transformed Surfaces...DONE");  
    _imAnaVectors.push_back(transformedImAnas);
  }

  _imageNames.push_back(imageName);
  _imageFullFileNames.push_back(imageName);
  registeredImage->setDataType( Image<float>::Float );
  registeredImage->setOrientation(_loadedImages[atlasIndex]->getOrientation());
  _addImageToList(registeredImage);
  _updateStatusBuffer("Done applying Affine");
}


void
ImMap
::_applyFluidWarp(MultiScaleFluidWarp& fluidWarpInterface,
                  unsigned int atlasIndex,
                  unsigned int subjectIndex,
                  int createOToIImage,
                  int createOToISurfaces,
                  int createIToOSurfaces,
                  std::string& imageName)
{
  if (createOToIImage && subjectIndex > 0)
  {
    //
    // create hinv(overlay), ie warp overlay into space of image
    //
    try
    {
      // apply to atlas and add to loaded image list
      _updateStatusBuffer("Warping image ...");
      ImagePointer warpedImageNotResampled =
          fluidWarpInterface.apply(_loadedImages[subjectIndex]);
      ImagePointer warpedImage = new ImageType(*_loadedImages[subjectIndex]);
      ImageUtils::resampleWithTransparency(*warpedImageNotResampled, 
                                           *warpedImage);

      _imageNames.push_back(imageName);
      _imageFullFileNames.push_back(imageName);
      warpedImage->setDataType( Image<float>::Float );
      warpedImage->setOrientation(_loadedImages[atlasIndex]->getOrientation());
      _addImageToList(warpedImage);
      _updateStatusBuffer("Warping image ... DONE");
    }
    catch(...)
    {
      fl_alert("Failed to warp image.");
      _updateStatusBuffer("Failed to warp image.");
      return;
    }

    if (createOToISurfaces)
    {
      // 
      // create hinv(overlay surfaces) and add to hinv(overlay) images
      // list (this is the hard, slow direction for warping surfaces 
      //

      // add warped surfaces and anastructs to lists
      _updateStatusBuffer("Warping surfaces O->I ...");

      std::vector<ImAna> warpedImAnas;

      for (unsigned int surfaceIndex = 0; 
           surfaceIndex < _imAnaVectors[subjectIndex].size();
           ++surfaceIndex)
      {
        std::cerr << "######  warping  surface  ######" << std::endl;
        try 
        {

          // Create and possibly refine surface to warp
          int iavs = _imAnaVectors[subjectIndex].size();
          std::cout << iavs << std::endl;
          Surface warpedSurface(
              _imAnaVectors[subjectIndex][surfaceIndex].surface);
          if (refineButton->value())
          {
            SurfaceUtils::refineSurface(warpedSurface);
          }

        SurfaceUtils::worldToImageIndexXY(warpedSurface,
          _loadedImages[subjectIndex]->getOrigin(),
          _loadedImages[subjectIndex]->getSpacing());

          HField3DUtils::inverseApply(warpedSurface,
                                      fluidWarpInterface.getHField(),
                                      fluidWarpInterface.getROI());

        SurfaceUtils::imageIndexToWorldXY(warpedSurface,
          _loadedImages[atlasIndex]->getOrigin(),
          _loadedImages[atlasIndex]->getSpacing());

//           SurfaceUtils::worldToImageIndex(warpedSurface,
//                                           fluidWarpInterface.getHFieldOrigin(),
//                                           fluidWarpInterface.getHFieldSpacing());

//           SurfaceUtils::imageIndexToWorld(warpedSurface,
//                                           fluidWarpInterface.getHFieldOrigin(),
//                                           fluidWarpInterface.getHFieldSpacing());
                                      
          std::cout << " " <<   roiStartX->value()
                    << " " <<   roiStartY->value()
                    << " " <<   roiStartZ->value()
                    << " " <<   roiStopX->value()
                    << " " <<   roiStopY->value()
                    << " " <<   roiStopZ->value()
                    << "\n" << fluidWarpInterface.getROI().getStart()
                    << " " << fluidWarpInterface.getROI().getStop()
                    << "\n\n" << std::endl;

          // Create anastruct from warped surface
          Anastruct warpedAnastruct;

          std::string exten("_warp");
          std::string anastructName =
            _imAnaVectors[subjectIndex][surfaceIndex].anastruct.label;
          anastructName.append(exten);

          _createAnastruct(warpedSurface,
            _loadedImages[atlasIndex]->getOrigin(),
            _loadedImages[atlasIndex]->getSpacing(),
            anastructName,
            warpedAnastruct);

          ImAna warpedImAna(warpedAnastruct,
                          warpedSurface,
                          _imAnaVectors[subjectIndex][surfaceIndex].visible,
                          _imAnaVectors[subjectIndex][surfaceIndex].color,
                          _imAnaVectors[subjectIndex][surfaceIndex].aspect,
                          _imAnaVectors[subjectIndex][surfaceIndex].opacity);
           warpedImAnas.push_back(warpedImAna);
        }
        catch(...)
        {
          fl_alert("Warping surface failed.");
          _updateStatusBuffer("Warping surface failed.");
          return;
        }
      }
      _updateStatusBuffer("Warping surfaces O->I ... DONE");

      _imAnaVectors.push_back(warpedImAnas);
    }
    else
    {
      //add an empty ImAna vector to the list
      _imAnaVectors.push_back(std::vector<ImAna>());
    }
  }

  if (createIToOSurfaces && atlasIndex > 0 && subjectIndex > 0)
  {
    //
    // foreward warp surface from image->overlay and add
    // to overlay image's list
    //
    _updateStatusBuffer("Warping surfaces I->O ...");
    for (unsigned int surfaceIndex = 0; 
    surfaceIndex < _imAnaVectors[atlasIndex].size();
    ++surfaceIndex)
    {
      try 
      {
        Surface warpedSurface(_imAnaVectors[atlasIndex][surfaceIndex].surface);
        if (refineButton->value())
        {
        SurfaceUtils::refineSurface(warpedSurface);
        }
        SurfaceUtils::worldToImageIndexXY(warpedSurface,
          _loadedImages[atlasIndex]->getOrigin(),
          _loadedImages[atlasIndex]->getSpacing());
        fluidWarpInterface.apply(warpedSurface);
        SurfaceUtils::imageIndexToWorldXY(warpedSurface,
          _loadedImages[subjectIndex]->getOrigin(),
          _loadedImages[subjectIndex]->getSpacing());

        std::cerr << "###### creating anastruct ######" << std::endl;
        Anastruct warpedAnastruct;

        std::string exten("_warp");
        std::string anastructName=_imAnaVectors[atlasIndex][surfaceIndex].anastruct.label;
        anastructName = anastructName.insert(anastructName.length(),exten);

        _createAnastruct(warpedSurface,
          _loadedImages[subjectIndex]->getOrigin(),
          _loadedImages[subjectIndex]->getSpacing(),
          anastructName,
          warpedAnastruct);
        ImAna warpedImAna(warpedAnastruct,
                          warpedSurface,
                          _imAnaVectors[atlasIndex][surfaceIndex].visible,
                          _imAnaVectors[atlasIndex][surfaceIndex].color,
                          _imAnaVectors[atlasIndex][surfaceIndex].aspect,
                          _imAnaVectors[atlasIndex][surfaceIndex].opacity);
        _imAnaVectors[subjectIndex].push_back(warpedImAna);
      }
      catch(...)
      {
        fl_alert("Failed to warp surface.");
        _updateStatusBuffer("Failed to warp surface.");
      }
    }

    _updateStatusBuffer("Warping surfaces I->O ... DONE");
  }
}

/////////////////////////
// _createImageFromROI //     
/////////////////////////

void
ImMap
::_createImageFromROI(ImagePointer& inImage, ImagePointer& outImage)
{
  Array3D<VoxelType> roiArray;

  ROIUtils<VoxelType>::extractROIFromArray3D(*inImage, roiArray, _getROI() );
  *outImage = roiArray;

  Vector3D<int> roiStart = _getROI().getStart();
 
  Vector3D<float> origin = inImage->getOrigin();
  Vector3D<float> spacing = inImage->getSpacing();

	origin.x += roiStart.x*spacing.x;
	origin.y += roiStart.y*spacing.y;
	origin.z += roiStart.z*spacing.z;

  

  outImage->setOrigin(origin);
  outImage->setSpacing(inImage->getSpacing());
}

namespace {

// This function seems too specific for ImageUtils, but too
// non-GUI-related to belong here.  It can be moved later.  Note that
// it is simply a global function, not a member of the class.  But it
// is inaccessible outside this file because of the unnamed namespace.
void
fillGas(Image<float>& image, const Vector3D<unsigned int>& seedBodyExterior,
        int numErosions, int numDilations, float maxThresh, float fillValue)
{

  Array3D<unsigned char> exteriorMask;
  Array3D<float> gasMask(image.getSize());

  Array3DUtils::maskRegionDeterminedByThresholds(
    (Array3D<float>&)(image), exteriorMask, seedBodyExterior, 0.0f,
    maxThresh);

  float* pixel = image.getDataPointer();
  float* imageEnd = pixel + image.getNumElements();
  unsigned char* exteriorMaskValue = exteriorMask.getDataPointer();
  float* gasMaskValue = gasMask.getDataPointer();
  for ( ; pixel < imageEnd; ++pixel, ++exteriorMaskValue, ++gasMaskValue ) {
    *gasMaskValue = float(!(*exteriorMaskValue) && (*pixel < maxThresh));
  }

  // use morphological operators to clean mask
  for (int i = 0; i < numErosions; ++i) {
    Array3DUtils::minFilter3D(gasMask);
  }
  for (int i = 0; i < numDilations; ++i) {
    Array3DUtils::maxFilter3D(gasMask);
  }

  pixel = image.getDataPointer();
  gasMaskValue = gasMask.getDataPointer();
  for ( ; pixel < imageEnd; ++pixel, ++gasMaskValue ) {
    *pixel = (1.0f - (*gasMaskValue)) * (*pixel) + (*gasMaskValue) * fillValue;
  }

}

} // End of unnamed namespace

void ImMap::
fillGasCallback()
{
  _updateStatusBuffer("Filling gassy regions with solid intensity...");
  if (!axial2DWindow->haveValidImage()) {
    _updateStatusBuffer("Error: Fill region: No image loaded");
    return;
  }
  int imageIndex = imageChoice->value();
  ImagePointer filledImage = new ImageType(*_loadedImages[imageIndex]);
  fillGas(*filledImage, getImageCenter(), 0, 2,
          float(imageValMin->value()), 
          float(pixelValueInput->value()));

  _imAnaVectors.push_back(std::vector<ImAna>());
  std::string imageName("gas-filled image");
  _imageNames.push_back(imageName);
  _imageFullFileNames.push_back(imageName);
  _addImageToList(filledImage);

  _updateStatusBuffer("DONE");
}

void ImMap::
fillRegionCallback()
{
  _updateStatusBuffer("Filling region with solid intensity...");
  if (!axial2DWindow->haveValidImage()) {
    _updateStatusBuffer("Error: Fill region: No image loaded");
    return;
  }
  int imageIndex = imageChoice->value();

  ImagePointer filledImage = new ImageType(*_loadedImages[imageIndex]);

  Array3D<unsigned char> mask(filledImage->getSize());

  std::cout << imageValMin->value() 
            << "  "
            << imageValMax->value() << std::endl;

  Array3DUtils::maskRegionDeterminedByThresholds(
    *filledImage, mask, getImageCenter(), float(imageValMin->value()), 
    float(imageValMax->value()));

  float* pixel = filledImage->getDataPointer();
  float* imageEnd = pixel + filledImage->getNumElements();
  unsigned char* maskValue = mask.getDataPointer();
  for ( ; pixel < imageEnd; ++pixel, ++maskValue ) {
    if (*maskValue) *pixel = pixelValueInput->value();
  }

  _imAnaVectors.push_back(std::vector<ImAna>());
  std::string imageName("threshold-filled image");
  _imageNames.push_back(imageName);
  _imageFullFileNames.push_back(imageName);
  _addImageToList(filledImage);

  _updateStatusBuffer("DONE");
}

void ImMap::
fillROICallback()
{
  _updateStatusBuffer("Filling ROI with solid intensity...");

  if (!axial2DWindow->haveValidImage()) 
  {
    _updateStatusBuffer("Error: Fill Overlay: no overlay loaded");
    return;
  }
  int imageIndex = imageChoice->value();
  ImagePointer filledImage = new ImageType(*_loadedImages[imageIndex]);
  Array3DUtils::fillRegion(*filledImage, _getROI(),
                           float(pixelValueInput->value()));

  _imAnaVectors.push_back(std::vector<ImAna>());
  std::string imageName("image with filled ROI");
  _imageNames.push_back(imageName);
  _imageFullFileNames.push_back(imageName);
  _addImageToList(filledImage);

  _updateStatusBuffer("DONE");
}
