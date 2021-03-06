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

#ifndef ImMap_h
#define ImMap_h

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include <vector>
#include <string>

#include "ImMapGUI.h"  // Automatically generated by Fluid
#include "HistogramDat.h"
#include <Surface.h>
#include <Matrix.h>
#include <AffineTransform3D.h>
#include "Anastruct.h"
#include "gen.h"
#include <ImageIO.h>
#include <ROI.h>
#include <Array2D.h>
#include "MultiScaleFluidWarp.h"
#include "FluidWarpParameters.h"
#include "ImAna.h"

const int MAX_NB_ANASTRUCT_LOADED = 200;

class ImMap : public ImMapGUI
{

public:
  typedef float                     VoxelType;
  typedef Image<VoxelType>          ImageType;
  typedef ImageType*                ImagePointer;
  typedef ROI<int, unsigned int>    ImageRegionType;
  typedef Vector3D<double>	    ImageSizeType;
  typedef Vector3D<double>          ImageIndexType;
  typedef Array2D<double>           ScheduleType;
  typedef Array3D<float>     MaskType;
  typedef Array3D<Vector3D<float> > HFieldType;
  ImMap();
  virtual ~ImMap();
  
  // special 
  static void clickCallback(int button, float xIndex, float yIndex, float zIndex, void* obj);
  static void roiChangedCallback(IViewOrientationType orientation, void* obj);
  static void MinMaxChangedCallback(void* obj);
  
  static void BYUTableChangedCallback(Fl_Widget* w, void* obj );
  
  // Virtual methods from ImMapGUI
  
  //input image
  void loadImage();
  void loadImage(char *fileName);
  void saveImageCallback();
  void unloadImageCallback();
 
  //load if extension unknown
  void saveheaderCallback();
 
  //control panel
  void imageChoiceCallback();
  void overlayChoiceCallback();
  void scrollbarCallback();
  void imagePropertyChangedCallback();
  void overlayPropertyChangedCallback();
  void imageIWChangedCallback();
  void overlayIWChangedCallback();
  void imageValMinMaxCallback();
  void overlayValMinMaxCallback();
  void lockedIntensityCallback();
  void infoImageCallback();
  void infoOverlayCallback();
  void createOToIImageButtonCallback();
  
  //ROI
  void roiPropertyChangedCallback();
  void roiLockCallback();
  void roiImageAddAsImageCallback();
  void roiOverlayAddAsImageCallback();
  void roiInitializeCallback();
  void roiInitializeRegistrationCallback();
  void fillGasCallback();
  void fillRegionCallback();
  void fillROICallback();
  
  //debug functions not in the GUI
  void downSampleCallback();
  void downSizeCallback();

  //transformation
  void affineRegisterCallback();
  void fluidDeflateCallback();
  void fluidShrinkCallback();
  void fluidRegisterCallback();
  void applyTranslationVectorCallback();
  void alignCentroidsCallback();
  void applyHFieldCallback();
  void resampleCallback();	
  void applyMatrixCallback();
  void saveMatrixCallback();
  void setOriginCallback();
  
  //zoom
  void zoomCallback();
  void zoomButtonCallback(float val);
  
  //histogram
  void histogramImageCallback();
  void transmitHistoImageCallback();
  void histogramOverlayCallback();
  void transmitHistoOverlayCallback();
  void histogramButtonCallback();
  void histogramLinkCallback();
  
  //main window slider
  void histogramSliderCallback();
  void updateHistogramSliderCallback();
  
  //preset file
  void presetIntensityCallback(int imageIntensity);
  void loadPresetFileCallback();
  void setPresetFilename(std::string filename){_presetFilename=filename;};
  void fluidParametersCallback();
  void saveFluidParametersCallback();
  
  //BYU
  void displayBYUListCallback();
  void loadBYUCallback();
  void unloadBYUCallback();
  void saveBYUCallback();
  void BYUTableCallback(int val);
  double getVolume(const Surface& s);
  Vector3D<double> getCentroid(const Surface& s);
  void infoBYUCallback();
  void closeinfoBYUCallback();
  void refineBYUCallback();
  // void SaveBYUInfoCallback();
  // void SaveBYUInfo(char* filename);

  //anastruct
  void saveAnastructCallback();
  void loadAnastructCallback();
  void anastructChoiceCallback();
  void anastructPropertyChangedCallback();
  
  // Wizard Related Callback	
  void wizardTranslateRegisterCallback();
  void wizardLoadAnastructsCallback();
  void wizardRegistrationCallback();
  void referenceInfoCallback();
  void dailyInfoCallback();
  void saveTranslationCallback();    // Save the translation
  void saveTranslationResCallback(); // Save the resulting image
  void initWizardCallback();
  
  // DicomLoader
  void DICOMCallback();
  void DicomLoaderCallback();
  void SelectFilesCallback();
  void DeselectFilesCallback();
  void SliceSelectedCallback();
  void SortSelectionCallback(int val);
  void InvertSortCallback(int val); 
  void AddAllCallback();
  void RemoveAllCallback();
  void CreateImageCallback();

  // user interface preferences (misc tab)
  void viewCrosshairsCallback();
  void viewImageInfoCallback();
  void lineWidthCallback(const double& width);
  void window3DBGColorCallback();
  void printParametersCallback();

  //mask
  void createMaskCallback();
  void maskViewCallback();
  
  // screen capture callbacks
  void screenCaptureAxialCallback();
  void screenCaptureCoronalCallback();
  void screenCaptureSagittalCallback();
  void screenCapture3DCallback();
  void screenCaptureAllCallback();
  
  // script
  void scriptCallback();
  void runScript(const std::string& filename);
  bool applyScript(const std::string& key, const std::string& value);

  const Vector3D<unsigned int> getImageCenter()
  { return axial2DWindow->getWindowCenter(); }

private:
  
  struct  Intensity			{float relativeMin, relativeMax;};
  struct  colorRGB			{float red,green, blue;};
  struct  ColorPreset			{colorRGB image,overlay;};
  struct  fluidParameters		{FluidWarpParameters params[3];};
  
  //Preset Data
  std::vector<Intensity>           _presetIntensity;
  std::vector<ColorPreset>         _colorPresetIntensity;
  std::vector<fluidParameters>     _fluidParamPreset;
  
  //Image Data
  std::vector<HistogramDat>        _histograms;
  std::vector<ImagePointer>        _loadedImages;
  std::vector<std::string>         _imageNames;
  std::vector<std::string>         _imageFullFileNames;
  
  //Image Anatonical Structure

  std::vector<std::vector<ImAna> > _imAnaVectors;

  //mask  
  std::vector<MaskType>               _mask;                
  //keep track of the rows selected in the BYUTable
  int _lastRowsSelected[MAX_NB_ANASTRUCT_LOADED];

  //flags or increment number
  int _totalImagesLoaded;
  bool _histogramImage;
  bool _histogramLinked;
  float _zoomValue;
  bool _ROIcreated;
  std::vector< std::string > _deformationChoices;

  std::string _presetFilename;
  AffineTransform3D<double> registrationTransform;
  
  // load and save the images 
  ImagePointer _loadImageDICOM(char *fileName);
  ImagePointer _loadOtherImage(char *fileName);
  static void  _saveImage(ImagePointer image, char* fileName);
  void _addImageToList(ImagePointer imagePointer);
  
  //control panel
  void _imageChoice(const unsigned int& imageIndex);
  void _unloadImage(const unsigned int& removeIndex);

  //anastruct
  void _loadAnastruct(const std::string& fileName,
                      const unsigned int& imageIndex);
  void _loadBYU(const std::string& fileName, const unsigned int& imageIndex);

  //script
  void _parseVector3DInt(const std::string& s, Vector3D<int>& v);
  void _parseVector3DDouble(const std::string& s, Vector3D<double>& v);
  void _parseIntensityWindows(const std::string& s, double& min, double& max);

  // screen capture callbacks
  void _screenCaptureAxial(const std::string& fileName);
  void _screenCaptureCoronal(const std::string& fileName);
  void _screenCaptureSagittal(const std::string& fileName);
  void _screenCapture3D(const std::string& fileName);
  void _screenCaptureAll(const std::string& fileName);

  std::string _parseValue(const std::string& str);
  void _updateTransformDisplay();
  void _updateSurfaces();
  void _updateROIin3DWindow();
  void _updateContours();
  void _updateImagePanel();
  void _updateROIPanel();
  void _updateROIInfo();
  void _updatePositionSliders();
  void _updateStatusBuffer(const char *message);
  
  void _redrawImageWindows();
  void _centerImage(float xIndex, float yIndex, float zIndex);
  ImageRegionType _getROI();
  void _setROI(	int startX, int startY, int startZ,
    int stopX, int stopY, int stopZ);
  ScheduleType _getPyramidSchedule();
  
  void _createImageFromROI(ImagePointer& inImage, ImagePointer& outImage);

  void _createPresetFile();
  
  void _createAnastruct( const Surface& surface,
    ImageIndexType const imageOrigin,
    ImageSizeType const imageSpacing,
    const std::string& name,
    Anastruct& anastruct);

  void _applyTranslation( const float& tx, 
    const float& ty,
    const float& tz,
    unsigned int atlasIndex,
    unsigned int subjectIndex,
    std::string& imageName);

  void _applyAffineTransform( const AffineTransform3D<double>& transform, 
    unsigned int atlasIndex,
    unsigned int subjectIndex,
    std::string& imageName);
  
  void _applyFluidWarp(	MultiScaleFluidWarp& fluidWarp,
    unsigned int atlasIndex,
    unsigned int subjectIndex,
    int createOToIImage,
    int createOToISurfaces,
    int createIToOSurfaces,
    std::string& imageName);
  
  // DicomLoader
  void _sortList(int val);
  void _addDicomFileSelector(char * SliceName, float Zposition);
  void _addDicomFileSelected(char * SliceName, float Zposition);
  void _getFileListDicom(char *filename);

  bool _usingROI() const;
  bool _roiIsSet() const;
  static bool _testFileCanWrite(const char* filename);
  static bool _testFileCanWrite(const std::string& filename)
  { return _testFileCanWrite(filename.c_str()); }
  bool _imageIsLoaded() const;
  bool _overlayIsLoaded() const;
  int _getCurrentImageIndex() const;
  int _getCurrentOverlayIndex() const;
  void _getFluidParameters(FluidWarpParameters* params,
                           unsigned int& numScaleLevels) const;

  void _printParameters(const char* startStr, const char* stopStr,
                        const char* intensityStr);

  int 	num_files;
  char **infile;
  std :: string DirectoryName;  
  std :: vector<float> Zpositions;	
  std :: vector<std :: string> filenames;
  std :: vector<float> ZpositionsSelected;
  std :: vector<std :: string> filenamesSelected;

  Fl_Text_Buffer* textbuf;

  // To Load Image with unknown extension
  std::string  _ObjectType,_ElementType,_ElementDataFile,_BinaryData, _BinaryDataByteOrderMSB;
  int _NDims; 
  std::vector<float> _Offset, _ElementSpacing;
  std::vector<int> _DimSize; 
  bool _header_saved;
  
};

#endif
