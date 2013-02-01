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

#include "ApplicationUtils.h"
#include "itkImageIOBase.h"
#include "itkImageIOFactory.h"
#include <string>
#include <sys/stat.h>

void 
ApplicationUtils::
ParseOptionVoid(int& argc, char**& argv)
{
  argv++; argc--;
}

double 
ApplicationUtils::
ParseOptionDouble(int& argc, char**& argv)
{
  std::string s = ApplicationUtils::ParseOptionString(argc, argv);
  return atof(s.c_str());
}

int
ApplicationUtils::
ParseOptionInt(int& argc, char**& argv)
{
  std::string s = ApplicationUtils::ParseOptionString(argc, argv);
  return atoi(s.c_str());
}

bool
ApplicationUtils::
ParseOptionBool(int& argc, char**& argv)
{
  std::string s = ApplicationUtils::ParseOptionString(argc, argv);
  if (s == "true" || s == "True" || s == "TRUE" || s == "1")
  {
    return true;
  }
  return false;
}

std::string 
ApplicationUtils::
ParseOptionString(int& argc, char**& argv)
{
  if (argc == 0)
  {
    std::stringstream ss;
    ss << "Error parsing argument: no arguments";
    throw AtlasWerksException(__FILE__, __LINE__, ss.str().c_str());
  }

  std::string arg(argv[0]);
  std::string::size_type equalPos = arg.find('=');
  std::string val = "";
  if (equalPos != std::string::npos)
  {
    // parse key=value and return value
    val = arg.substr(equalPos+1);
    argv++; argc--;
  }
  else if (argc > 1)
  {
    val = argv[1];
    argv++; argc--;
    argv++; argc--;
  }
  else
  {
    std::stringstream ss;
    ss << "Error parsing argument: " << arg;
    throw AtlasWerksException(__FILE__, __LINE__, ss.str().c_str());
  }

  // uncomment this to debug argument parsing
  //std::cerr << "arg=" << arg << ", val=" << val << std::endl;

  return val;
}

bool
ApplicationUtils::
ITKHasImageFileReader(const char* fileName)
{
  itk::ImageIOBase::Pointer imageIO = 
    itk::ImageIOFactory::CreateImageIO(fileName, 
                                       itk::ImageIOFactory::ReadMode);
  return (!imageIO.IsNull());
}

void        
ApplicationUtils::
ReadHeaderITK(const char* fileName,
	      SizeType &size,
	      OriginType &origin,
	      SpacingType &spacing)
{
  //
  // create a temporary reader with an arbitrary pixel type in order
  // to figure out what the real pixel type is
  typedef itk::Image<unsigned short, 3>           InImageType;
  typedef itk::ImageFileReader<InImageType>       ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(fileName);
  reader->UpdateOutputInformation();
  //    itk::ImageIOBase::IOComponentType inputComponentType = 
  //      reader->GetImageIO()->GetComponentType();
  //    itk::ImageIOBase::IOPixelType pixelType = 
  //      reader->GetImageIO()->GetPixelType();
  unsigned int nDims = 
    reader->GetImageIO()->GetNumberOfDimensions();
  if(nDims != 3){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, number of dimensions != 3");
  }
  for(unsigned int dim=0;dim<nDims;dim++){
    size[dim] = reader->GetImageIO()->GetDimensions(dim);
    origin[dim] = reader->GetImageIO()->GetOrigin(dim);
    spacing[dim] = reader->GetImageIO()->GetSpacing(dim);
  }
}

bool 
ApplicationUtils::
FileExists(const char *filename)
{
  struct stat buffer;
  if( stat(filename, &buffer) == 0) return true;
  return false;
}

// take filename such as /some/path/file.ext
// return /some/path/, file, ext
void 
ApplicationUtils::
SplitName(const char *fullPathIn, 
	  std::string &path,
	  std::string &nameBase,
	  std::string &nameExt)
{
  std::string fullPath(fullPathIn);

  // find the path
  size_t idx = fullPath.find_last_of('/');
  if(idx == std::string::npos){
    path = "";
    nameBase = fullPath;
  }else{
    path = fullPath.substr(0,idx+1);
    nameBase = fullPath.substr(idx+1);
  }
  // separate the base and ext
  size_t dotIdx = nameBase.find_last_of('.');
  if(dotIdx == std::string::npos){
    nameExt = "";
  }else{
    nameExt  = nameBase.substr(dotIdx+1);
    nameBase = nameBase.substr(0,dotIdx);
  }
}

void 
ApplicationUtils::
Distribute(int numTotalElements, int numBins, int binNum, 
	   int &beginId, int &numElements)
{

  beginId=0;

  if(numBins > 1){
    
    // min number of elements per bin
    numElements = static_cast<unsigned int>(floor(static_cast<float>(numTotalElements)/static_cast<float>(numBins)));
    assert(numElements > 0);
    int remaining = numTotalElements%numBins;
    if(binNum<remaining){
      numElements++;
    }
    
    // begin ID
    if(binNum<remaining){
      beginId = numElements*binNum;
    }else{
      beginId = (numElements+1)*remaining + numElements*(binNum-remaining);
    }
    
  }else{
    // only one bin, it gets all images
    numElements = numTotalElements;
  }
  
}
