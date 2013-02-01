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

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageIOBase.h>
#include <itkCastImageFilter.h>
#include <itkImage.h>
#include <string>
#include <unistd.h>
#include <sstream>
#include <tclap/CmdLine.h>

/**
 * \page ImageConvert
 *
 * Convert an image from one format to another.  Supports 2D and 3D
 * images, vector and scalar images, and all ITK-supported formats.
 * Can generate compressed output. Can be used to convert between the
 * following datatypes:
 *
 * CHAR
 * UCHAR
 * SHORT
 * USHORT
 * INT
 * UINT
 * LONG
 * ULONG
 * FLOAT
 * DOUBLE
 *
 * Run 'ImageConvert -h' for detailed usage 
 */


//
// These macros are used to handle the templated intput and output
// types more cleanly.  The statement 'call' is made with ITKIO_TT
// typedef'd to the correct datatype
#define itkIOTemplateDataMacroCase(typeN, type, call)     \
  case typeN: { typedef type ITKIO_TT; call; }; break

#define itkIOTemplateDataMacro(call)                                            \
  itkIOTemplateDataMacroCase(itk::ImageIOBase::UCHAR,  unsigned char,  call);   \
  itkIOTemplateDataMacroCase(itk::ImageIOBase::CHAR,   char,           call);   \
  itkIOTemplateDataMacroCase(itk::ImageIOBase::USHORT, unsigned short, call);   \
  itkIOTemplateDataMacroCase(itk::ImageIOBase::SHORT,  short,          call);   \
  itkIOTemplateDataMacroCase(itk::ImageIOBase::UINT,   unsigned int,   call);   \
  itkIOTemplateDataMacroCase(itk::ImageIOBase::INT,    int,            call);   \
  itkIOTemplateDataMacroCase(itk::ImageIOBase::ULONG,  unsigned long,  call);   \
  itkIOTemplateDataMacroCase(itk::ImageIOBase::LONG,   long,           call);   \
  itkIOTemplateDataMacroCase(itk::ImageIOBase::FLOAT,  float,          call);   \
  itkIOTemplateDataMacroCase(itk::ImageIOBase::DOUBLE, double,         call);   

// The statement 'call' is made with ITKIO_IN_TT and ITKIO_OUT_TT
// typedef'd to the correct pixel types, either vector or scalar
// versions of inType and outType
#define itkIOTemplateTypeMacro(inType, outType, dims, pixelTypeVar, call) \
  if(pixelTypeVar == itk::ImageIOBase::SCALAR){ \
    typedef inType ITKIO_IN_TT; \
    typedef outType ITKIO_OUT_TT; \
    std::cout << "Pixel type is scalar" << std::endl; \
    call; \
  }else if(pixelTypeVar == itk::ImageIOBase::VECTOR){ \
    typedef itk::Vector<inType, dims> ITKIO_IN_TT; \
    typedef itk::Vector<outType, dims> ITKIO_OUT_TT; \
    std::cout << "Pixel type is vector" << std::endl; \
    call; \
  }else{ \
     std::cerr << "Error: unknown pixel type." << std::endl; \
  }


template <class InPixelType, class OutPixelType, unsigned int Dimension>
void
convertImage(const std::string& inputImageFile,
             const std::string& outputImageFile,
             bool useCompression)
{
  //
  // read input image and cast it to output image type, and write it
  //

  typedef itk::Image<InPixelType, Dimension>              InImageType;
  typedef itk::Image<OutPixelType, Dimension>             OutImageType;

  typedef itk::ImageFileReader<InImageType>               ReaderType;
  typedef itk::CastImageFilter<InImageType, OutImageType> CasterType;
  typedef itk::ImageFileWriter<OutImageType>              WriterType;

  typename ReaderType::Pointer   reader                 = ReaderType::New();
  typename CasterType::Pointer   caster                 = CasterType::New();
  typename WriterType::Pointer   writer                 = WriterType::New();

  reader->SetFileName(inputImageFile.c_str());
  caster->SetInput(reader->GetOutput());
  writer->SetInput(caster->GetOutput());
  writer->SetFileName(outputImageFile.c_str());
  writer->SetUseCompression(useCompression);

  //
  // update the pipeline, writing the image
  try 
    {
      writer->Update();
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << "ExceptionObject caught:" << std::endl;
      std::cerr << err << std::endl;
    }  
}

template <class InDataType, class OutDataType, unsigned int Dimension>
void
convertPixelTypeHandler(const std::string& inputImageFile,
			const std::string& outputImageFile,
			itk::ImageIOBase::IOPixelType pixelType,	     
			bool useCompression)
{
  itkIOTemplateTypeMacro(InDataType, OutDataType, Dimension, pixelType, (convertImage<ITKIO_IN_TT, ITKIO_OUT_TT, Dimension>(inputImageFile, outputImageFile, useCompression)));
}


template <class InDataType, unsigned int Dimension>
void
convertOutputDataTypeHandler(const std::string& inputImageFile,
			     const std::string& outputImageFile,
			     itk::ImageIOBase::IOComponentType outputType,
			     itk::ImageIOBase::IOPixelType pixelType,	     
			     bool useCompression)
{
  switch (outputType)
    {
    itkIOTemplateDataMacro((convertPixelTypeHandler<InDataType, ITKIO_TT, Dimension>(inputImageFile, outputImageFile, pixelType, useCompression)));
    default:
      std::cerr << "Error: unknown pixel type." << std::endl;
    }
}

template <unsigned int Dimension>
void
convertInputDataTypeHandler(const std::string& inputImageFile,
			const std::string& outputImageFile,
			itk::ImageIOBase::IOComponentType inputType,
			itk::ImageIOBase::IOComponentType outputType,
			itk::ImageIOBase::IOPixelType pixelType,	     
			bool useCompression)
{
  switch (inputType)
    {
    itkIOTemplateDataMacro((convertOutputDataTypeHandler<ITKIO_TT, Dimension>(inputImageFile, outputImageFile, outputType, pixelType, useCompression)));
    default:
      std::cerr << "Error: unknown pixel type." << std::endl;
    }
}


void
templatedConversionHandler(const std::string& inputImageFile,
			   const std::string& outputImageFile,
			   itk::ImageIOBase::IOComponentType inputType,
			   itk::ImageIOBase::IOComponentType outputType,
			   itk::ImageIOBase::IOPixelType pixelType,
			   unsigned int dim,
			   bool useCompression)
{
  if(dim == 2){
    convertInputDataTypeHandler<2>(inputImageFile, outputImageFile, inputType, outputType, pixelType, useCompression);
  }else if(dim == 3){
    convertInputDataTypeHandler<3>(inputImageFile, outputImageFile, inputType, outputType, pixelType, useCompression);
  }else{
      std::cerr << "Error: only two- and three-dimensional images supported" << std::endl;
  }
}


itk::ImageIOBase::IOComponentType parseOutputType(const char* str)
{
  std::string typeString(str);
  if (typeString == "CHAR")
  {
    return itk::ImageIOBase::CHAR;
  }
  if (typeString == "UCHAR")
  {
    return itk::ImageIOBase::UCHAR;
  }
  if (typeString == "SHORT")
  {
    return itk::ImageIOBase::SHORT;
  }
  if (typeString == "USHORT")
  {
    return itk::ImageIOBase::USHORT;
  }
  if (typeString == "INT")
  {
    return itk::ImageIOBase::INT;
  }
  if (typeString == "UINT")
  {
    return itk::ImageIOBase::UINT;
  }
  if (typeString == "LONG")
  {
    return itk::ImageIOBase::LONG;
  }
  if (typeString == "ULONG")
  {
    return itk::ImageIOBase::ULONG;
  }
  if (typeString == "FLOAT")
  {
    return itk::ImageIOBase::FLOAT;
  }
  if (typeString == "DOUBLE")
  {
    return itk::ImageIOBase::DOUBLE;
  }
  std::cerr << "Unknown component type: " << typeString << std::endl;
  std::cerr << "Use one of:" << std::endl
            << " CHAR" << std::endl
            << " UCHAR" << std::endl
            << " SHORT" << std::endl
            << " USHORT" << std::endl
            << " INT" << std::endl
            << " UINT" << std::endl
            << " LONG" << std::endl
            << " ULONG" << std::endl
            << " FLOAT" << std::endl
            << " DOUBLE" << std::endl
            << std::endl;
  return itk::ImageIOBase::UNKNOWNCOMPONENTTYPE;
}

int main( int argc, char ** argv )
{
  //
  // parse command line
  //
  std::string inputImageFileName;
  std::string outputImageFileName;
  bool outputTypeSpecified;
  itk::ImageIOBase::IOComponentType outputComponentType;
  bool compressOutputImage;
  bool verbose;

  try 
  {
    TCLAP::CmdLine cmd("ImageConvert", ' ', "1.0");
    
    TCLAP::SwitchArg
      verboseArg("v","verbose",
                 "Print extra output",
                 cmd, false);

    TCLAP::ValueArg<std::string> 
      inputArg("i","input",
               "Input image file name",
               true,"","string", cmd);

    TCLAP::ValueArg<std::string> 
      outputArg("o","output",
                "Output image file name",
                true,"","string", cmd);

    TCLAP::ValueArg<std::string> 
      outputTypeArg("t","outputVoxelType",
                    "Convert input voxels to this type before writing "
		    "(CHAR,UCHAR,SHORT,USHORT,INT,UINT,LONG,ULONG,FLOAT,DOUBLE)",
                    false,"","string", cmd);

    TCLAP::SwitchArg
      useCompressionArg("c","compressOutputImage",
                        "Compress the output image (if supported by writer)",
                        cmd, false);

    cmd.parse( argc, argv );

    //
    // image conversion parameters
    verbose                    = verboseArg.getValue();
    inputImageFileName         = inputArg.getValue();
    outputImageFileName        = outputArg.getValue();
    outputTypeSpecified        = !outputTypeArg.getValue().empty();
    if (outputTypeSpecified)
    {
      outputComponentType = parseOutputType(outputTypeArg.getValue().c_str());
    }
    compressOutputImage        = useCompressionArg.getValue();
  }
  catch (TCLAP::ArgException &e)  
  { 
    std::cerr << "error: " << e.error() << " for arg " << e.argId() 
              << std::endl; 
    exit(1);
  }

  //
  // create a temporary reader with an arbitrary pixel type in order
  // to figure out what the real pixel type is
  typedef itk::Image<unsigned short, 3>           InImageType;
  typedef itk::ImageFileReader<InImageType>       ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputImageFileName.c_str());
  reader->UpdateOutputInformation();
  itk::ImageIOBase::IOComponentType inputComponentType = 
    reader->GetImageIO()->GetComponentType();
  itk::ImageIOBase::IOPixelType pixelType = 
    reader->GetImageIO()->GetPixelType();
  unsigned int nDims = 
    reader->GetImageIO()->GetNumberOfDimensions();
  //
  // set the output component type to the input component type (if not
  // overridden by command line argument)
  if (!outputTypeSpecified)
    {
    outputComponentType = inputComponentType;
    }
  
  //
  // print conversion parameters
  if (verbose)
    {
    std::cerr << "Input Image:           " << inputImageFileName  << std::endl;
    std::cerr << "Input Voxel Type:      " 
              << reader->GetImageIO()->
      GetComponentTypeAsString(inputComponentType)  << std::endl;
    std::cerr << "Output Image:          " << outputImageFileName << std::endl;
    std::cerr << "Output Voxel Type:     " 
              << reader->GetImageIO()->
      GetComponentTypeAsString(outputComponentType) << std::endl;
    std::cerr << "Request Compression:   " << compressOutputImage << std::endl;
    std::cerr << std::endl;
    }

  //
  // convert the image (using template handlers to deal with combining
  // input and output types)
  templatedConversionHandler(inputImageFileName,
			     outputImageFileName,
			     inputComponentType,
			     outputComponentType,
			     pixelType,
			     nDims,
			     compressOutputImage);
}
