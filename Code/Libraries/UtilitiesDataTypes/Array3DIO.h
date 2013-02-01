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

#ifndef ARRAY3D_IO_H
#define ARRAY3D_IO_H

#include "Vector3D.h"
#include "Array3D.h"
#include "DownsampleFilter3D.h"
#include "StringUtils.h"
#include <limits>
#include <string>
#include <numeric>
#include <vector>
#include <iostream>
#include <fstream>

#include "Timer.h"

#include "AtlasWerksTypes.h"

class Array3DIO
{
public:
  enum SliceOrientation {SliceOrientationX, 
                         SliceOrientationY, 
                         SliceOrientationZ};

  //
  // write this array in binary to an ostream
  //
  // bcd 2004
  //
  template <class T>
  static
  void writeRawVolume(const Array3D<T>& array,
                      std::ostream& out)
  {
    if (out.bad())
    {
      throw std::runtime_error("ostream bad");
    }
    out.write((char*)array.getDataPointer(), array.getSizeBytes());
    if (out.bad() || out.fail())
    {
      throw std::runtime_error("ostream write failed");
    }
  }

  //
  // write this array as a binary file
  //
  // bcd 2004
  //
  template <class T>
  static
  void writeRawVolume(const Array3D<T>& array,
		      const char* filename)
  {
    std::ofstream out(filename, std::ios::binary);
    if (out.bad())
    {
      throw std::runtime_error("error opening file");
    }
    writeRawVolume(array, out);
    if (out.bad() || out.fail())
    {
      throw std::runtime_error("error writing file");
    }
    out.close();	    
  }

  //
  // read this array in binary from an istream.  
  // Note: the number of bytes read will depend on the size of the
  // array.
  // bcd 2003
  //
  template <class T>
  static
  void readRawVolume(const Array3D<T>& array,
		     std::istream& in)
  {
    if (in.bad())
    {
      throw std::runtime_error("istream bad");
    }
    in.read((char*)array.getDataPointer(), 
	    array.getSizeBytes());
    if (in.bad() || in.fail())
    {
      throw std::runtime_error("istream read failed");
    }
  }

  //
  // read this array from a binary file.
  // Note: the number of bytes read will depend on the size of the
  // array.
  // bcd 2003
  //
  template <class T>
  static
  void readRawVolume(const Array3D<T>& array,
		     const char* filename)
  {
    std::ifstream in(filename, std::ios::binary);
    if (in.bad())
    {
      throw std::runtime_error("error opening file");
    }
    readRawVolume(array, in);
    if (in.bad() || in.fail())
    {
      throw std::runtime_error("error reading file");
    }
    in.close();	    
  }

  //
  // read this array from the end of a binary file.
  // Note: the number of bytes read will depend on the size of the
  // array.
  // bcd 2004
  //
  template <class T>
  static
  void readRawVolumeEnd(const Array3D<T>& array,
			const char* filename)
  {
    std::ifstream in(filename, std::ios::binary);
    if (in.bad())
    {
      throw std::runtime_error("error opening file");
    }
    in.seekg(-array.getSizeBytes(), std::ios::end);
    readRawVolume(array, in);
    if (in.bad() || in.fail())
    {
      throw std::runtime_error("error reading file");
    }
    in.close();	    
  }

  //
  // write this array as a binary file with a short binary header
  // containing the size of the array
  //
  // bcd 2004
  // 
  template <class T>
  static
  void writeVolume(const Array3D<T>& array,
		   const char* filename)
  {
    std::ofstream out(filename, std::ios::binary);
    if (out.bad())
    {
      throw std::runtime_error("error opening file");
    }

    // write size
    Vector3D<unsigned int> size = array.getSize();
    out.write((char*)&size.x, sizeof(unsigned int));
    out.write((char*)&size.y, sizeof(unsigned int));
    out.write((char*)&size.z, sizeof(unsigned int));

    // write contents
    out.write((char*)array.getDataPointer(), array.getSizeBytes());

    if (out.fail())
    {
      throw std::runtime_error("error writing file");
    }

    out.close();	    
  }

  //
  // read this array from a volume file (size and raw data)
  //
  // bcd 2004
  //
  template <class T>
  static
  void readVolume(const Array3D<T>& array,
		  const char* filename)
  {
    std::ifstream in(filename, std::ios::binary);
    if (in.bad())
    {
      throw std::runtime_error("error opening file");
    }
    
    unsigned int sizeX, sizeY, sizeZ;
    in.read((char*)&sizeX, sizeof(unsigned int));
    in.read((char*)&sizeY, sizeof(unsigned int));
    in.read((char*)&sizeZ, sizeof(unsigned int));

    array.resize(sizeX, sizeY, sizeZ);

    in.read((char*)array.getDataPointer(), 
	    sizeof(T) * array.getNumElements());

    if (in.fail())
    {
      throw std::runtime_error("error reading file");
    }

    in.close();	    
  }

  //
  // write a slice of this array as a raw file
  //
  // bcd 2003
  //
  template <class T>
  static
  void writeRawSliceX(const Array3D<T>& array,
		      unsigned int slice,
		      const char* filename)
  {
    std::ofstream out(filename, std::ios::binary);
    if (out.bad())
    {
      throw std::runtime_error("error opening file");
    }
    for (unsigned int z = 0; z < array.getSizeZ(); ++z) {
      for (unsigned int y = 0; y < array.getSizeY(); ++y) {
	out.write((char*) &array(slice, y, z), sizeof(T));
      }
    }
    out.close();	      
  }
  
  //
  // write a slice of this array as a raw file
  //
  // bcd 2003
  //
  template <class T>
  static
  void writeRawSliceY(const Array3D<T>& array,
		      unsigned int slice,
		      const char* filename)
  {
    std::ofstream out(filename, std::ios::binary);
    if (out.bad())
    {
      throw std::runtime_error("error opening file");
    }
    for (unsigned int z = 0; z < array.getSizeZ(); ++z) {
      for (unsigned int x = 0; x < array.getSizeX(); ++x) {
	out.write((char*) &array(x, slice, z), sizeof(T));
      }
    }
    out.close();	      
  }

  //
  // write a slice of this array as a raw file
  //
  // bcd 2003
  //
  template <class T>
  static
  void writeRawSliceZ(const Array3D<T>& array,
		      unsigned int slice,
		      const char* filename)
  {
    std::ofstream out(filename, std::ios::binary);
    if (out.bad())
    {
      throw std::runtime_error("error opening file");
    }
    for (unsigned int y = 0; y < array.getSizeY(); ++y) {
      for (unsigned int x = 0; x < array.getSizeX(); ++x) {
	out.write((char*) &array(x, y, slice), sizeof(T));
      }
    }
    out.close();	      
  }

template <class T>
  static
  void writePGMSliceX(const Array3D<T>& array,
		      unsigned int slice,
		      const char* filename)
  {
    std::ofstream out(filename, std::ios::binary);
    if (out.bad())
    {
      throw std::runtime_error("error opening file");
    }
    out << "P5\n" 
        << array.getSizeY() << "\n" 
        << array.getSizeZ() << "\n"
        << 255 << "\n";

    for (unsigned int z = 0; z < array.getSizeZ(); ++z) {
    for (unsigned int y = 0; y < array.getSizeY(); ++y) {
        unsigned char pixel = (unsigned char) array(slice,y,z);
	out.write((char*) &pixel, 1);
      }
    }
    out.close();	      
  }

template <class T>
  static
  void writePGMSliceY(const Array3D<T>& array,
		      unsigned int slice,
		      const char* filename)
  {
    std::ofstream out(filename, std::ios::binary);
    if (out.bad())
    {
      throw std::runtime_error("error opening file");
    }
    out << "P5\n" 
        << array.getSizeX() << "\n" 
        << array.getSizeZ() << "\n"
        << 255 << "\n";

    for (unsigned int z = 0; z < array.getSizeZ(); ++z) {
      for (unsigned int x = 0; x < array.getSizeX(); ++x) {
      unsigned char pixel = (unsigned char) array(x,slice,z);
	out.write((char*) &pixel, 1);
      }
    }
    out.close();	      
  }

  template <class T>
  static
  void writePGMSliceZ(const Array3D<T>& array,
		      unsigned int slice,
		      const char* filename)
  {
    std::ofstream out(filename, std::ios::binary);
    if (out.bad())
    {
      throw std::runtime_error("error opening file");
    }
    out << "P5\n" 
        << array.getSizeX() << "\n" 
        << array.getSizeY() << "\n"
        << 255 << "\n";

    for (unsigned int y = 0; y < array.getSizeY(); ++y) {
      for (unsigned int x = 0; x < array.getSizeX(); ++x) {
        unsigned char pixel = (unsigned char) array(x,y,slice);
	out.write((char*) &pixel, 1);
      }
    }
    out.close();	      
  }

  //
  // discern the element type for writing a META file header
  //
  // bcd 2004
  // 
  template <class T>
  static
  std::string getMETAElementType(const Array3D<T>& array)
  {
    std::string elementType = "MET_OTHER";
    if (std::numeric_limits<T>::is_specialized)
    {
      if (std::numeric_limits<T>::is_integer)
        {
          if (std::numeric_limits<T>::is_signed)
            {
              if (std::numeric_limits<T>::digits == 7)
                {
                  elementType = "MET_CHAR";
                }
              else if (std::numeric_limits<T>::digits == 15)
                {
                  elementType = "MET_SHORT";
                }
              else if (std::numeric_limits<T>::digits == 31)
                {
                  elementType = "MET_INT";
                }
            }
          else
            {
              if (std::numeric_limits<T>::digits == 8)
                {
                  elementType = "MET_UCHAR";
                }
              else if (std::numeric_limits<T>::digits == 16)
                {
                  elementType = "MET_USHORT";
                }
              else if (std::numeric_limits<T>::digits == 32)
                {
                  elementType = "MET_UINT";
                }	    
            }
        }
      else
        {
          if (std::numeric_limits<T>::digits == 24)
            {
              elementType = "MET_FLOAT";
            }
          else if (std::numeric_limits<T>::digits == 53)
            {
              elementType = "MET_DOUBLE";
            }
        }
    } 
    return elementType;
  }
  
  template <class T>
  static
  void writeMETAHeaderSlice(const Array3D<T>& array,
                            Array3DIO::SliceOrientation orientation,
                            unsigned int sliceNumber,
                            const char* headerFilename,
                            const char* dataFilename,
                            bool threeDimensional = false)
  {
    std::ofstream out(headerFilename);
    if (out.bad())
      {
	throw std::runtime_error("error opening file");
      }
    
    //
    // figure out if we are little or big endian
    //
    int x = 1;
    std::string isBigEndian = (*(char*) &x == 1 ? "False" : "True");

    //
    // figure out what the data type is
    //
    std::string elementType = getMETAElementType(array);

    if(threeDimensional) {
      out << "ObjectType = Image" << std::endl
          << "NDims = 3" << std::endl
          << "BinaryData = True" << std::endl
          << "BinaryDataByteOrderMSB = "<< isBigEndian << std::endl
          << "Rotation = 1 0 0 0 1 0 0 0 1" << std::endl
          << "Offset = 0.0 0.0 0.0" << std::endl
          << "ElementSpacing = 1.0 1.0 1.0" << std::endl
          << "DimSize = ";

      if (orientation == SliceOrientationX)
        {
          out << array.getSizeY() << " "  
              << array.getSizeZ() << " 1" << std::endl;
        }
      else if (orientation == SliceOrientationY)
        {
          out << array.getSizeX() << " "  
              << array.getSizeZ() << " 1" << std::endl;
        }
      else 
        {
          out << array.getSizeX() << " "  
              << array.getSizeY() << " 1" << std::endl;
        }
    }
    else {
      out << "ObjectType = Image" << std::endl
          << "NDims = 2" << std::endl
          << "BinaryData = True" << std::endl
          << "BinaryDataByteOrderMSB = "<< isBigEndian << std::endl
          << "Offset = 0.0 0.0" << std::endl
          << "ElementSpacing = 1.0 1.0" << std::endl
          << "DimSize = ";

      if (orientation == SliceOrientationX)
        {
          out << array.getSizeY() << " "  
              << array.getSizeZ() << std::endl;
        }
      else if (orientation == SliceOrientationY)
        {
          out << array.getSizeX() << " "  
              << array.getSizeZ() << std::endl;
        }
      else 
        {
          out << array.getSizeX() << " "  
              << array.getSizeY() << std::endl;
        }
    }

    out << "ElementType = " << elementType << std::endl
	<< "ElementDataFile = " << dataFilename << std::endl;
    out.close();	    
  }

  //
  // write the META header file for this array
  //
  // bcd 2004
  //
  template <class T>
  static
  void writeMETAHeaderVolume(const Array3D<T>& array,
                             const Vector3D<double>& origin,
                             const Vector3D<double>& spacing,
			     const char* headerFilename,
			     const char* dataFilename)
  {
    std::ofstream out(headerFilename);
    if (out.bad())
      {
	throw std::runtime_error("error opening file");
      }
    
    //
    // figure out if we are little or big endian
    //
    int x = 1;
    std::string isBigEndian = (*(char*) &x == 1 ? "False" : "True");

    //
    // figure out what the data type is
    //
    std::string elementType = getMETAElementType(array);
    
    out << "ObjectType = Image" << std::endl
	<< "NDims = 3" << std::endl
	<< "BinaryData = True" << std::endl
	<< "BinaryDataByteOrderMSB = "<< isBigEndian << std::endl
	<< "Offset = " 
        << origin.x << " " 
        << origin.y << " " 
        << origin.z << std::endl
	<< "ElementSpacing = " 
        << spacing.x << " " 
        << spacing.y << " " 
        << spacing.z << std::endl
	<< "DimSize = " 
	<< array.getSizeX() << " "  
	<< array.getSizeY() << " "  
	<< array.getSizeZ() << std::endl
	<< "ElementType = " << elementType << std::endl
	<< "ElementDataFile = " << dataFilename << std::endl;
    out.close();	    
  }
  
  //
  // write the META header file and data file for an X slice of this
  // array
  //
  // bcd 2004
  //
  template <class T>
  static
  void writeMETASliceX(const Array3D<T>& array,
                       unsigned int slice,
		       const char* filenamePrefix,
                       bool threeDimensional = false)
  {
    std::string headerFilename = filenamePrefix;
    headerFilename += ".mhd";
    std::string dataFilename   = filenamePrefix;
    dataFilename += ".raw";

    writeRawSliceX(array, slice, dataFilename.c_str());
    writeMETAHeaderSlice(array, 
                         SliceOrientationX,
                         slice,
                         headerFilename.c_str(),
                         dataFilename.c_str(),
                         threeDimensional);    
  }

  //
  // write the META header file and data file for a Y slice of this
  // array
  //
  // bcd 2004
  //
  template <class T>
  static
  void writeMETASliceY(const Array3D<T>& array,
                       unsigned int slice,
		       const char* filenamePrefix,
                       bool threeDimensional = false)

  {
    std::string headerFilename = filenamePrefix;
    headerFilename += ".mhd";
    std::string dataFilename   = filenamePrefix;
    dataFilename += ".raw";

    writeRawSliceY(array, slice, dataFilename.c_str());
    writeMETAHeaderSlice(array,
                         SliceOrientationY,
                         slice,
                         headerFilename.c_str(),
                         dataFilename.c_str(),
                         threeDimensional);    
  }

  //
  // write the META header file and data file for a Z slice of this
  // array
  //
  // bcd 2004
  //
  template <class T>
  static
  void writeMETASliceZ(const Array3D<T>& array,
                       unsigned int slice,
		       const char* filenamePrefix,
                       bool threeDimensional = false)

  {
    std::string headerFilename = filenamePrefix;
    headerFilename += ".mhd";
    std::string dataFilename   = filenamePrefix;
    dataFilename += ".raw";

    writeRawSliceZ(array, slice, dataFilename.c_str());
    writeMETAHeaderSlice(array,
                         SliceOrientationZ,
                         slice,
                         headerFilename.c_str(),
                         dataFilename.c_str(),
                         threeDimensional);    
  }

  //
  // write the META header file and data file for this array
  //
  // bcd 2004
  //
  template <class T>
  static
  void writeMETAVolume(const Array3D<T>& array,
		       const char* filenamePrefix)
  {
    writeMETAVolume(array, 
                    Vector3D<double>(0.0, 0.0, 0.0),
                    Vector3D<double>(1.0, 1.0, 1.0),
                    filenamePrefix);
  }

  //
  // write the META header file and data file for this array, specify
  // origin and spacing
  //
  // bcd 2004
  //
  template <class T>
  static
  void writeMETAVolume(const Array3D<T>& array,
                       const Vector3D<double>& origin,
                       const Vector3D<double>& spacing,
		       const char* filenamePrefix)
  {
    std::string headerFilename = filenamePrefix;
    headerFilename += ".mhd";
    std::string dataFilename   = filenamePrefix;
    dataFilename += ".raw";

    writeRawVolume(array, dataFilename.c_str());
    writeMETAHeaderVolume(array, 
                          origin,
                          spacing,
			  headerFilename.c_str(),
			  StringUtils::getPathFile(dataFilename).c_str());    
  }

  // Specialize for complex images
  // We'll output two Real images, separating the real and complex
  // parts
  // jdh 2011
  static
  void writeMETAVolume(const Array3D<Complex>& array,
                       const Vector3D<double>& origin,
                       const Vector3D<double>& spacing,
		       const char* filenamePrefix)
  {
    Array3D<Real> realArray;
    realArray.resize(array.getSize());

    // convert to real and write
    std::string realHeaderFilename = filenamePrefix;
    realHeaderFilename += ".real.mhd";
    std::string realDataFilename   = filenamePrefix;
    realDataFilename += ".real.raw";

    for (unsigned int z = 0; z < array.getSizeZ(); ++z)
      for (unsigned int y = 0; y < array.getSizeY(); ++y)
        for (unsigned int x = 0; x < array.getSizeX(); ++x)
          realArray.set(x,y,z, real(array.get(x,y,z)));

    writeRawVolume(realArray, realDataFilename.c_str());
    writeMETAHeaderVolume(realArray, 
                          origin,
                          spacing,
			  realHeaderFilename.c_str(),
			  StringUtils::getPathFile(realDataFilename).c_str());

    // convert to image and write
    std::string imagHeaderFilename = filenamePrefix;
    imagHeaderFilename += ".imag.mhd";
    std::string imagDataFilename   = filenamePrefix;
    imagDataFilename += ".imag.raw";

    for (unsigned int z = 0; z < array.getSizeZ(); ++z)
      for (unsigned int y = 0; y < array.getSizeY(); ++y)
        for (unsigned int x = 0; x < array.getSizeX(); ++x)
          realArray.set(x,y,z, imag(array.get(x,y,z)));

    writeRawVolume(realArray, imagDataFilename.c_str());
    writeMETAHeaderVolume(realArray, 
                          origin,
                          spacing,
			  imagHeaderFilename.c_str(),
			  StringUtils::getPathFile(imagDataFilename).c_str());
  }
}; // class Array3DIO

#endif
