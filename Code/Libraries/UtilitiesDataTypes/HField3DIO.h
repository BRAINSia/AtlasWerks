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

#ifndef HFIELD3D_IO_H
#define HFIELD3D_IO_H

#include "Vector3D.h"
#include "Array3D.h"
#include "Array3DIO.h"
#include "Array3DUtils.h"
#include "metaImage.h"
#include <string>
#include "StringUtils.h"

#ifdef max
#undef max
#endif

class HField3DIO
{
public:
  template <class T>
  static
  void writeMETA(const Array3D<Vector3D<T> >& h,
                 const std::string& filenamePrefix)
  {
    writeMETA(h, 
              Vector3D<double>(0.0, 0.0, 0.0),
              Vector3D<double>(1.0, 1.0, 1.0),
              filenamePrefix);
  }

  template <class T>
  static
  void writeMETAXComponent(const Array3D<Vector3D<T> >& h,
                           const std::string& filenamePrefix,
			   bool rescale = true)
  {
    Array3D<T> hx(h.getSize());
    for (unsigned int z = 0; z < h.getSize().z; ++z) {
      for (unsigned int y = 0; y < h.getSize().y; ++y) {
        for (unsigned int x = 0; x < h.getSize().x; ++x) {
          hx(x,y,z) = h(x,y,z).x;
        }
      }
    }
    if(rescale){
      Array3DUtils::rescaleElements(hx, 0.0F, 1.0F);
    }
    Array3DIO::writeMETAVolume(hx, filenamePrefix.c_str());      
  }
  template <class T>
  static
  void writeMETAXComponent(const Array3D<Vector3D<T> >& h,
                           const Vector3D<double>& origin,
                           const Vector3D<double>& spacing,
                           const std::string& filenamePrefix)
  {
    Array3D<T> hx(h.getSize());
    for (unsigned int z = 0; z < h.getSize().z; ++z) {
      for (unsigned int y = 0; y < h.getSize().y; ++y) {
        for (unsigned int x = 0; x < h.getSize().x; ++x) {
          hx(x,y,z) = h(x,y,z).x;
        }
      }
    }
    Array3DUtils::rescaleElements(hx, 0.0F, 1.0F);
    Array3DIO::writeMETAVolume(hx, origin, spacing, filenamePrefix.c_str());      
  }

  template <class T>
  static
  void writeMETAYComponent(const Array3D<Vector3D<T> >& h,
                           const std::string& filenamePrefix,
			   bool rescale = true)
  {
    Array3D<T> hx(h.getSize());
    for (unsigned int z = 0; z < h.getSize().z; ++z) {
      for (unsigned int y = 0; y < h.getSize().y; ++y) {
        for (unsigned int x = 0; x < h.getSize().x; ++x) {
          hx(x,y,z) = h(x,y,z).y;
        }
      }
    }
    if(rescale){
      Array3DUtils::rescaleElements(hx, 0.0F, 1.0F);
    }
    Array3DIO::writeMETAVolume(hx, filenamePrefix.c_str());      
  }
  template <class T>
  static
  void writeMETAYComponent(const Array3D<Vector3D<T> >& h,
                           const Vector3D<double>& origin,
                           const Vector3D<double>& spacing,
                           const std::string& filenamePrefix)
  {
    Array3D<T> hx(h.getSize());
    for (unsigned int z = 0; z < h.getSize().z; ++z) {
      for (unsigned int y = 0; y < h.getSize().y; ++y) {
        for (unsigned int x = 0; x < h.getSize().x; ++x) {
          hx(x,y,z) = h(x,y,z).y;
        }
      }
    }
    Array3DUtils::rescaleElements(hx, 0.0F, 1.0F);
    Array3DIO::writeMETAVolume(hx, origin, spacing, filenamePrefix.c_str());      
  }

  template <class T>
  static
  void writeMETAZComponent(const Array3D<Vector3D<T> >& h,
                           const std::string& filenamePrefix,
			   bool rescale = true)
  {
    Array3D<T> hx(h.getSize());
    for (unsigned int z = 0; z < h.getSize().z; ++z) {
      for (unsigned int y = 0; y < h.getSize().y; ++y) {
        for (unsigned int x = 0; x < h.getSize().x; ++x) {
          hx(x,y,z) = h(x,y,z).z;
        }
      }
    }
    if(rescale){
      Array3DUtils::rescaleElements(hx, 0.0F, 1.0F);
    }
    Array3DIO::writeMETAVolume(hx, filenamePrefix.c_str());      
  }
  template <class T>
  static
  void writeMETAZComponent(const Array3D<Vector3D<T> >& h,
                           const Vector3D<double>& origin,
                           const Vector3D<double>& spacing,
                           const std::string& filenamePrefix)
  {
    Array3D<T> hx(h.getSize());
    for (unsigned int z = 0; z < h.getSize().z; ++z) {
      for (unsigned int y = 0; y < h.getSize().y; ++y) {
        for (unsigned int x = 0; x < h.getSize().x; ++x) {
          hx(x,y,z) = h(x,y,z).z;
        }
      }
    }
    //Array3DUtils::rescaleElements(hx, 0.0F, 1.0F);
    Array3DIO::writeMETAVolume(hx, origin, spacing, filenamePrefix.c_str());      
  }

  template <class T>
  static
  void
  writeMETA(const Array3D<Vector3D<T> >& h, 
            const Vector3D<double>& origin,
            const Vector3D<double>& spacing,            
            const std::string& filenamePrefix)
  {
    std::string filename = filenamePrefix + ".mhd";
    MetaImage mi(h.getSizeX(), h.getSizeY(), h.getSizeZ(),
                 spacing.x, spacing.y, spacing.z,
                 MET_GetPixelType(typeid(T)), 
                 3, (void*) h.getDataPointer());
    mi.Origin(0, origin.x);
    mi.Origin(1, origin.y);
    mi.Origin(2, origin.z);

    mi.ObjectSubTypeName("HField");

    if (!mi.Write(filename.c_str()))
      {
        throw std::runtime_error("Error writing file.");
      }
  }

  template <class T>
  static 
  void
  readMETA(Array3D<Vector3D<T> >& h,
           Vector3D<double>& origin,
           Vector3D<double>& spacing,
           const std::string& filename)
  {
    MetaImage metaImage;
    if (!metaImage.Read(filename.c_str()))
      {
        throw std::runtime_error("Error loading file.");
      }

    if (
       // StringUtils::toUpper(std::string(metaImage.ObjectTypeName())) 
        //!= StringUtils::toUpper(std::string("Image")) ||
        //StringUtils::toUpper(std::string(metaImage.ObjectSubTypeName())) 
        //!= StringUtils::toUpper(std::string("HField")) ||
        metaImage.ElementNumberOfChannels() != 3)
      {
        throw std::runtime_error("This meta is not an HField.");
      }

    for (unsigned int i = 0; i < 3; ++i) 
    {
      origin[i] = metaImage.Origin(i);
      spacing[i] = metaImage.ElementSpacing(i);
    }

    h.resize(metaImage.DimSize(0), metaImage.DimSize(1), metaImage.DimSize(2));
    
    T* hPtr = (T*) h.getDataPointer();
    
    switch(metaImage.ElementType()) {
    case MET_UCHAR:
      {
        unsigned char* metaPtr = (unsigned char*) metaImage.ElementData();
        for(unsigned int i = 0; i < 3 * h.getNumElements(); ++i)
          {
            *hPtr++ = (T) *metaPtr++;
          }
      }
      break;
    case MET_CHAR:
      {
        char* metaPtr = (char*) metaImage.ElementData();
        for(unsigned int i = 0; i < 3 * h.getNumElements(); ++i)
          {
            *hPtr++ = (T) *metaPtr++;
          }
      }
      break;
    case MET_USHORT:
      {
      unsigned short* metaPtr = (unsigned short*) metaImage.ElementData();
      for(unsigned int i = 0; i < 3 * h.getNumElements(); ++i)
        {
          *hPtr++ = (T) *metaPtr++;
        }
      }
      break;
    case MET_SHORT:
      {
        short* metaPtr = (short*) metaImage.ElementData();
        for(unsigned int i = 0; i < 3 * h.getNumElements(); ++i)
          {
            *hPtr++ = (T) *metaPtr++;
          }
      }
      break;
    case MET_UINT:
      {
        unsigned int* metaPtr = (unsigned int*) metaImage.ElementData();
        for(unsigned int i = 0; i < 3 * h.getNumElements(); ++i)
          {
            *hPtr++ = (T) *metaPtr++;
          }
      }
      break;
    case MET_INT:
      {
        int* metaPtr = (int*) metaImage.ElementData();
        for(unsigned int i = 0; i < 3 * h.getNumElements(); ++i)
          {
            *hPtr++ = (T) *metaPtr++;
          }
      }
      break;
    case MET_ULONG:
      {
        unsigned long* metaPtr = (unsigned long*) metaImage.ElementData();
        for(unsigned int i = 0; i < 3 * h.getNumElements(); ++i)
          {
            *hPtr++ = (T) *metaPtr++;
          }
      }
      break;
    case MET_LONG:
      {
        long* metaPtr = (long*) metaImage.ElementData();
        for(unsigned int i = 0; i < 3 * h.getNumElements(); ++i)
          {
            *hPtr++ = (T) *metaPtr++;
          }
      }
      break;
    case MET_FLOAT:
      {
        float* metaPtr = (float*) metaImage.ElementData();
        for(unsigned int i = 0; i < 3 * h.getNumElements(); ++i)
          {
            *hPtr++ = (T) *metaPtr++;
          }
      }
      break;
    case MET_DOUBLE:
      {
        double* metaPtr = (double*) metaImage.ElementData();
        for(unsigned int i = 0; i < 3 * h.getNumElements(); ++i)
          {
            *hPtr++ = (T) *metaPtr++;
          }
      }
      break;
    default:
      throw std::runtime_error("unknown meta image type");
      break;
    }
  }

  template <class T>
  static 
  void
  readMETA(Array3D<Vector3D<T> >& h, const std::string& filename)
  {
    Vector3D<double> origin;
    Vector3D<double> spacing;
    readMETA(h, origin, spacing, filename);
  }

  // Just get the origin and spacing.
  static void
  readMETAHeader(Vector3D<unsigned int>& size,
                 Vector3D<double>& origin,
                 Vector3D<double>& spacing,
                 const std::string& filename)
  {
    MetaImage metaImage;
    if (!metaImage.Read(filename.c_str(), false))
    {
      throw std::runtime_error("Error loading file.");
    }

    if (
        //StringUtils::toUpper(std::string(metaImage.ObjectTypeName())) 
        //!= StringUtils::toUpper(std::string("Image")) ||
        //StringUtils::toUpper(std::string(metaImage.ObjectSubTypeName())) 
        //!= StringUtils::toUpper(std::string("HField")) ||
        metaImage.ElementNumberOfChannels() != 3)
    {
      throw std::runtime_error("This meta is not an HField.");
    }

    for (unsigned int i = 0; i < 3; ++i) 
    {
      size[i] = metaImage.DimSize(i);
      origin[i] = metaImage.Origin(i);
      spacing[i] = metaImage.ElementSpacing(i);
    }
  }

};

#endif
