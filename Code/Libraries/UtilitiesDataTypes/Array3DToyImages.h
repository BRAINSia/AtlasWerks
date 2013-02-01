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

#ifndef ARRAY3D_TOY_IMAGES_H
#define ARRAY3D_TOY_IMAGES_H

class Array3DToyImages
{
public:
  //
  // Creates a regular grid
  //
  // pjl 2004
  //
  template <class T>
  static
  void createGrid(Array3D<T>& array,
		  const typename Array3D<T>::SizeType& arraySize,
		  const typename Array3D<T>::SizeType& gridSpacing,
		  const T& minValue = 0,
		  const T& maxValue = 255)
  {
    array.resize(arraySize);
    for(unsigned int zIndex = 0; zIndex < arraySize.z; zIndex++) {
      for(unsigned int yIndex = 0; yIndex < arraySize.y; yIndex++) {
	for(unsigned int xIndex = 0; xIndex < arraySize.x; xIndex++) {
	  if(((zIndex % gridSpacing.z == 0) && 
	      (yIndex % gridSpacing.y == 0))     ||
	     ((zIndex % gridSpacing.z == 0) && 
	      (xIndex % gridSpacing.x == 0))     ||
	     ((xIndex % gridSpacing.x == 0) && 
	      (yIndex % gridSpacing.y == 0))) {
	    // line region
	    array(xIndex, yIndex, zIndex) = maxValue;
	  }
	  else 
          {
	    // non-line region
	    array(xIndex, yIndex, zIndex) = minValue;
	  }
	}
      }
    }
  }

  template <class T>
  static
  void createCellGrid(Array3D<T>& array,
                      const typename Array3D<T>::SizeType& arraySize,
                      const typename Array3D<T>::SizeType& gridSpacing,
                      const T& minValue = 0,
                      const T& maxValue = 255)
  {
    typedef T VoxelType;
    VoxelType increment = (maxValue - minValue) / 3;
    array.resize(arraySize);
    for(unsigned int zIndex = 0; zIndex < arraySize.z; zIndex++) {
      for(unsigned int yIndex = 0; yIndex < arraySize.y; yIndex++) {
	for(unsigned int xIndex = 0; xIndex < arraySize.x; xIndex++) {
          VoxelType value = minValue;
	  if (zIndex % gridSpacing.z == 0) value += increment;
	  if (yIndex % gridSpacing.y == 0) value += increment;
	  if (xIndex % gridSpacing.x == 0) value += increment;
          if (value > maxValue) value = maxValue;
          array(xIndex, yIndex, zIndex) = value;
        }
      }
    }
  }

  template <class T>
  static
  void
  fillSphere(Array3D<T>& array, 
	     double diameter=2.0/3.0,
	     float sphereVal = 0.0f) 
  {
    Vector3D<unsigned int> size   = array.getSize();
    Vector3D<double> center((double)(size.x)/2,
                            (double)(size.y)/2,
                            (double)(size.z)/2);
    double radius = double(size.minElement()) * diameter / 2.0;
    for (unsigned int z = 0; z < size.z; ++z) {
      for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {        
          Vector3D<double> here((double)x, (double)y, (double)z);
          if (here.distance(center) < radius)
            {
              array(x,y,z) = static_cast<T>(sphereVal);
            }
        }
      }
    }
  }

  // diameter is given as a fraction of the smallest dimension of the
  // image
  template <class T>
  static
  void
  createSphere(Array3D<T>& array, 
               double diameter = 2.0/3.0,
               float background = 1.0f,
               float foreground = 0.0f) 
  {
    array.fill(static_cast<T>(background));
    fillSphere(array, diameter, foreground);
  }

  template <class T>
  static
  void
  createGradientX(Array3D<T>& array)
  {
    Vector3D<unsigned int> size = array.getSize();
    for (unsigned int z = 0; z < size.z; ++z) {
      for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {        
          array(x,y,z) = static_cast<T>(x);
        }
      }
    }
  }

  template <class T>
  static
  void
  createGradientY(Array3D<T>& array)
  {
    Vector3D<unsigned int> size = array.getSize();
    for (unsigned int z = 0; z < size.z; ++z) {
      for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {        
          array(x,y,z) = static_cast<T>(y);
        }
      }
    }
  }

  template <class T>
  static
  void
  createGradientZ(Array3D<T>& array)
  {
    Vector3D<unsigned int> size = array.getSize();
    for (unsigned int z = 0; z < size.z; ++z) {
      for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {        
          array(x,y,z) = static_cast<T>(z);
        }
      }
    }
  }

  template <class T>
  static
  void
  createConesTestImage(Array3D<T>& array)
  {
    Vector3D<unsigned int> size = array.getSize();
    array.fill(0.0F);

    for (unsigned int x = 0; x < size.x; ++x) {
      int startY = 
        int(float(size.y)/4.0F - 
            (float(size.y)/2.0F*float(x)/float(size.x))/2.0F);
      int stopY = 
        int(float(size.y)/4.0F + 
            (float(size.y)/2.0F*float(x)/float(size.x))/2.0F);
      if (startY < 0) startY = 0;
      if (stopY >= int(size.y)) stopY = int(size.y - 1);
      for (int y = startY; y < stopY; ++y) {
        int startZ = 
          int(float(size.z)/4.0F - 
              (float(size.z)/2.0F*float(x)/float(size.x))/2.0F);
        int stopZ = 
          int(float(size.z)/4.0F + 
              (float(size.z)/2.0F*float(x)/float(size.x))/2.0F);
        if (startZ < 0) startZ = 0;
        if (stopZ >= int(size.z)) stopZ = int(size.z - 1);
        for (int z = startZ; z < stopZ; ++z) {
          array(x,y,z) = 0.25F;
        }
      }
    }

    for (unsigned int y = 0; y < size.y; ++y) {
      int startX = 
        int(float(size.x)*3.0F/4.0F - 
            (float(size.x)/2.0F*float(y)/float(size.y))/2.0F);
      int stopX = 
        int(float(size.x)*3.0F/4.0F + 
            (float(size.x)/2.0F*float(y)/float(size.y))/2.0F);
      if (startX < 0) startX = 0;
      if (stopX >= int(size.x)) stopX = int(size.x - 1);
      for (int x = startX; x < stopX; ++x) {
        int startZ = 
          int(float(size.z)*3.0F/4.0F - 
              (float(size.z)/2.0F*float(y)/float(size.y))/2.0F);
        int stopZ = 
          int(float(size.z)*3.0F/4.0F + 
              (float(size.z)/2.0F*float(y)/float(size.y))/2.0F);
        if (startZ < 0) startZ = 0;
        if (stopZ >= int(size.z)) stopZ = int(size.z - 1);
        for (int z = startZ; z < stopZ; ++z) {
          array(x,y,z) = 0.5F;
        }
      }
    }

    for (unsigned int z = 0; z < size.z; ++z) {
      int startX = 
        int(float(size.x)/4.0F - 
            (float(size.x)/2.0F*float(z)/float(size.z))/2.0F);
      int stopX = 
        int(float(size.x)/4.0F + 
            (float(size.x)/2.0F*float(z)/float(size.z))/2.0F);
      if (startX < 0) startX = 0;
      if (stopX >= int(size.x)) stopX = int(size.x - 1);
      for (int x = startX; x < stopX; ++x) {
        int startY = 
          int(float(size.y)*3.0F/4.0F - 
              (float(size.y)/2.0F*float(z)/float(size.z))/2.0F);
        int stopY = 
          int(float(size.y)*3.0F/4.0F + 
              (float(size.y)/2.0F*float(z)/float(size.z))/2.0F);
        if (startY < 0) startY = 0;
        if (stopY >= int(size.y)) stopY = int(size.y - 1);
        for (int y = startY; y < stopY; ++y) {
          array(x,y,z) = 0.75F;
        }
      }
    }
  }
};

#endif
