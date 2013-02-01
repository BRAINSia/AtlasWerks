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

//////////////////////////////////////////////////////////////////////
//
// File: ROI.h
//
// The class represents a "region of interest" and is internally
// represented by "start index" and a "size".
//
//////////////////////////////////////////////////////////////////////
#ifndef ROI_H
#define ROI_H

#include <iostream>

#include "Vector3D.h"

//////////////////////////////
// Region Of Interest Class //
//////////////////////////////

// A region of interest is defined by 'start' and 'size'.  The value
// 'stop' is the sum of both and cannot be set at initialization.
// When you set 'size', it updates 'stop', and when you set 'stop', it
// updates 'size'.

template <class IndexType, class SizeType>
class ROI
{
public :
  typedef Vector3D<IndexType>  ROIIndexType;
  typedef Vector3D<SizeType>   ROISizeType;

  ROI();
  ROI(const ROIIndexType& start, const ROISizeType& size);
  ROI(const ROI& rhs);
  ROI& operator=(const ROI& rhs);

  ROIIndexType getStart() const;
  IndexType getStartX() const;
  IndexType getStartY() const;
  IndexType getStartZ() const;

  // Setting the start index does not change the size
  void setStart(const ROIIndexType& start);
  void setStart(const IndexType& x, const IndexType& y, const IndexType& z);
  void setStartX(const IndexType& x);
  void setStartY(const IndexType& y);
  void setStartZ(const IndexType& z);

  ROISizeType getSize()  const;
  SizeType getSizeX() const;
  SizeType getSizeY() const;
  SizeType getSizeZ() const;

  // Setting the size index does not change the start index
  void setSize(const ROISizeType& size);
  void setSize(const SizeType& x, 
               const SizeType& y, 
               const SizeType& z);
  void setSizeX(const SizeType& x);
  void setSizeY(const SizeType& y);
  void setSizeZ(const SizeType& z);

  ROIIndexType getStop()  const;
  IndexType getStopX() const;
  IndexType getStopY() const;
  IndexType getStopZ() const;

  // Setting the stop index (potentially) changes the size
  void setStop(const ROIIndexType& stop);
  void setStop(const IndexType& x, const IndexType& y, const IndexType& z);  
  void setStopX(const IndexType& x);
  void setStopY(const IndexType& y);
  void setStopZ(const IndexType& z);

protected :
  ROIIndexType _start;
  ROISizeType _size;

}; // class ROI
#include "ROI.txx"
#endif
