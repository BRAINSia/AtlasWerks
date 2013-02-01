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

#include "HistogramDat.h"
#include <iostream>

HistogramDat
::HistogramDat()
{
  relativeMin = relativeMax = 0.0;
  absoluteMin = absoluteMax = 0.0;
}

HistogramDat
::~HistogramDat()
{
}

void
HistogramDat
::setRelativeMin(float relMin)
{
  relativeMin=relMin;
}

void
HistogramDat
::setRelativeMax(float relMax)
{
  relativeMax=relMax;
}

void
HistogramDat
::setAbsoluteMinMax(float min, float max)
{
  absoluteMin=relativeMin=min;
  absoluteMax=relativeMax=max;

  int newSize; 
  if(max<2)
  	newSize = static_cast<int>((max-min+1)*100);
  else	
        newSize = static_cast<int>(max-min+1);

  //int newSize = static_cast<int>(max-min+1);

  // delete old vector
  histogram.clear();
  
  histogram.resize(newSize , 0 );
}

float
HistogramDat
::getRelativeMin()
{
  return relativeMin;
}

float
HistogramDat
::getRelativeMax()
{
  return relativeMax;
}

float
HistogramDat
::getAbsoluteMin()
{
  return absoluteMin;
}

float
HistogramDat
::getAbsoluteMax()
{
  return absoluteMax;
}

std::vector<double> 
HistogramDat
::getHistogram()
{
  return histogram;
}

void 
HistogramDat
::setHistogram(std::vector<double> _histogram)
{
  histogram=_histogram;
}
