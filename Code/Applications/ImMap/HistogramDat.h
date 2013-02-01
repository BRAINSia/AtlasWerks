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

#ifndef HISTOGRAM_DAT_H
#define HISTOGRAM_DAT_H

#include <vector>

class HistogramDat{

public :
  HistogramDat();
  ~HistogramDat();
  
  void setRelativeMin(float relMin);
  void setRelativeMax(float relMax);
  void setAbsoluteMinMax(float min, float max);
  
  float getRelativeMin();
  float getRelativeMax();
  float getAbsoluteMin();
  float getAbsoluteMax();

  std::vector<double> getHistogram();
  void setHistogram(std::vector<double> _histogram);

private :

  std::vector<double> histogram;
  
  float relativeMin, relativeMax;
  float absoluteMin, absoluteMax;
};

#endif
