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

#ifndef __NDDR_UTILS_H
#define __NDDR_UTILS_H

#include<Vector3D.h>

void readRawVolumeData(char* rawFile, float* data, int len);
void writeToNrrd(float* data, int w, int h, int l, const char* name);

void writeDeviceToNrrd(float* d_data, int w, int h, int l, const char* name);
void writeDeviceToNrrd(float* d_data, const Vector3Di& size, const char* name);



#endif
