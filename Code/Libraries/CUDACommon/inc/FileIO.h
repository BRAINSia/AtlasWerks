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

#ifndef __FILEIO_H__
#define __FILEIO_H__

#include <vector>
#include <string>
#include <cpuImage3D.h>

void writePGM(char* rawFile, unsigned char* data, int width, int height);
void writeRawVolumeData_f(char* rawFile, float* data, int len);
void writeToNrrd(float* data, int w, int h, int l, char *name);
void readRawVolumeData(char* rawFile, float* data, int len);

#endif
