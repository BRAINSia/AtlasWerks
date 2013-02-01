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

#ifndef __IMAGE_DISPLAY_H
#define __IMAGE_DISPLAY_H

void horizontalConcateImages(float* result, float* img0, float* img1, int w, int h);
void horizontalConcateImages(float* result, float* img0, float* img1, float* img2, int w, int h);
void horizontalConcateImages(float* result, float* img0, float* img1, float* img2, float* img3, int w, int h);

void cpliImageCompose4x4(float* result, float* imgArray[4][4], int w, int h);
void rectange2x2ConcateImages(float* result, float * img0, float* img1, float* img2, float* img3, int w, int h);


// change the shape of the image from w * (h * nImgs) to (w * nCols) * (h * nImgs / nCols)
void reshapeImages(float* d_result, float* d_img, int w, int h, int nImgs, int nCols);


// concate all the slices from the volumes to make w * ( h * nIm 
void concateVolumeSlices(float* d_oImg, float** d_iImgs, int nImgs, int slice,
                         unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ);

#endif
