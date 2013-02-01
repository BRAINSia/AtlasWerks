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

#ifndef __ASYN_IO_H
#define __ASYN_IO_H


template<typename T>
void readArrayFromRawFile(T* h_a, const char* fname, unsigned int n);

template<typename T>
void writeArrayToRawFile(T* h_a, const char* fname, unsigned int n);

template<typename T>
void readArrayFromRawFileAsyn(T* h_a, const char* fname, unsigned int n);


template<typename T>
void writeArrayToRawFileAsyn(T* h_a, const char* fname, unsigned int n);

#endif
