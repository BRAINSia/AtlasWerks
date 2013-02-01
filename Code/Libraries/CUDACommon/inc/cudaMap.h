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

#ifndef __CUDA__MAP__H
#define __CUDA__MAP__H

#include "typeConvert.h"

template<typename T>
void cplScatter(T*d_o, T* d_i, uint* d_iPos, uint n, cudaStream_t stream=NULL);

template<typename T>
void cplScatter(T*d_o, T* d_i, uint* d_iPos, uint n, int nPar, cudaStream_t stream=NULL);


/*----------------------------------------------------------------------
  Mapping function (backward mapping) that look up value from from input
           d_o[i] = d_i[d_iPos[i]];
  Inputs : d_i    : input array
           d_iPos : mapping position

           no     : size of the output (n)
           ni     : size of the input

           nPar   : for multipass algorithm 
  Output :
           d_o    : Scatter output  
  ---------------------------------------------------------------------*/

template <typename T>
void cplMap(T*d_o, T* d_i, uint* d_iPos, uint n, cudaStream_t stream=NULL);

template<typename T>
void cplMap_tex(T* d_o, T* d_i, uint* d_iPos, uint n, cudaStream_t stream=NULL);

template<typename T>
void cplMap(T* d_o, T* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream=NULL);


/*----------------------------------------------------------------------
  Ectract function base on texture that look up value from input
           d_o[i] = d_i[d_iPos[i]] if    d_iPos[i] < ni
                    keep           other wise 

  Inputs : d_i    : input array
           d_pos  : position array that have same size with output (no)
           
           no     : size of the output
           ni     : size of the input

  Output :
           d_o    : Extract output   
  ---------------------------------------------------------------------*/
template<typename T>
void extract_tex(T*  d_o, T*  d_i, uint* d_pos, uint no, uint ni,cudaStream_t stream=NULL);

template<typename T>
void addExtract_tex(T* d_o, T* d_i, uint* last, uint no, uint ni,cudaStream_t stream=NULL);

void extract4fTof4(float4* d_o,
                   float*  d_ix, float*  d_iy, float*  d_iz, float*  d_iw,
                   uint* d_iPos, uint no, uint ni,cudaStream_t stream=NULL);

/*----------------------------------------------------------------------
  Test functions
  ---------------------------------------------------------------------*/
void testMergeSplit( int argc, char** argv);
void testSparseRead(int n, int s);
void testScatter(int n);
void testGather(int n);

#endif
