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

#ifndef __CUDA_SEGMENTED_SCAN_UTILS_H
#define __CUDA_SEGMENTED_SCAN_UTILS_H

void buildSegmentedFlagsFromPos(uint* d_flags, int n,
                                uint* d_pos, int nSegs, cudaStream_t stream=cudaStream_t(NULL));

void buildSegmentedFlags_share(uint* d_f, uint* d_i, int n, cudaStream_t stream=cudaStream_t(NULL));
void buildSegmentedFlags(uint* d_f, uint* d_i, int n, cudaStream_t stream=cudaStream_t(NULL));
void buildSegmentedFlags_tex(uint* d_f, uint* d_i, int n, cudaStream_t stream=cudaStream_t(NULL));

void findLastPos(uint* d_o, uint* d_i, uint n, cudaStream_t stream=cudaStream_t(NULL));
void findLastPos(uint* g_pos, uint4* g_iData, uint n);

void compressFlags(uint* g_flags, uint* g_cnt, uint n, cudaStream_t stream=cudaStream_t(NULL));

void cplCompressFlags(uint* d_of, uint* d_if, int n, cudaStream_t stream=cudaStream_t(NULL));
void cplDecompressFlags(uint* d_of, uint* d_if, int n, cudaStream_t stream=cudaStream_t(NULL));

void cudaSegmentedScanUtilsTest(int n, int s);

#endif
