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

#ifndef __CUDA_INDEX_ARRAY__H
#define __CUDA_INDEX_ARRAY__H

#include <libDefine.h>

struct cplIndex3DArray {
    cplIndex3DArray():m_memType(undef_mem),
                       aIdx(NULL), bIdx(NULL), cIdx(NULL) {
    }

    cplIndex3DArray(int size, mem_type mtype):n(size){
        allocate(n, mtype);
    }

    mem_type  m_memType;
    uint *aIdx, *bIdx, *cIdx;
    unsigned int n;
    unsigned int nAlign;

    void allocate(int aN, mem_type mtype);
    
    // TODO Make a streaming verion of this 
    void copy(const cplIndex3DArray& a);
};


inline void allocateDeviceVector3DArray(cplIndex3DArray& d_i, int n){
    d_i.allocate(n, gpu_mem);
}

#endif
