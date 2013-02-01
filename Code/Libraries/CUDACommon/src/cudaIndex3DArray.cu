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

#include <cudaIndex3DArray.h>
#include <cudaInterface.h>
#include <cutil_comfunc.h>

void cplIndex3DArray::allocate(int aN, mem_type mtype){

    m_memType = mtype;
    n = aN;
    nAlign = iAlignUp(n, 128);
    
    if (m_memType == gpu_mem)
        dmemAlloc(aIdx, nAlign * 3);
    else
        aIdx = new uint [nAlign * 3];
    
    bIdx = aIdx + nAlign;
    cIdx = aIdx + 2 * nAlign;
}

// TODO Make a streaming verion of this 
void cplIndex3DArray::copy(const cplIndex3DArray& a){
    if (m_memType == cpu_mem){
        if (a.m_memType == cpu_mem)
            memcpy(aIdx, a.aIdx, nAlign * 3 * sizeof(int));
        else
            copyArrayFromDevice(aIdx, a.aIdx, nAlign * 3);
    } else {
        if (a.m_memType == gpu_mem)
            copyArrayDeviceToDevice(aIdx, a.aIdx, nAlign * 3);
        else
            copyArrayToDevice(aIdx, a.aIdx, nAlign * 3);
    }
}
