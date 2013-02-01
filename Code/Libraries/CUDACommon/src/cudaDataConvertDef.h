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

#include <cutil_comfunc.h>
#include <cplMacro.h>
#include <cudaDataConvert.h>

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T, class T2>
__global__ void convertXYtoX_Y_kernel(T* d_x, T* d_y, T2* d_xy, uint n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);
    if (id < n){
        volatile T2 data = d_xy[id];
        d_x[id] = data.x;
        d_y[id] = data.y;
    }
}

template<class T, class T2>
void convertXYtoX_Y(T* d_x, T* d_y, T2* d_xy, uint n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    convertXYtoX_Y_kernel<<<grids, threads,0, stream>>>(d_x, d_y, d_xy, n);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T2, class T>
__global__ void convertX_YtoXY_kernel(T2* d_xy, T* d_x, T* d_y, uint n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n){
        T2 data;
        data.x = d_x[id];
        data.y = d_y[id];
        
        d_xy[id]= data;
    }
}

template<class T2, class T>
void convertX_YtoXY(T2* d_xy, T* d_x, T* d_y, uint n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    convertX_YtoXY_kernel<T2, T><<<grids, threads,0, stream>>>(d_xy, d_x, d_y, n);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

template<class T, class T3>
__global__ void convertXYZtoX_Y_Z_kernel(T* d_x, T* d_y, T* d_z, T3* d_xyz, uint n){
    uint blockId = get_blockID();
#if 0
    uint id      = get_threadID(blockId);
    if (id < n){
        d_x[id] = d_xyz[id].x;
        d_y[id] = d_xyz[id].y;
        d_z[id] = d_xyz[id].z;
    }
#else
    __shared__ T sdata[256 * 3];
    uint id =  blockId * blockDim.x * 3 + threadIdx.x;
    T* d_i = (T*) d_xyz;
    if (id < n * 3)
        sdata[threadIdx.x] = d_i[id];

    if (id + blockDim.x < n * 3)
        sdata[threadIdx.x + blockDim.x] = d_i[id + blockDim.x ];

    if (id + 2 * blockDim.x < n * 3)
        sdata[threadIdx.x + 2 * blockDim.x] = d_i[id + 2 * blockDim.x];

    __syncthreads();

    id = blockId * blockDim.x + threadIdx.x;
    if (id < n){
        d_x[id] = sdata[threadIdx.x * 3    ];
        d_y[id] = sdata[threadIdx.x * 3 + 1];
        d_z[id] = sdata[threadIdx.x * 3 + 2];
    }
#endif
}

template<class T, class T3>
void convertXYZtoX_Y_Z(T* d_x, T* d_y, T* d_z, T3* d_xyz, uint n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    convertXYZtoX_Y_Z_kernel<<<grids, threads,0, stream>>>(d_x, d_y, d_z, d_xyz, n);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T3, class T>
__global__ void convertX_Y_ZtoXYZ_kernel(T3* d_xyz, T* d_x, T* d_y, T* d_z, uint n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);
#if 0
    if (id < n){
        T3 data;
        data.x = d_x[id];
        data.y = d_y[id];
        data.z = d_z[id];
        
        d_xyz[id]= data;
    }
#else
    __shared__ T sdata[256 * 3];
    if (id < n){
        sdata[threadIdx.x * 3    ] = d_x[id];
        sdata[threadIdx.x * 3 + 1] = d_y[id];
        sdata[threadIdx.x * 3 + 2] = d_z[id];
    }
    __syncthreads();
    
    T* d_o  = (T*) (d_xyz);
    uint n3 = n * 3;
    id = blockId * blockDim.x * 3 + threadIdx.x;
    if (id < n3)
        d_o[id] = sdata[threadIdx.x ];
    
    if (id + blockDim.x)
        d_o[id + blockDim.x] = sdata[threadIdx.x+blockDim.x];
    
    if (id + 2 * blockDim.x)
        d_o[id + 2 * blockDim.x] = sdata[threadIdx.x+ 2 * blockDim.x];
#endif
}

template<class T3, class T>
void convertX_Y_ZtoXYZ(T3* d_xyz, T* d_x, T* d_y, T* d_z, uint n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    convertX_Y_ZtoXYZ_kernel<<<grids, threads,0, stream>>>(d_xyz, d_x, d_y, d_z, n);
}


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T2, class T, bool overflow>
__global__ void convertX_YtoXY_kernel(T2* d_xy, T* d_x, T* d_y, uint n, uint totalBlocks){
    uint blockId = blockIdx.x;
    uint id      = get_threadID(blockId);
    
    while(!overflow || id < n){
        T2 data;
        data.x = d_x[id];
        data.y = d_y[id];
        
        d_xy[id]= data;

        if (overflow)
            id += gridDim.x * blockDim.x;
        else
            break;
    }
}

template<class T2, class T>
void convertX_YtoXY_new(T2* d_xy, T* d_x, T* d_y, uint n, cudaStream_t stream){
    dim3 threads(256);
    uint nBlocks  = iDivUp(n, threads.x);
    bool overflow = nBlocks > 65535;
    
    if (overflow){
        convertX_YtoXY_kernel<T2, T, true><<<65535, threads,0, stream>>>(d_xy, d_x, d_y, n, nBlocks);
    }else {
        convertX_YtoXY_kernel<T2, T, false><<<nBlocks, threads,0, stream>>>(d_xy, d_x, d_y, n, nBlocks);
    }
}


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T, class T4>
__global__ void convertXYZWtoX_Y_Z_W_kernel(T* d_x, T* d_y, T* d_z, T* d_w, T4* d_xyzw, uint n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);
    
    if (id < n){
        volatile T4 data = d_xyzw[id];
        
        d_x[id] = data.x;
        d_y[id] = data.y;
        d_z[id] = data.z;
        d_w[id] = data.w;
    }
}

template<class T, class T4>
void convertXYZWtoX_Y_Z_W(T* d_x, T* d_y, T* d_z, T* d_w, T4* d_xyzw, uint n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    convertXYZWtoX_Y_Z_W_kernel<<<grids, threads,0, stream>>>(d_x, d_y, d_z, d_w, d_xyzw, n);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T4, class T>
__global__ void convertX_Y_Z_WtoXYZW_kernel(T4* d_xyzw, T* d_x, T* d_y, T* d_z, T* d_w, uint n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n){
        T4 data;
        data.x = d_x[id];
        data.y = d_y[id];
        data.z = d_z[id];
        data.w = d_w[id];
        
        d_xyzw[id]= data;
    }

}

template<class T4, class T>
void convertX_Y_Z_WtoXYZW(T4* d_xyzw, T* d_x, T* d_y, T* d_z, T* d_w, uint n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    convertX_Y_Z_WtoXYZW_kernel<<<grids, threads,0, stream>>>(d_xyzw, d_x, d_y, d_z, d_w, n);
}


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T4, class T>
__global__ void convertX_Y_ZtoXYZW_kernel(T4* d_xyzw, T* d_x, T* d_y, T* d_z, uint n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n){
        T4 data;
        data.x = d_x[id];
        data.y = d_y[id];
        data.z = d_z[id];
        data.w = 1.f;
        
        d_xyzw[id]= data;
    }

}

template<class T4, class T>
void convertX_Y_ZtoXYZW(T4* d_xyzw, T* d_x, T* d_y, T* d_z, uint n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    convertX_Y_ZtoXYZW_kernel<<<grids, threads,0, stream>>>(d_xyzw, d_x, d_y, d_z, n);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T, class T4>
__global__ void convertXYZWtoX_Y_Z_kernel(T* d_x, T* d_y, T* d_z, T4* d_xyzw, uint n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);
    
    if (id < n){
        volatile T4 data = d_xyzw[id];
        
        d_x[id] = data.x;
        d_y[id] = data.y;
        d_z[id] = data.z;
    }
}

template<class T, class T4>
void convertXYZWtoX_Y_Z(T* d_x, T* d_y, T* d_z, T4* d_xyzw, uint n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    convertXYZWtoX_Y_Z_kernel<<<grids, threads,0, stream>>>(d_x, d_y, d_z, d_xyzw, n);
}
