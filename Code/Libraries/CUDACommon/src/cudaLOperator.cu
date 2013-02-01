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

#include <cudaTexFetch.h>
#include <cpl.h>
#include <cudaLOperator.h>
#include <cudaReduceStream.h>

void createTest3D(float *f, int n){
    float h = 1.f / (n-1);
    const float M_PI3 = M_PI * M_PI * M_PI;
    
    for (int k=0; k< n; ++k)
        for (int j=0; j< n; ++j)
            for (int i=0; i< n; ++i){
                float x = h * i;
                float y = h * j;
                float z = h * k;
                f[i + j * n + k * n* n] = M_PI3 * cosf(M_PI * x) * cosf(M_PI * y) * cosf(M_PI * z);
            }
}

void createTest3D(float *f, Vector3Di& size){
    float hx = 1.f / (size.x-1);
    float hy = 1.f / (size.y-1);
    float hz = 1.f / (size.z-1);
    
    const float M_PI3 = M_PI * M_PI * M_PI;
    for (int k=0; k< size.z; ++k)
        for (int j=0; j< size.y; ++j)
            for (int i=0; i< size.x; ++i){
                float x = hx * i;
                float y = hy * j;
                float z = hz * k;
                f[i + j * size.x + k * size.x * size.y] = M_PI3 * sinf(M_PI * x) * sinf(M_PI * y) * sinf(M_PI * z);
            }
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
__global__ void helmhotlz3DMulVector_tex_kernel(float* b, float* x,
                                                float alpha, float gamma,
                                                int sizeX, int sizeY, int sizeZ){
    uint xid      = threadIdx.x + blockIdx.x * blockDim.x;
    uint yid      = threadIdx.y + blockIdx.y * blockDim.y;
    uint id       = xid + yid * sizeX;
    uint planeSize= sizeX * sizeY;
    
    if (xid < sizeX && yid < sizeY){
        float zo = 0;
        float zc = fetch(id, (float*)NULL);
        float zn = fetch(id + planeSize, (float*)NULL);
        float r;
        
        // zid = 0 to n-2
        for (uint zid=0; zid<sizeZ-1; ++zid){
            r  = zo + zn;
            r += (xid   >=1)? fetch(id - 1, (float*)NULL) : 0;
            r += (xid+1 < sizeX)? fetch(id+1, (float*)NULL) : 0;
            r += (yid   >=1)? fetch(id-sizeX, (float*)NULL) : 0;
            r += (yid+1 < sizeY)? fetch(id+sizeX, (float*)NULL) : 0;
            b[id] = zc * (6 * alpha + gamma) - alpha  * r;
            
            //advance the plane
            id += planeSize;
            zo  = zc;
            zc  = zn;
            zn  = fetch(id + planeSize, (float*)NULL);
        }

        // zid = n-1
        r = zo;
        r+= (xid   >=1)? fetch(id-1, (float*)NULL) : 0;
        r+= (xid+1 < sizeX)? fetch(id+1, (float*)NULL) : 0;
        r+= (yid   >=1)? fetch(id-sizeX, (float*)NULL) : 0;
        r+= (yid+1 < sizeY)? fetch(id+sizeX, (float*)NULL) : 0;
        b[id] = zc * (6 * alpha + gamma) - alpha  * r;
    }
}

void matrixMulVector(float* b, helmholtzMatrix3D& A, float* x, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(A.size.x, threads.x),iDivUp(A.size.y, threads.y));
    cache_bind(x);
    helmhotlz3DMulVector_tex_kernel<<<grids, threads, 0, stream>>>(b, x,
                                                                   A.alpha, A.gamma,
                                                                   A.size.x, A.size.y, A.size.z);
}


////////////////////////////////////////////////////////////////////////////////
// Compute the residual r = b - Ax
////////////////////////////////////////////////////////////////////////////////
__global__ void residualHelmhotlz3D_tex_kernel(float* r, float* b, float* x,
                                               float alpha, float gamma,
                                               uint sizeX, uint sizeY, uint sizeZ)
{
    uint xid      = threadIdx.x + blockIdx.x * blockDim.x;
    uint yid      = threadIdx.y + blockIdx.y * blockDim.y;
    uint id       = xid + yid * sizeX;
    uint planeSize= sizeX * sizeY;
    
    if (xid < sizeX && yid < sizeY){
        float zo = 0;
        float zc = fetch(id, (float*)NULL);
        float zn = fetch(id + planeSize, (float*)NULL);
        float s;
        
        // zid = 0 to n-2
        for (uint zid=0; zid<sizeZ-1; ++zid){
            s  = zo + zn;
            s += (xid   >=1)? fetch(id - 1, (float*)NULL) : 0;
            s += (xid+1 < sizeX)? fetch(id+1, (float*)NULL) : 0;
            s += (yid   >=1)? fetch(id-sizeX, (float*)NULL) : 0;
            s += (yid+1 < sizeY)? fetch(id+sizeX, (float*)NULL) : 0;
            r[id] = b[id] - (zc * (6 * alpha + gamma) - alpha  * s);
            
            //advance the plane
            id += planeSize;
            zo  = zc;
            zc  = zn;
            zn  = fetch(id + planeSize, (float*)NULL);
        }

        // zid = n-1
        s = zo;
        s+= (xid   >=1)? fetch(id-1, (float*)NULL) : 0;
        s+= (xid+1 < sizeX)? fetch(id+1, (float*)NULL) : 0;
        s+= (yid   >=1)? fetch(id-sizeX, (float*)NULL) : 0;
        s+= (yid+1 < sizeY)? fetch(id+sizeX, (float*)NULL) : 0;
        r[id] = b[id] - (zc * (6 * alpha + gamma) - alpha  * s);
    }
}

void computeResidual(float* r, float* b, helmholtzMatrix3D& A, float* x, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(A.size.x, threads.x),iDivUp(A.size.y, threads.y));
    cache_bind(x);
    //cudaBindTexture(0, com_tex_float, x, sizeof(float) * A.getNumElements());
    residualHelmhotlz3D_tex_kernel<<<grids, threads, 0, stream>>>(r, b, x,
                                                                  A.alpha, A.gamma,
                                                                  A.size.x,A.size.y,A.size.z);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
__global__ void helmhotlz3DMulVector_cyclic_tex_kernel(float* b, float* x,
                                                       float alpha, float gamma,
                                                       int w, int h, int l){
    uint xid      = threadIdx.x + blockIdx.x * blockDim.x;
    uint yid      = threadIdx.y + blockIdx.y * blockDim.y;
    uint  id       = xid + yid * w;
    uint  planeSize= w * h;
    
    if (xid < w && yid < h){
        float zo, zc, zn;
        zo= fetch(id + (l-1) * planeSize, (float*)NULL);
        zc= fetch(id, (float*)NULL);
        zn= fetch(id + planeSize, (float*)NULL);

        // zid = 1 to n-2
        for (int zid=0; zid<l-1; ++zid){
            float b1 = zo + zn;
            b1+=(xid   >=1)? fetch(id-1, (float*)NULL) : fetch(id + w-1, (float*)NULL);
            b1+=(xid+1 < w)? fetch(id+1, (float*)NULL) : fetch(id + 1-w, (float*)NULL);
            b1+=(yid   >=1)? fetch(id-w, (float*)NULL) : fetch(id + h*w - w, (float*)NULL);
            b1+=(yid+1 < h)? fetch(id+w, (float*)NULL) : fetch(id + w - h*w, (float*)NULL);

            b[id] = zc * (6 * alpha + gamma) - alpha  * b1;

            id += planeSize;
            zo  = zc;
            zc  = zn;
            zn  = (zid < l-2) ? fetch(id + planeSize, (float*)NULL): fetch(xid + yid * w, (float*)NULL);
        }
        
        float b1 = zo + zn;
        b1+=(xid   >=1)? fetch(id-1, (float*)NULL) : fetch(id + w-1, (float*)NULL);
        b1+=(xid+1 < w)? fetch(id+1, (float*)NULL) : fetch(id + 1-w, (float*)NULL);
        b1+=(yid   >=1)? fetch(id-w, (float*)NULL) : fetch(id + h*w - w, (float*)NULL);
        b1+=(yid+1 < h)? fetch(id+w, (float*)NULL) : fetch(id + w - h*w, (float*)NULL);
        b[id] = zc * (6 * alpha + gamma) - alpha  * b1;
    }
}

void matrixMulVector(float* b, helmholtzMatrix3D_cyclic& A, float* x, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(A.size.x, threads.x),iDivUp(A.size.y, threads.y));
    cache_bind(x);
    //cudaBindTexture(0, com_tex_float, x, sizeof(float) * A.getNumElements());
    helmhotlz3DMulVector_cyclic_tex_kernel<<<grids, threads, 0, stream>>>(b, x,
                                                                          A.alpha, A.gamma,
                                                                          A.size.x, A.size.y, A.size.z);
}


__global__ void helmhotlz3Dresidual_cyclic_tex_kernel(float* r, float* b, float* x,
                                                      float alpha, float gamma,
                                                      int w, int h, int l){
    uint xid      = threadIdx.x + blockIdx.x * blockDim.x;
    uint yid      = threadIdx.y + blockIdx.y * blockDim.y;
    uint  id       = xid + yid * w;
    uint  planeSize= w * h;
    
    if (xid < w && yid < h){
        float zo, zc, zn;
        zo= fetch(id + (l-1) * planeSize, (float*)NULL);
        zc= fetch(id, (float*)NULL);
        zn= fetch(id + planeSize, (float*)NULL);

        // zid = 1 to n-2
        for (int zid=0; zid<l-1; ++zid){
            float b1 = zo + zn;
            b1+=(xid   >=1)? fetch(id-1, (float*)NULL) : fetch(id + w-1, (float*)NULL);
            b1+=(xid+1 < w)? fetch(id+1, (float*)NULL) : fetch(id + 1-w, (float*)NULL);
            b1+=(yid   >=1)? fetch(id-w, (float*)NULL) : fetch(id + h*w - w, (float*)NULL);
            b1+=(yid+1 < h)? fetch(id+w, (float*)NULL) : fetch(id + w - h*w, (float*)NULL);

            r[id] = b[id] - zc * (6 * alpha + gamma) + alpha  * b1;

            id += planeSize;
            zo  = zc;
            zc  = zn;
            zn  = (zid < l-2) ? fetch(id + planeSize, (float*)NULL): fetch(xid + yid * w, (float*)NULL);
        }
        
        float b1 = zo + zn;
        b1+=(xid   >=1)? fetch(id-1, (float*)NULL) : fetch(id + w-1, (float*)NULL);
        b1+=(xid+1 < w)? fetch(id+1, (float*)NULL) : fetch(id + 1-w, (float*)NULL);
        b1+=(yid   >=1)? fetch(id-w, (float*)NULL) : fetch(id + h*w - w, (float*)NULL);
        b1+=(yid+1 < h)? fetch(id+w, (float*)NULL) : fetch(id + w - h*w, (float*)NULL);
        r[id] = b[id] - zc * (6 * alpha + gamma) + alpha  * b1;
    }
}

void computeResidual(float* r, float* b, helmholtzMatrix3D_cyclic& A, float* x, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(A.size.x, threads.x),iDivUp(A.size.y, threads.y));
    cache_bind(x);
    //cudaBindTexture(0, com_tex_float, x, sizeof(float) * A.getNumElements());
    helmhotlz3Dresidual_cyclic_tex_kernel<<<grids, threads, 0, stream>>>(r, b, x,
                                                                         A.alpha, A.gamma,
                                                                         A.size.x, A.size.y, A.size.z);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

// Three banded matrix mul vector
void tridiagonalMatrix::init(){
    dmemAlloc(d_a, n);
    dmemAlloc(d_a_m, n);
    dmemAlloc(d_a_p, n);
}

void tridiagonalMatrix::clean(){
    dmemFree(d_a);
    dmemFree(d_a_m);
    dmemFree(d_a_p);
}

__global__ void matrixMulVector_kernel(float* b,
                                       float* a, float *a_m, float *a_p,
                                       int moff, int poff,
                                       float* x, int n){
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    int id       = blockId * blockDim.x + tid;

    if (id <n) {
        float r = a[id] * x[id];
        r +=(id - moff >=0)? a_m[id] * x[id - moff] : 0;
        r +=(id + poff < n)? a_p[id] * x[id + poff] : 0;
        b[id] = r;
    }
}

void matrixMulVector(float* b, tridiagonalMatrix& A, float* x, cudaStream_t stream){
    dim3 threads(256);
    int nBlocks = iDivUp(A.getSize(), threads.x);
    dim3 grids(nBlocks);
    checkConfig(grids);
    matrixMulVector_kernel<<<grids, threads, 0, stream>>>(b, A.d_a, A.d_a_m, A.d_a_p,
                                                          A.moff, A.poff, x, A.getSize());
}

__global__ void computeResidual_kernel(float* r, float* b,
                                       float* a, float *a_m, float *a_p,
                                       int moff, int poff,
                                       float* x, int n){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int tid     = threadIdx.x;
    int id      = blockId * blockDim.x + tid;

    float b1 = a[id] * x[id];
    b1 +=(id - moff >=0)? a_m[id] * x[id - moff] : 0;
    b1 +=(id + poff < n)? a_p[id] * x[id + poff] : 0;
    
    r[id] = b[id] - b1;
}

void computeResidual(float* r, float* b, tridiagonalMatrix& A, float* x, cudaStream_t stream){
    dim3 threads(256);
    int nBlocks = iDivUp(A.n, threads.x);
    dim3 grids(nBlocks);
    checkConfig(grids);
    computeResidual_kernel<<<grids, threads, 0, stream>>>(r, b, A.d_a, A.d_a_m, A.d_a_p,
                                                          A.moff, A.poff, x, A.n);
}

////////////////////////////////////////////////////////////////////////////////
// CG solver
////////////////////////////////////////////////////////////////////////////////
template<class T>
void CG_impl(float* d_b, T& d_A, float* d_x, int imax, cplReduce* rd,
             float* d_r, float* d_d, float* d_q, cudaStream_t stream)
{
    int n = d_A.getNumElements();
    // r = b - Ax
    computeResidual(d_r, d_b, d_A, d_x, stream);
    // d = r
    copyArrayDeviceToDeviceAsync(d_d, d_r, n, stream); 
    // deltanew = r^Tr
    //float delta = getDeviceSumDouble(d_r, n);
    float delta_new = rd->Sum2(d_r, n);
    float delta0    = delta_new;
    float delta_old;
    float eps = 1e-5;
    int i =0;

    while ((i < imax) && (delta_new >= eps * delta0)){
        // q = Ad
        matrixMulVector(d_q, d_A, d_d, stream);
        
        // alpha = delta_new / d^Tq
        float alpha = delta_new / rd->Dot(d_d, d_q, n);

        // x = x + alpha * d
        cplVectorOpers::Add_MulC_I(d_x, d_d, alpha, n, stream);
        
        // r = r - alpha * q
        cplVectorOpers::Add_MulC_I(d_r, d_q, -alpha, n, stream);
        
        delta_old = delta_new;

        // delta_new = r^Tr
        delta_new = rd->Sum2(d_r, n);
        
        // beta = delta_new / delta_old
        float beta = delta_new / delta_old;
        
        // d = beta * d + r
        cplVectorOpers::MulCAdd_I(d_d, beta, d_r, n, stream);
        ++i;
    }
}

template void CG_impl(float* d_b, helmholtzMatrix3D& d_A, float* d_x, int imax,
                      cplReduce* rd,
                      float* d_r, float* d_d, float* d_q, cudaStream_t stream);

template void CG_impl(float* d_b, helmholtzMatrix3D_cyclic& d_A, float* d_x, int imax,
                      cplReduce* rd,
                      float* d_r, float* d_d, float* d_q, cudaStream_t stream);

template<class T>
void CG(float* d_b, T& d_A, float* d_x, int imax,
        cplReduce* rd,
        float* d_r, float* d_d, float* d_q, cudaStream_t stream)
{
    int n = d_A.getNumElements();

    bool has_rd  = (rd  != NULL);
    bool has_d_r = (d_r != NULL);
    bool has_d_d = (d_d != NULL);
    bool has_d_q = (d_q != NULL);

    if (!has_rd)  rd = new cplReduce();
    if (!has_d_r) dmemAlloc(d_r, n);
    if (!has_d_d) dmemAlloc(d_d, n);
    if (!has_d_q) dmemAlloc(d_q, n);

    CG_impl(d_b, d_A, d_x, imax, rd, d_r, d_d, d_q, stream);

    if (!has_rd)  delete rd;
    if (!has_d_r) dmemFree(d_r);
    if (!has_d_d) dmemFree(d_d);
    if (!has_d_q) dmemFree(d_q);
}

template void CG(float* d_b, helmholtzMatrix3D& d_A, float* d_x, int imax,
                 cplReduce* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream);
template void CG(float* d_b, helmholtzMatrix3D_cyclic& d_A, float* d_x, int imax,
                 cplReduce* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template<class T>
void CG(cplVector3DArray& d_b, T& d_A, cplVector3DArray& d_x, int imax,
        cplReduce* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream)
{
    int n = d_A.getNumElements();
    
    bool has_rd  = (rd  != NULL);
    bool has_d_r = (d_r != NULL);
    bool has_d_d = (d_d != NULL);
    bool has_d_q = (d_q != NULL);
    
    if (!has_rd)  rd = new cplReduce();
    if (!has_d_r) dmemAlloc(d_r, n);
    if (!has_d_d) dmemAlloc(d_d, n);
    if (!has_d_q) dmemAlloc(d_q, n);

    CG_impl(d_b.x, d_A, d_x.x, imax, rd, d_r, d_d, d_q, stream);
    CG_impl(d_b.y, d_A, d_x.y, imax, rd, d_r, d_d, d_q, stream);
    CG_impl(d_b.z, d_A, d_x.z, imax, rd, d_r, d_d, d_q, stream);
    
    if (!has_rd)  delete rd;
    if (!has_d_r) dmemFree(d_r);
    if (!has_d_d) dmemFree(d_d);
    if (!has_d_q) dmemFree(d_q);
}

template void CG(cplVector3DArray& d_b, helmholtzMatrix3D& d_A, cplVector3DArray& d_x, int imax,
                 cplReduce* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream);

template void CG(cplVector3DArray& d_b, helmholtzMatrix3D_cyclic& d_A, cplVector3DArray& d_x, int imax,
                 cplReduce* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
// CG solver stream version
////////////////////////////////////////////////////////////////////////////////
template<int p>
__global__ void cplAdd_MulCDivC_I_kernel_s(float* d_data, float* d_a, uint n)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n)
    {
        d_data[id] += d_a[id] * fetch(p, (float*)NULL) /  fetch(2, (float*)NULL);
    }
}

template<int p>
__global__ void cplSub_MulCDivC_I_kernel_s(float* d_data, float* d_a, uint n)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n)
    {
        d_data[id] -= d_a[id] * fetch(p, (float*)NULL) /  fetch(2, (float*)NULL);
    }
}

void cplAdd_MulCDivC_I_s(float* d_data, float* d_a, float* d_c, unsigned int n, int flip, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids = make_large_grid(iDivUp(n, 256));

    cache_bind(d_c);
    if (flip == 0)
        cplAdd_MulCDivC_I_kernel_s<0><<<grids, threads, 0, stream>>>(d_data, d_a, n);
    else if (flip == 1)
        cplAdd_MulCDivC_I_kernel_s<1><<<grids, threads, 0, stream>>>(d_data, d_a, n);
}

void cplSub_MulCDivC_I_s(float* d_data, float* d_a, float* d_c, unsigned int n, int flip, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids = make_large_grid(iDivUp(n, 256));

    cache_bind(d_c);
    if (flip == 0)
        cplSub_MulCDivC_I_kernel_s<0><<<grids, threads, 0, stream>>>(d_data, d_a, n);
    else if (flip == 1)
        cplSub_MulCDivC_I_kernel_s<1><<<grids, threads, 0, stream>>>(d_data, d_a, n);
}


template<int p>
__global__ void cplMulCAdd_I_kernel_s(float* d_data, float* d_a, uint n)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n)
    {
        if (p == 0)
            d_data[id] = d_data[id] * fetch(0, (float*)NULL) /  fetch(1, (float*)NULL) + d_a[id];
        else if (p == 1)
            d_data[id] = d_data[id] * fetch(1, (float*)NULL) /  fetch(0, (float*)NULL) + d_a[id];
    }
}

void cplMulCAdd_I_s(float* d_data, float* d_c, float* d_a,  unsigned int n, int flip, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids = make_large_grid(iDivUp(n, 256));
    cache_bind(d_c);
    if (flip ==0)
        cplMulCAdd_I_kernel_s<0><<<grids, threads, 0, stream>>>(d_data, d_a, n);
    else if (flip ==1)
        cplMulCAdd_I_kernel_s<1><<<grids, threads, 0, stream>>>(d_data, d_a, n);
}

template<class T>
void CG_stream(float* d_b, T& d_A, float* d_x, int imax, cplReduceS* rd,
               float* d_r, float* d_d, float* d_q, float* d_cTemp3,                     
               cudaStream_t stream)
{
    assert((rd!= NULL)&&(d_r!=NULL)&&(d_d!=NULL)&&(d_q!=NULL)&(d_cTemp3!=NULL));
    int n = d_A.getNumElements();
    // r = b - Ax
    computeResidual(d_r, d_b, d_A, d_x, stream);
    
    // d = r
    copyArrayDeviceToDeviceAsync(d_d, d_r, n, stream);

    //d_cTemp3 = [(delta_new, delta_old), dot]
    rd->Sum2(d_cTemp3, d_r, n, stream);

    int flip = 0;
    for (int i=0; i< imax; ++i){
        // q = Ad
        matrixMulVector(d_q, d_A, d_d, stream);
        
        // alpha = delta_new / d^Tq
        rd->Dot(d_cTemp3 + 2, d_d, d_q, n, stream);

        //float alpha = delta_new / rd->Dot(d_d, d_q, n);
        // x = x + alpha * d
        cplAdd_MulCDivC_I_s(d_x, d_d, d_cTemp3, n, flip, stream);
        
        // r = r - alpha * q
        cplSub_MulCDivC_I_s(d_r, d_q, d_cTemp3, n, flip, stream);
        
        flip = 1 - flip; //delta_old = delta_new;
        // delta_new = r^Tr
        rd->Sum2(d_cTemp3 + flip, d_r, n, stream);
        
        // beta = delta_new / delta_old
        // d = beta * d + r
        cplMulCAdd_I_s(d_d, d_cTemp3, d_r, n, flip, stream);
    }
}

template void CG_stream(float* d_b, helmholtzMatrix3D_cyclic& d_A, float* d_x, int imax, cplReduceS* rd,
                        float* d_r, float* d_d, float* d_q, float* d_cTemp3, cudaStream_t stream);

template void CG_stream(float* d_b, helmholtzMatrix3D& d_A, float* d_x, int imax, cplReduceS* rd,
                        float* d_r, float* d_d, float* d_q, float* d_cTemp3, cudaStream_t stream);

template<class T>
void CG_stream(cplVector3DArray& d_b, T& d_A, cplVector3DArray& d_x, int imax,
               cplReduceS* rd, float* d_r, float* d_d, float* d_q, float* d_cTemp3, cudaStream_t stream)
{
    CG_stream(d_b.x, d_A, d_x.x, imax, rd, d_r, d_d, d_q, d_cTemp3, stream);
    CG_stream(d_b.y, d_A, d_x.y, imax, rd, d_r, d_d, d_q, d_cTemp3, stream);
    CG_stream(d_b.z, d_A, d_x.z, imax, rd, d_r, d_d, d_q, d_cTemp3, stream);
}

template void CG_stream(cplVector3DArray& d_b, helmholtzMatrix3D& d_A, cplVector3DArray& d_x, int imax,
                        cplReduceS* rd, float* d_r, float* d_d, float* d_q, float* d_cTemp3, cudaStream_t stream);

template void CG_stream(cplVector3DArray& d_b, helmholtzMatrix3D_cyclic& d_A, cplVector3DArray& d_x, int imax,
                        cplReduceS* rd, float* d_r, float* d_d, float* d_q, float* d_cTemp3, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
// 1D Poisson matrix vector multiply
////////////////////////////////////////////////////////////////////////////////
__global__ void poison1DMulVector_kernel(float* b, float* x, int n){
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    int id       = blockId * blockDim.x + tid;

    if (id <n) {
        float r = x[id] * 2;
        r -=(id - 1 >=0)? -x[id - 1] : 0;
        r -=(id + 1 < n)? -x[id + 1] : 0;
        b[id] = r;
    }
}

void matrixMulVector(float* b, poisonMatrix1D& A, float* x, cudaStream_t stream){
    dim3 threads(256);
    int nBlocks = iDivUp(A.n, threads.x);
    dim3 grids(nBlocks);
    checkConfig(grids);
    poison1DMulVector_kernel<<<grids, threads, 0, stream>>>(b, x, A.n);
}


////////////////////////////////////////////////////////////////////////////////
// 2D Poisson matrix vector multiply
////////////////////////////////////////////////////////////////////////////////
__global__ void poison2DMulVector_kernel(float* b, float* x, int n, int m){
    uint xid     = threadIdx.x + blockIdx.x * blockDim.x;
    uint yid     = threadIdx.y + blockIdx.y * blockDim.y;
    int id       = xid + n * yid;

    if (xid < n && yid <m){
        float r = x[id] * 4;
        r -=(xid   >=1)? x[id - 1] : 0;
        r -=(xid+1 < n)? x[id + 1] : 0;
        r -=(yid   >=1)? x[id - n] : 0;
        r -=(yid+1 < m)? x[id + n] : 0;

        b[id] = r;
    }
}

void matrixMulVector(float* b, poisonMatrix2D& A, float* x, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(A.n, threads.x),iDivUp(A.m, threads.y));
    poison2DMulVector_kernel<<<grids, threads, 0, stream>>>(b, x, A.n, A.m);
}

////////////////////////////////////////////////////////////////////////////////
// 3D Poisson matrix vector multiply
////////////////////////////////////////////////////////////////////////////////
__global__ void poison3DMulVector_kernel(float* b, float* x, int w, int h, int l){
    uint xid      = threadIdx.x + blockIdx.x * blockDim.x;
    uint yid      = threadIdx.y + blockIdx.y * blockDim.y;

    int  id       = xid + yid * w;
    int  planeSize= w * h;
    
    if (xid < w && yid < h){
        float zo, zc, zn;
        zc= x[id];
        zn= x[id+planeSize];

        // zid == 0
        float r = zc * 6;
        r -=(xid   >=1)? x[id - 1] : 0;
        r -=(xid+1 < w)? x[id + 1] : 0;
        r -=(yid   >=1)? x[id - w] : 0;
        r -=(yid+1 < h)? x[id + w] : 0;
        r -=zn;
        b[id] = r;
        id+=planeSize;
        zo = zc;
        zc = zn;
        zn = x[id+planeSize];

        // zid = 1 to n-2
        for (int zid=1; zid<l-1; ++zid){
            r = zc * 6;
            r -=(xid   >=1)? x[id - 1] : 0;
            r -=(xid+1 < w)? x[id + 1] : 0;
            r -=(yid   >=1)? x[id - w] : 0;
            r -=(yid+1 < h)? x[id + w] : 0;
            r -=zo;
            r -=zn;
            b[id] = r;

            id += planeSize;
            zo  = zc;
            zc  = zn;
            zn  = x[id + planeSize];
        }

        // zid = n-1
        r  = zc * 6;
        r -=(xid   >=1)? x[id - 1] : 0;
        r -=(xid+1 < w)? x[id + 1] : 0;
        r -=(yid   >=1)? x[id - w] : 0;
        r -=(yid+1 < h)? x[id + w] : 0;
        r -=zo;
        b[id] = r;
    }
}

__global__ void poison3DMulVector_tex_kernel(float* b, float* x, int w, int h, int l){
    uint xid      = threadIdx.x + blockIdx.x * blockDim.x;
    uint yid      = threadIdx.y + blockIdx.y * blockDim.y;

    int  id       = xid + yid * w;
    int  planeSize= w * h;
    
    if (xid < w && yid < h){
        float zo, zc, zn;
        zc= fetch(id, (float*)NULL);
        zn= fetch(id + planeSize, (float*)NULL);

        // zid == 0
        float r = zc * 6 - zn;
        r-=(xid   >=1)? fetch(id-1, (float*)NULL) : 0;
        r-=(xid+1 < w)? fetch(id+1, (float*)NULL) : 0;
        r-=(yid   >=1)? fetch(id-w, (float*)NULL) : 0;
        r-=(yid+1 < h)? fetch(id+w, (float*)NULL) : 0;
        b[id] = r;
        id+=planeSize;
        zo = zc;
        zc = zn;
        zn= fetch(id + planeSize, (float*)NULL);

        // zid = 1 to n-2
        for (int zid=1; zid<l-1; ++zid){
            r = zc * 6 - zo - zn;
            r-=(xid   >=1)? fetch(id-1, (float*)NULL) : 0;
            r-=(xid+1 < w)? fetch(id+1, (float*)NULL) : 0;
            r-=(yid   >=1)? fetch(id-w, (float*)NULL) : 0;
            r-=(yid+1 < h)? fetch(id+w, (float*)NULL) : 0;

            b[id] = r;
            id += planeSize;
            zo  = zc;
            zc  = zn;
            zn= fetch(id + planeSize, (float*)NULL);
        }

        // zid = n-1
        r = zc * 6 - zo;
        r-=(xid   >=1)? fetch(id-1, (float*)NULL) : 0;
        r-=(xid+1 < w)? fetch(id+1, (float*)NULL) : 0;
        r-=(yid   >=1)? fetch(id-w, (float*)NULL) : 0;
        r-=(yid+1 < h)? fetch(id+w, (float*)NULL) : 0;
        b[id] = r;
    }
}

void matrixMulVector(float* b, poisonMatrix3D& A, float* x, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(A.size.x, threads.x),iDivUp(A.size.y, threads.y));
    //printDeviceArray1D(x, 10, "Test test test");
#if 1
    cache_bind(x);
    //cudaBindTexture(0, com_tex_float, x, sizeof(float) * A.getNumElements());
    poison3DMulVector_tex_kernel<<<grids, threads, 0, stream>>>(b, x, A.size.x, A.size.y, A.size.z);
    cutilCheckMsg("Poison3dmulvector_Tex_Kernel");
#else
    poison3DMulVector_kernel<<<grids, threads, 0, stream>>>(b, x, A.size.x, A.size.y, A.size.z);
#endif
}


__global__ void residualPoission2D_kernel(float* r, float* b, float* x, int n, int m){
    uint xid     = threadIdx.x + blockIdx.x * blockDim.x;
    uint yid     = threadIdx.y + blockIdx.y * blockDim.y;
    int id       = xid + n * yid;

    if (xid < n && yid <m){
        float b1 = x[id] * 4;
        b1-=(xid   >=1)? x[id - 1] : 0;
        b1-=(xid+1 < n)? x[id + 1] : 0;
        b1-=(yid   >=1)? x[id - n] : 0;
        b1-=(yid+1 < m)? x[id + n] : 0;
        r[id] = b[id] - b1;
    }
}

void computeResidual(float* r, float* b, poisonMatrix2D& A, float* x, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(A.n, threads.x),iDivUp(A.m, threads.y));
    residualPoission2D_kernel<<<grids, threads, 0, stream>>>(r, b, x, A.n, A.m);
}

__global__ void residualPoission3D_kernel(float* r, float* b, float* x, int w, int h, int l){
    uint xid      = threadIdx.x + blockIdx.x * blockDim.x;
    uint yid      = threadIdx.y + blockIdx.y * blockDim.y;
    int  id       = xid + yid * w;
    int  planeSize= w * h;
    
    if (xid < w && yid < h){
        float zo, zc, zn;
        zo= 0;
        zc= x[id];
        zn= x[id+planeSize];

        // zid = 0 to n-2
        for (int zid=0; zid<l-1; ++zid){
            float b1= zc * 6 - zo - zn;
            b1-=(xid   >=1)? x[id - 1] : 0;
            b1-=(xid+1 < w)? x[id + 1] : 0;
            b1-=(yid   >=1)? x[id - w] : 0;
            b1-=(yid+1 < h)? x[id + w] : 0;

            r[id] = b[id] - b1;

            id += planeSize;
            zo  = zc;
            zc  = zn;
            zn  = x[id + planeSize];
        }

        // zid = n-1
        float b1 = zc * 6 - zo;
        b1-=(xid   >=1)? x[id - 1] : 0;
        b1-=(xid+1 < w)? x[id + 1] : 0;
        b1-=(yid   >=1)? x[id - w] : 0;
        b1-=(yid+1 < h)? x[id + w] : 0;
        r[id] = b[id] - b1;
    }
}

__global__ void residualPoission3D_tex_kernel(float* r, float* b, float* x, int w, int h, int l){
    uint xid      = threadIdx.x + blockIdx.x * blockDim.x;
    uint yid      = threadIdx.y + blockIdx.y * blockDim.y;
    int  id       = xid + yid * w;
    int  planeSize= w * h;
    
    if (xid < w && yid < h){
        float zo, zc, zn;
        zo= 0;
        zc= fetch(id, (float*)NULL);
        zn= fetch(id + planeSize, (float*)NULL);

        for (int zid=0; zid<l-1; ++zid){
            float b1= zc * 6 - zo - zn;
            b1-=(xid   >=1)? fetch(id-1, (float*)NULL) : 0;
            b1-=(xid+1 < w)? fetch(id+1, (float*)NULL) : 0;
            b1-=(yid   >=1)? fetch(id-w, (float*)NULL) : 0;
            b1-=(yid+1 < h)? fetch(id+w, (float*)NULL) : 0;

            r[id] = b[id] - b1;

            id += planeSize;
            zo  = zc;
            zc  = zn;
            zn  = fetch(id + planeSize, (float*)NULL);
        }

        // zid = n-1
        float b1 = zc * 6 - zo;
        b1-=(xid   >=1)? fetch(id-1, (float*)NULL) : 0;
        b1-=(xid+1 < w)? fetch(id+1, (float*)NULL) : 0;
        b1-=(yid   >=1)? fetch(id-w, (float*)NULL) : 0;
        b1-=(yid+1 < h)? fetch(id+w, (float*)NULL) : 0;
        r[id] = b[id] - b1;
    }
}

void computeResidual(float* r, float* b, poisonMatrix3D& A, float* x, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(A.size.x, threads.x),iDivUp(A.size.y, threads.y));

    cache_bind(x);
    //cudaBindTexture(0, com_tex_float, x, sizeof(float) * A.getNumElements());
    residualPoission3D_tex_kernel<<<grids, threads, 0, stream>>>(r, b, x, A.size.x, A.size.y, A.size.z);
    //residualPoission3D_kernel<<<grids, threads, 0, stream>>>(r, b, x, A.size.x, A.size.y, A.size.z);
}


////////////////////////////////////////////////////////////////////////////////
// Create the full matrix in CPU memory to test 
////////////////////////////////////////////////////////////////////////////////

void createPoison2D(float* a, int n, int m){
    int id = 0;
    int size = n * m;
    memset(a,0, sizeof(float) * size * size);
    for (int i=0; i< size; ++i){
        int xid = i % n;
        int yid = i / n;

        a[id] = 4;
        if (xid - 1 >=0) a[id-1] = -1;
        if (xid + 1 < n) a[id+1] = -1;
        if (yid - 1 >=0) a[id-n] = -1;
        if (yid + 1 < m) a[id+n] = -1;
        id += size + 1;
    }
}

void createPoison3D(float* a, int w, int h, int l){
    int id = 0;
    int size = w * h * l;
    
    memset(a,0, sizeof(float) * size * size);
    for (int i=0; i< size; ++i){
        int xid = i % w;
        int yid = (i / w) % h;
        int zid = i / (w * h);
                
        a[id] = 6;

        if (xid - 1 >=0) a[id -1] = -1;
        if (xid + 1 < w) a[id +1] = -1;

        if (yid - 1 >=0) a[id -w] = -1;
        if (yid + 1 < h) a[id +w] = -1;

        if (zid - 1 >=0) a[id -w*h] = -1;
        if (zid + 1 < l) a[id +w*h] = -1;

        id += size + 1;
    }
}

void createHelmhotlz3D(float* a,
                       int w, int h, int l,
                       float alpha, float gamma){
    int id = 0;
    int size = w * h * l;
    memset(a,0, sizeof(float) * size * size);
    
    for (int i=0; i< size; ++i){
        int xid = i % w;
        int yid = (i / w) % h;
        int zid = i / (w * h);
                
        a[id] = 6 * alpha + gamma;

        if (xid - 1 >=0) a[id -1] = -alpha;
        if (xid + 1 < w) a[id +1] = -alpha;

        if (yid - 1 >=0) a[id -w] = -alpha;
        if (yid + 1 < h) a[id +w] = -alpha;

        if (zid - 1 >=0) a[id -w*h] = -alpha;
        if (zid + 1 < l) a[id +w*h] = -alpha;

        id += size + 1;
    }
}

void matrixMulVector_cpu(float* b, float* a, float* x, int m, int n){
    // for each colume 
    for (int i=0; i<m; ++i){
        float* s = a + i * n;
        float sum = 0;
        for (int j=0; j< n; ++j)
            sum += s[j] * x[j];
        b[i] = sum;
    }
}

void computeResidual_cpu(float* r, float* b, float* a, float* x, int m, int n){
    // for each colume 
    for (int i=0; i<m; ++i){
        float* s = a + i * n;
        float sum = 0;
        for (int j=0; j< n; ++j)
            sum += s[j] * x[j];
        r[i] = b[i] - sum;
    }
}

void runPoissonResidualTest( int argc, char** argv) 
{
    fprintf(stderr, "Run Poission residual ... ");

    const int n = 32;
    int n3 = n * n * n;
   
    // Generate test on the CPU side
    float* h_A = new float [n3 * n3];
    float* h_x = new float [n3];
    float* h_b = new float [n3];
    float* h_r = new float [n3];

    for (int i=0; i< n3; ++i){
        h_x[i] = 1;
        h_b[i] = rand() % n3;
    }

    createPoison3D(h_A, n, n, n);
    computeResidual_cpu(h_r, h_b, h_A, h_x, n3, n3);

    // Test on GPU side
    poisonMatrix3D d_A(n,n,n);
    float* d_x, *d_b, *d_r;
    printCUDAMemoryUsage();
    dmemAlloc(d_x, n3);
    dmemAlloc(d_b, n3);
    dmemAlloc(d_r, n3);
    copyArrayToDevice(d_x, h_x, n3);
    copyArrayToDevice(d_b, h_b, n3);
    computeResidual(d_r, d_b, d_A, d_x, 0);
    testError(h_r, d_r, 1e-5, n3, "Residual compute Poisson 3D");
    
    delete []h_A;
    delete []h_b;
    delete []h_x;
    delete []h_r;

    dmemFree(d_x);
    dmemFree(d_b);
    dmemFree(d_r);
    printCUDAMemoryUsage();
}

void runPoissionSolverTest(int argc, char** argv) 
{
    printCUDAMemoryUsage();
    fprintf(stderr, "Run Poisson Solver ...");
    const int n = 20;
    int n3      = n * n * n;
    
    float* h_x = new float [n3];
    float* h_b = new float [n3];
    float* h_A = new float [n3 * n3];

    // On CPU side
    createTest3D(h_x, n);
    createPoison3D(h_A, n, n, n);
    matrixMulVector_cpu(h_b, h_A, h_x, n3, n3);
    
    // On GPU side
    poisonMatrix3D d_A(n,n,n);

    float* d_x, *d_b, *d_r;
    dmemAlloc(d_x, n3);
    dmemAlloc(d_b, n3);
    dmemAlloc(d_r, n3);

    // Test multiply first
    copyArrayToDevice(d_x, h_x, n3);
    matrixMulVector(d_b, d_A, d_x);
    testError(h_b, d_b, 1e-4, n3, "Possion matrix vector multiply");
    
    cplVectorOpers::SetMem(d_x, 0.f, n3);
    copyArrayToDevice(d_b, h_b, n3);
    CG(d_b, d_A, d_x, 100);
    testError(h_x, d_x, 1e-4, n3,  "CG Possion solver 3D");
    
    delete []h_x;
    delete []h_b;
    delete []h_A;
    
    dmemFree(d_x);
    dmemFree(d_b);
    dmemFree(d_r);
    printCUDAMemoryUsage();
}

void runPoissionSolverPerformanceTest(int argc, char** argv)
{
    int n    = 256;
    int n3   = n * n * n;
    fprintf(stderr, "Problem size %d \n", n);

    float* h_x = new float [n3];
    createTest3D(h_x, n);
        
    float* d_x, *d_b, *d_r;
    float* d_d, *d_q, *d_cTemp3;

    dmemAlloc(d_x, n3);
    dmemAlloc(d_b, n3);
    dmemAlloc(d_r, n3);
    dmemAlloc(d_d, n3);
    dmemAlloc(d_q, n3);
    dmemAlloc(d_cTemp3, 3);
    cutilCheckMsg("Allocate memory");
    
    poisonMatrix3D d_A(n,n,n);
    copyArrayToDevice(d_x, h_x, n3);

// Compute d_b = A d_x;
    //cudaSetMem(d_b, 0.f, n3);
    matrixMulVector(d_b, d_A, d_x);
    cutilCheckMsg("Matrix Mul test");
    
    cplReduce rd;
    cplReduceS rdS;
    rd.init();
    rdS.init();

    unsigned int timer = 0;
    cutCreateTimer(&timer);
        
    // Test speed of non-stream version
    cplVectorOpers::SetMem(d_x, 0.f, n3);
    CG(d_b, d_A, d_x, 50, &rd, d_r, d_d, d_q);
    cutilCheckMsg("CG first run");
    cudaThreadSynchronize();
    
    int nIters = 100;
    cutResetTimer(timer);
    cutStartTimer(timer);
    for (int i=0; i < nIters; ++i){
        cplVectorOpers::SetMem(d_x, 0.f, n3);
        CG(d_b, d_A, d_x, 50, &rd, d_r, d_d, d_q);
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    float reduceTime = cutGetTimerValue(timer) / nIters;
    fprintf(stderr, "Size %d Sum Bandwidth:    %f GB/s\n", n3, (n3 * sizeof(int)) / (reduceTime * 1.0e6));

    // Test speed of stream version
    cplVectorOpers::SetMem(d_x, 0.f, n3);
    CG_stream(d_b, d_A, d_x, 50, &rdS, d_r, d_d, d_q, d_cTemp3);
    cudaThreadSynchronize();

    cutResetTimer(timer);
    cutStartTimer(timer);
    for (int i=0; i < nIters; ++i){
        cplVectorOpers::SetMem(d_x, 0.f, n3);
        CG_stream(d_b, d_A, d_x, 50, &rdS, d_r, d_d, d_q, d_cTemp3);
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    reduceTime = cutGetTimerValue(timer) / nIters;
    fprintf(stderr, "Size %d Sum Bandwidth:    %f GB/s\n", n3, (n3 * sizeof(int)) / (reduceTime * 1.0e6));
    
    dmemFree(d_x);
    dmemFree(d_b);
    dmemFree(d_r);
    dmemFree(d_d);
    dmemFree(d_q);
    dmemFree(d_cTemp3);
}


void runPoissionSolverStreamTest(int argc, char** argv) 
{
    printCUDAMemoryUsage();
    fprintf(stderr, "Run Poisson Solver ...");
    const int n = 20;
    int n3      = n * n * n;
    
    float* h_x = new float [n3];
    float* h_b = new float [n3];
    float* h_A = new float [n3 * n3];

    // On CPU side
    createTest3D(h_x, n);
    createPoison3D(h_A, n, n, n);
    matrixMulVector_cpu(h_b, h_A, h_x, n3, n3);
    
    // On GPU side
    poisonMatrix3D d_A(n,n,n);

    float* d_x, *d_b, *d_r;
    float* d_d, *d_q, *d_cTemp3;
    dmemAlloc(d_x, n3);
    dmemAlloc(d_b, n3);
    dmemAlloc(d_r, n3);
    dmemAlloc(d_d, n3);
    dmemAlloc(d_q, n3);
    dmemAlloc(d_cTemp3, 3);
        
    // Test multiply first
    copyArrayToDevice(d_x, h_x, n3);
    matrixMulVector(d_b, d_A, d_x);
    testError(h_b, d_b, 1e-4, n3, "Possion matrix vector multiply");
    
    cplVectorOpers::SetMem(d_x, 0.f, n3);
    copyArrayToDevice(d_b, h_b, n3);
    cplReduceS rd;
    rd.init();
    
    CG_stream(d_b, d_A, d_x, 50,
              &rd, d_r, d_d, d_q, d_cTemp3);
    testError(h_x, d_x, 1e-5, n3,  "CG Possion solver 3D");
    
    delete []h_x;
    delete []h_b;
    delete []h_A;
    
    dmemFree(d_x);
    dmemFree(d_b);
    dmemFree(d_r);
    dmemFree(d_d);
    dmemFree(d_q);
    dmemFree(d_cTemp3);

    printCUDAMemoryUsage();
}


void runHelmHoltzTest( int argc, char** argv) 
{
    fprintf(stderr, "Run helmholtz ...\n");
    const int n = 20;
    int n3      = n * n * n;
    
    float* h_x = new float [n3];
    float* h_b = new float [n3];
    float* h_A = new float [n3 * n3];

    float alpha = 1;
    float gamma = 0.0;

    // Generate test on the CPU side
    createTest3D(h_x, n);
    createHelmhotlz3D(h_A, n, n, n, alpha, gamma);
    matrixMulVector_cpu(h_b, h_A, h_x, n3, n3);
    
    // Test on the GPU side
    helmholtzMatrix3D d_A(n,n,n, alpha, gamma);
    poisonMatrix3D d_A2(n,n,n);
    float* d_x, *d_b, *d_r;

    dmemAlloc(d_x, n3);
    dmemAlloc(d_b, n3);
    dmemAlloc(d_r, n3);

    copyArrayToDevice(d_x, h_x, n3);
    matrixMulVector(d_b, d_A, d_x);
    testError(h_b, d_b, 1e-5, n3, "Helmholtz matrix vector multiply");
    
    cplVectorOpers::SetMem(d_x, 0.f, n3);
    copyArrayToDevice(d_b, h_b, n3);
    CG(d_b, d_A, d_x, 50);
    testError(h_x, d_x, 1e-5, n3, "CG Helmholtz solver");
    
    
    delete []h_x;
    delete []h_b;
    delete []h_A;
    
    dmemFree(d_x);
    dmemFree(d_b);
    dmemFree(d_r);
}

void CGSolverPlan::setParams(const Vector3Di& size, float alpha, float gamma)
{
    d_A.setParams(size, alpha, gamma);
}

void CGSolverPlan::solve(float* d_x,  float* d_b,
                         float alpha, float gamma,  const Vector3Di& size, int nIters,
                         cplReduce* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream)
{
    d_A.setParams(size, alpha, gamma);
    CG(d_b, d_A, d_x, nIters, rd, d_r, d_d, d_q, stream);
};

void CGSolverPlan::solve(cplVector3DArray& d_x, cplVector3DArray& d_b, float alpha, float gamma,  const Vector3Di& size, int nIters,
                         cplReduce* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream)
{
    d_A.setParams(size, alpha, gamma);
    CG(d_b, d_A, d_x, nIters, rd, d_r, d_d, d_q, stream);
}


cplCGSolverStream::~cplCGSolverStream()
{
    if (d_cTemp3)
        dmemFree(d_cTemp3);
}
void cplCGSolverStream::init(){
    if (d_cTemp3 == NULL)
        dmemAlloc(d_cTemp3, 3);
}
void cplCGSolverStream::setParams(const Vector3Di& size, float alpha, float gamma){
    d_A.setParams(size, alpha, gamma);
}

void cplCGSolverStream::solve(float* d_x,  float* d_b, int nIters, 
                              cplReduceS* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream)
{
    CG_stream(d_b, d_A, d_x, nIters, rd, d_r, d_d, d_q, d_cTemp3, stream);
}

void cplCGSolverStream::solve(cplVector3DArray& d_x, cplVector3DArray& d_b, int nIters,
                               cplReduceS* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream)
{
    CG_stream(d_b, d_A, d_x, nIters, rd, d_r, d_d, d_q, d_cTemp3, stream);
}
