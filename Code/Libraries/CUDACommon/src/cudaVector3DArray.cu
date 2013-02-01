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

#include <cpl.h>
#include <cudaVector3DArray.h>

cplVector3DArray::cplVector3DArray(float* buffer, int aN, mem_type type)
{
    m_memType = type;
    n      = aN;
    nAlign = iAlignUp(n, CUDA_DATA_BLOCK_ALIGN);

    x = buffer;
    y = x + nAlign;
    z = y + nAlign;
}

void allocatePinnedVector3DArray(cplVector3DArray& a, uint aN)
{
    a.n         = aN;
    a.nAlign    = iAlignUp(a.n, CUDA_DATA_BLOCK_ALIGN);
    a.m_memType = cpu_mem;

    allocatePinnedHostArray(a.x, 3 * a.nAlign);
    
    a.y = a.x + a.nAlign;
    a.z = a.x + 2 * a.nAlign;
}

void freePinnedVector3DArray(cplVector3DArray& a)
{
    cudaFreeHost(a.x);
}

/*----------------------------------------------------------------------------------------------------*/
void allocateVector3DArray(cplVector3DArray& a, uint aN, mem_type type)
{
    a.n       = aN;
    a.nAlign  = iAlignUp(a.n, CUDA_DATA_BLOCK_ALIGN);
    a.m_memType = type;

    if (a.m_memType == cpu_mem)        a.x = new float [3 * a.nAlign];
    else if (a.m_memType == gpu_mem)   dmemAlloc(a.x, 3 * a.nAlign);

    a.y = a.x + a.nAlign;
    a.z = a.x + 2 * a.nAlign;
}
void allocateHostVector3DArray(cplVector3DArray& a, uint aN){
    allocateVector3DArray(a, aN, cpu_mem);
}
void allocateDeviceVector3DArray(cplVector3DArray& a, uint aN){
    allocateVector3DArray(a, aN, gpu_mem);
}

/*----------------------------------------------------------------------------------------------------*/
void freeVector3DArray(cplVector3DArray& a)
{
    if (a.x != NULL) {
        if (a.m_memType == cpu_mem) delete []a.x;
        else if (a.m_memType == gpu_mem) dmemFree(a.x);
    }
    a.x = a.y = a.z = NULL;

}

void freeHostVector3DArray(cplVector3DArray& a){
    assert(a.m_memType == cpu_mem);
    if (a.x != NULL)
        delete []a.x;
    a.x = a.y = a.z = NULL;
}

void freeDeviceVector3DArray(cplVector3DArray& a){
    assert(a.m_memType == gpu_mem);
    if (a.x != NULL)
        dmemFree(a.x);
    a.x = a.y = a.z = NULL;
}

/*----------------------------------------------------------------------------------------------------*/
void copyArrayToDevice(cplVector3DArray& d_o, cplVector3DArray& h_i, int nElems)
{
    if ((h_i.size()==d_o.size()) && (h_i.size() == nElems)){
        copyArrayToDevice(d_o.x, h_i.x, 3 * h_i.capacity());
    }
    else {
        copyArrayToDevice(d_o.x, h_i.x, nElems);
        copyArrayToDevice(d_o.y, h_i.y, nElems);
        copyArrayToDevice(d_o.z, h_i.z, nElems);
    }
}
void copyArrayFromDevice(cplVector3DArray& h_o, cplVector3DArray& d_i, int nElems)
{
    if ((d_i.size()==h_o.size()) && (d_i.size() == nElems)){
        copyArrayFromDevice(h_o.x, d_i.x, 3 * d_i.capacity());
    }
    else {
        copyArrayFromDevice(h_o.x, d_i.x, nElems);
        copyArrayFromDevice(h_o.y, d_i.y, nElems);
        copyArrayFromDevice(h_o.z, d_i.z, nElems);
    }
}

void copyArrayToDeviceAsync(cplVector3DArray& d_o, cplVector3DArray& h_i, int nElems, cudaStream_t stream)
{
    if ((h_i.size()==d_o.size()) && (h_i.size() == nElems)){
        copyArrayToDeviceAsync(d_o.x, h_i.x, 3 * h_i.capacity(), stream);
    }
    else {
        copyArrayToDeviceAsync(d_o.x, h_i.x, nElems, stream);
        copyArrayToDeviceAsync(d_o.y, h_i.y, nElems, stream);
        copyArrayToDeviceAsync(d_o.z, h_i.z, nElems, stream);
    }
}

void copyArrayFromDeviceAsync(cplVector3DArray& h_o, cplVector3DArray& d_i, int nElems, cudaStream_t stream)
{
    if ((d_i.size()==h_o.size()) && (d_i.size() == nElems)){
        copyArrayFromDeviceAsync(h_o.x, d_i.x, 3 * d_i.capacity(), stream);
    }
    else {
        copyArrayFromDeviceAsync(h_o.x, d_i.x, nElems, stream);
        copyArrayFromDeviceAsync(h_o.y, d_i.y, nElems, stream);
        copyArrayFromDeviceAsync(h_o.z, d_i.z, nElems, stream);
    }
}

void copyArrayDeviceToDeviceAsync(cplVector3DArray& d_o, const cplVector3DArray& d_i, uint nElems, cudaStream_t stream)
{
    if ((d_i.size()==d_o.size()) && (d_i.size() == nElems)){
        copyArrayDeviceToDeviceAsync(d_o.x, d_i.x, 3 * d_i.capacity(), stream);
    }
    else {
        copyArrayDeviceToDeviceAsync(d_o.x, d_i.x, nElems, stream);
        copyArrayDeviceToDeviceAsync(d_o.y, d_i.y, nElems, stream);
        copyArrayDeviceToDeviceAsync(d_o.z, d_i.z, nElems, stream);
    }
}
/*----------------------------------------------------------------------------------------------------*/
void copyArrayDeviceToDevice(cplVector3DArray& d_o, const cplVector3DArray& d_i){
    if (d_i.size() == d_o.size()){
        copyArrayDeviceToDevice(d_o.x, d_i.x, 3 * d_i.capacity());
    }
    else {
        copyArrayDeviceToDevice(d_o.x, d_i.x, d_i.size());
        copyArrayDeviceToDevice(d_o.y, d_i.y, d_i.size());
        copyArrayDeviceToDevice(d_o.z, d_i.z, d_i.size());
    }
}

void copyArrayDeviceToDevice(cplVector3DArray& d_o, const cplVector3DArray& d_i, uint nElems)
{
    if ((d_i.size()==d_o.size()) && (d_i.size() == nElems)){
        copyArrayDeviceToDevice(d_o.x, d_i.x, 3 * d_i.capacity());
    }
    else {
        copyArrayDeviceToDevice(d_o.x, d_i.x, nElems);
        copyArrayDeviceToDevice(d_o.y, d_i.y, nElems);
        copyArrayDeviceToDevice(d_o.z, d_i.z, nElems);
    }
}


namespace cplVector3DOpers {
    inline bool is1D(const cplVector3DArray& d_i, uint n) {
        return (d_i.isContinous()) && (d_i.n == n);
    }

    inline bool is1D(const cplVector3DArray& d_i, const cplVector3DArray& d_i1, uint n) {
        return is1D(d_i, n) && is1D(d_i1, n);
    }

    inline bool is1D(const cplVector3DArray& d_i, const cplVector3DArray& d_i1, const cplVector3DArray& d_i2, uint n) {
        return is1D(d_i, n) && is1D(d_i1, n) && is1D(d_i2, n);
    }

    inline bool is1D(const cplVector3DArray& d_i, const cplVector3DArray& d_i1, const cplVector3DArray& d_i2, const cplVector3DArray& d_i3, uint n) {
        return is1D(d_i, n) && is1D(d_i1, n) && is1D(d_i2, n) && is1D(d_i3, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Set the memory of cplVector3DArray with a float3 value
//////////////////////////////////////////////////////////////////////////////////
    __global__ void SetMem_kernel(float* d_o_x, float* d_o_y, float* d_o_z,
                                  float x, float y, float z, uint n)
    {
        uint blockId = get_blockID();
        uint id   = get_threadID(blockId);

        if (id < n){
            d_o_x[id] = x;
            d_o_y[id] = y;
            d_o_z[id] = z;
        }
    }

    __global__ void SetMem_kernel(float* d_o_x,  float x, float y, float z,
                                  uint n, uint nAlign)
    {
        uint blockId = get_blockID();
        uint id   = get_threadID(blockId);
  
        if (id < n){
            d_o_x[id      ] = x;
            d_o_x[id + nAlign  ] = y;
            d_o_x[id + 2*nAlign ] = z;
        }
    }

    void SetMem(cplVector3DArray& d_o, const Vector3Df& v, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        SetMem_kernel<<<grids, threads,0,stream>>>(d_o.x, v.x, v.y, v.z, n, d_o.nAlign);
    }

/////////////////////////////////////////////////////////////////////////////////
// Set the memory of cplVector3DArray with a float value
//////////////////////////////////////////////////////////////////////////////////
    void SetMem(cplVector3DArray& d_o, float c, uint n, cudaStream_t stream){
        if (is1D(d_o, n)){
            cplVectorOpers::SetMem(d_o.x, c, 3 * n, stream);
        }
        else {
            cplVectorOpers::SetMem(d_o.x, c, n, stream);
            cplVectorOpers::SetMem(d_o.y, c, n, stream);
            cplVectorOpers::SetMem(d_o.z, c, n, stream);
        }
    }

////////////////////////////////////////////////////////////////////////////////
// Compute the element magnitude of the 3D vector array
////////////////////////////////////////////////////////////////////////////////
    __global__ void Magnitude_kernel(float* d_o, const float* d_ix, const float* d_iy, const float* d_iz, uint n){
        uint blockId = get_blockID();
        uint id   = get_threadID(blockId);
    
        if (id < n){
            d_o[id] = sqrt(d_ix[id] * d_ix[id] + d_iy[id] * d_iy[id] + d_iz[id] * d_iz[id]);
        }
    }
    void Magnitude(float* d_o, const cplVector3DArray& d_i, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        
        Magnitude_kernel<<<grids, threads, 0, stream>>>(d_o, d_i.x, d_i.y, d_i.z, n);
    }

////////////////////////////////////////////////////////////////////////////////
// Compute the element magnitude of the 3D vector array
////////////////////////////////////////////////////////////////////////////////
    __global__ void SqrMagnitude_kernel(float* d_o, const float* d_ix, const float* d_iy, const float* d_iz, uint n){
        uint blockId = get_blockID();
        uint id   = get_threadID(blockId);

        if (id < n){
            d_o[id] = d_ix[id] * d_ix[id] + d_iy[id] * d_iy[id] + d_iz[id] * d_iz[id];
        }
    }

    void SqrMagnitude(float* d_o, const cplVector3DArray& d_i, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        SqrMagnitude_kernel<<<grids, threads, 0, stream>>>(d_o, d_i.x, d_i.y, d_i.z, n);
    }

////////////////////////////////////////////////////////////////////////////////
// Compute the pointwise dot product of the two vector arrays
////////////////////////////////////////////////////////////////////////////////
  __global__ void DotProd_kernel(float* d_o, 
				 const float* d_ax, const float *d_bx,
				 const float* d_ay, const float *d_by,
				 const float* d_az, const float *d_bz,
				 uint n)
  {
    uint blockId = get_blockID();
    uint id   = get_threadID(blockId);
    
    if (id < n){
      d_o[id] = 
	d_ax[id] * d_bx[id] + 
	d_ay[id] * d_by[id] + 
	d_az[id] * d_bz[id];
    }
  }

  void DotProd(float *d_o, 
	       const cplVector3DArray& d_a, 
	       const cplVector3DArray& d_b, 
	       int n, cudaStream_t stream)
  {
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    DotProd_kernel<<<grids, threads, 0, stream>>>(d_o, 
						  d_a.x, d_b.x, 
						  d_a.y, d_b.y, 
						  d_a.z, d_b.z, 
						  n);
  }

////////////////////////////////////////////////////////////////////////////////
// d_o = d_i + c
////////////////////////////////////////////////////////////////////////////////
    __global__ void AddC_kernel(float* d_o_x, const float* d_i_x,
                                float cx, float cy, float cz,
                                uint n, uint n_oAlign, uint n_iAlign){

        uint blockId = get_blockID();
        uint id   = get_threadID(blockId);

        if (id < n){
            d_o_x[id             ] = d_i_x[id             ] + cx;
            d_o_x[id + n_oAlign  ] = d_i_x[id + n_iAlign  ] + cy;
            d_o_x[id + 2*n_oAlign] = d_i_x[id + 2*n_iAlign] + cz;
        }
    }

    void AddC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& c, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        AddC_kernel<<<grids, threads,0,stream>>>(d_o.x, d_i.x, 
                                                 c.x, c.y, c.z, n, d_o.nAlign, d_i.nAlign);
    }


////////////////////////////////////////////////////////////////////////////////
// d_o += c
////////////////////////////////////////////////////////////////////////////////
    __global__ void AddC_I_kernel(float* d_o_x, float* d_o_y, float* d_o_z,
                                  float cx, float cy, float cz, uint n)
    {
        uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
        uint  id = threadIdx.x + blockId * blockDim.x;
        if (id < n){
            d_o_x[id] += cx;
            d_o_y[id] += cy;
            d_o_z[id] += cz;
        }
    }

    void AddC_I(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& c, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        AddC_I_kernel<<<grids, threads, 0, stream>>>(d_o.x, d_o.y, d_o.z,
                                                     c.x  ,   c.y,   c.z, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add two vector3 array
// d_o = d_a + d_b
//////////////////////////////////////////////////////////////////////////////////
    void Add(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, uint n, cudaStream_t stream){
        if (is1D(d_o, d_a, d_b, n)){
            cplVectorOpers::Add(d_o.x, d_a.x, d_b.x, 3 * n, stream);
        }else {
            cplVectorOpers::Add(d_o.x, d_a.x, d_b.x, n, stream);
            cplVectorOpers::Add(d_o.y, d_a.y, d_b.y, n, stream);
            cplVectorOpers::Add(d_o.z, d_a.z, d_b.z, n, stream);
        }
    }

/////////////////////////////////////////////////////////////////////////////////
// Add two vector3 array inplace
// d_o += d_b
//////////////////////////////////////////////////////////////////////////////////
    void Add_I(cplVector3DArray& d_o, const cplVector3DArray& d_b, uint n, cudaStream_t stream){
        if (is1D(d_o, d_b, n)){
            cplVectorOpers::Add_I(d_o.x, d_b.x, 3 * n, stream);
        }else {
            cplVectorOpers::Add_I(d_o.x, d_b.x, n, stream);
            cplVectorOpers::Add_I(d_o.y, d_b.y, n, stream);
            cplVectorOpers::Add_I(d_o.z, d_b.z, n, stream);
        }
    }

/////////////////////////////////////////////////////////////////////////////////
// Subtract two array 3
// d_o = d_a - d_b
//////////////////////////////////////////////////////////////////////////////////
    void Sub(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, uint n, cudaStream_t stream)
    {
        if (is1D(d_o, d_a, d_b, n)){
            cplVectorOpers::Sub(d_o.x, d_a.x, d_b.x, 3 * n, stream);
        }else {
            cplVectorOpers::Sub(d_o.x, d_a.x, d_b.x, n, stream);
            cplVectorOpers::Sub(d_o.y, d_a.y, d_b.y, n, stream);
            cplVectorOpers::Sub(d_o.z, d_a.z, d_b.z, n, stream);
        }
    }

/////////////////////////////////////////////////////////////////////////////////
// Subtract two array 3
// d_o = d_a - d_b
//////////////////////////////////////////////////////////////////////////////////
    void Sub_I(cplVector3DArray& d_o, const cplVector3DArray& d_b, uint n, cudaStream_t stream)
    {
        if (is1D(d_o, d_b, n)){
            cplVectorOpers::Sub_I(d_o.x, d_b.x, 3 * n, stream);
        }else {
            cplVectorOpers::Sub_I(d_o.x, d_b.x, n, stream);
            cplVectorOpers::Sub_I(d_o.y, d_b.y, n, stream);
            cplVectorOpers::Sub_I(d_o.z, d_b.z, n, stream);
        }
    }

    void SubC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& c, uint n,cudaStream_t stream){
        AddC(d_o, d_i, -c, n, stream);
    }

    void SubC_I(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& c, uint n,cudaStream_t stream){
        AddC(d_o, d_i, -c, n, stream);
    }

/////////////////////////////////////////////////////////////////////////////////
// Mul two vector3 array inplace
// d_o = d_a * d_b
//////////////////////////////////////////////////////////////////////////////////
    void Mul(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, uint n, cudaStream_t stream){
        if (is1D(d_o, d_a, d_b, n)){
            cplVectorOpers::Mul(d_o.x, d_a.x, d_b.x, 3 * n, stream);
        }else {
            cplVectorOpers::Mul(d_o.x, d_a.x, d_b.x, n, stream);
            cplVectorOpers::Mul(d_o.y, d_a.y, d_b.y, n, stream);
            cplVectorOpers::Mul(d_o.z, d_a.z, d_b.z, n, stream);
        }
    }

/////////////////////////////////////////////////////////////////////////////////
// Mul two vector3 array inplace
// d_o += d_b
//////////////////////////////////////////////////////////////////////////////////

    void Mul_I(cplVector3DArray& d_o, const cplVector3DArray& d_b, uint n, cudaStream_t stream){
        if (is1D(d_o, d_b, n)){
            cplVectorOpers::Mul_I(d_o.x, d_b.x, 3 * n, stream);
        }else {
            cplVectorOpers::Mul_I(d_o.x, d_b.x, n, stream);
            cplVectorOpers::Mul_I(d_o.y, d_b.y, n, stream);
            cplVectorOpers::Mul_I(d_o.z, d_b.z, n, stream);
        }
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply Vector3 array with a float array inplace version
//////////////////////////////////////////////////////////////////////////////////
    __global__ void Mul_I_kernel(float* d_o_x, float* d_o_y, float* d_o_z, const float* d_b, uint n){
        uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
        uint  id = threadIdx.x + blockId * blockDim.x;
        if (id < n){
            float b = d_b[id];
            d_o_x[id] *= b;
            d_o_y[id] *= b;
            d_o_z[id] *= b;
        }
    }

    void Mul_I(cplVector3DArray& d_o, const float* d_b, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
  
        Mul_I_kernel<<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z, d_b, n);
    }

    __global__ void Add_I_kernel(float* d_o_x, float* d_o_y, float* d_o_z, const float* d_b, uint n){
        uint blockId = get_blockID();
        uint  id = get_threadID(blockId);

        if (id < n){
            float b = d_b[id];
            d_o_x[id] += b;
            d_o_y[id] += b;
            d_o_z[id] += b;
        }
    }

    void Add_I(const cplVector3DArray& d_o, const float* d_b, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
  
        Add_I_kernel<<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z, d_b, n);
    }

    __global__ void Sub_I_kernel(float* d_o_x, float* d_o_y, float* d_o_z, const float* d_b, uint n){
        uint blockId = get_blockID();
        uint  id = get_threadID(blockId);

        if (id < n){
            float b = d_b[id];
            d_o_x[id] -= b;
            d_o_y[id] -= b;
            d_o_z[id] -= b;
        }
    }

    void Sub_I(const cplVector3DArray& d_o, const float* d_b, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
  
        Sub_I_kernel<<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z, d_b, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply Vector3 array with a float array
//////////////////////////////////////////////////////////////////////////////////
    __global__ void Mul_kernel(float* d_o_x, uint noAlign,
                               const float* d_a_x, uint naAlign,
                               const float* d_b, uint n){
        uint blockId = get_blockID();
        uint  id = get_threadID(blockId);

        if (id < n){
            float b = d_b[id];
            d_o_x[id] = d_a_x[id] * b;
            d_o_x[id+noAlign]= d_a_x[id+naAlign] * b;
            d_o_x[id+2*noAlign]= d_a_x[id+2*naAlign] * b;
        }
    }

    void Mul(cplVector3DArray& d_o, const cplVector3DArray& d_a, const float* d_b, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
  
        Mul_kernel<<<grids, threads, 0, stream>>>(d_o.x, d_o.nAlign,
                                                  d_a.x, d_a.nAlign,
                                                  d_b, n);
    }

    __global__ void Add_kernel(float* d_o_x, uint noAlign,
                               const float* d_a_x, uint naAlign,
                               const float* d_b, uint n){
        uint blockId = get_blockID();
        uint  id = get_threadID(blockId);

        if (id < n){
            float b       = d_b[id];
            d_o_x[id]      = d_a_x[id] + b;
            d_o_x[id+noAlign]  = d_a_x[id+naAlign] + b;
            d_o_x[id+2*noAlign] = d_a_x[id+2*naAlign] + b;
        }
    }

    void Add(cplVector3DArray& d_o, const cplVector3DArray& d_a, const float* d_b, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
  
        Add_kernel<<<grids, threads, 0, stream>>>(d_o.x, d_o.nAlign,
                                                  d_a.x, d_a.nAlign,
                                                  d_b, n);
    }


    __global__ void Sub_kernel(float* d_o_x, uint noAlign,
                               const float* d_a_x, uint naAlign,
                               const float* d_b, uint n){
        uint blockId = get_blockID();
        uint  id = get_threadID(blockId);

        if (id < n){
            float b = d_b[id];
            d_o_x[id] = d_a_x[id] - b;
            d_o_x[id+noAlign]= d_a_x[id+naAlign] -b;
            d_o_x[id+2*noAlign]= d_a_x[id+2*naAlign] - b;
        }
    }

    void Sub(cplVector3DArray& d_o, const cplVector3DArray& d_a, const float* d_b, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
  
        Sub_kernel<<<grids, threads, 0, stream>>>(d_o.x, d_o.nAlign,
                                                  d_a.x, d_a.nAlign,
                                                  d_b, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply Vector3 array with a ant
//////////////////////////////////////////////////////////////////////////////////
    void MulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, float c, uint n, cudaStream_t stream){
        if (is1D(d_o, d_i, n)){
            cplVectorOpers::MulC(d_o.x, d_i.x, c, 3 * n, stream);
        }else {
            cplVectorOpers::MulC(d_o.x, d_i.x, c, n, stream);
            cplVectorOpers::MulC(d_o.y, d_i.y, c, n, stream);
            cplVectorOpers::MulC(d_o.z, d_i.z, c, n, stream);
        }
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply Vector3 array with a ant inplace version
//////////////////////////////////////////////////////////////////////////////////

    void MulC_I(cplVector3DArray& d_o, float c, uint n, cudaStream_t stream){
        if (is1D(d_o, n)){
            cplVectorOpers::MulC_I(d_o.x, c, 3 * n, stream);
        }else {
            cplVectorOpers::MulC_I(d_o.x, c, n, stream);
            cplVectorOpers::MulC_I(d_o.y, c, n, stream);
            cplVectorOpers::MulC_I(d_o.z, c, n, stream);
        }
    }

    __global__ void MulC_I_kernel(float* d_o_x, float* d_o_y, float* d_o_z,
                                  float c_x, float c_y, float c_z, uint n){
        uint blockId = get_blockID();
        uint id   = get_threadID(blockId);

        if (id < n){
            d_o_x[id] *= c_x;
            d_o_y[id] *= c_y;
            d_o_z[id] *= c_z;
        }
    }

    void MulC_I(cplVector3DArray& d_o, const Vector3Df& c, uint n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);

        MulC_I_kernel<<<grids, threads, 0, stream>>>(d_o.x, d_o.y, d_o.z,
                                                     c.x, c.y, c.z, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// Complex function
//////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// d_o = d_a + d_b * c
//////////////////////////////////////////////////////////////////////////////////
    void Add_MulC_I(cplVector3DArray& d_o,
                    const cplVector3DArray& d_a, const cplVector3DArray& d_b, float c, uint n, cudaStream_t stream)
    {
        if (is1D(d_o, d_a, d_b,n)){
            cplVectorOpers::Add_MulC(d_o.x, d_a.x, d_b.x, c, 3 * n, stream);
        }else {
            cplVectorOpers::Add_MulC(d_o.x, d_a.x, d_b.x, c, n, stream);
            cplVectorOpers::Add_MulC(d_o.y, d_a.y, d_b.y, c, n, stream);
            cplVectorOpers::Add_MulC(d_o.z, d_a.z, d_b.z, c, n, stream);
        }
    }

/////////////////////////////////////////////////////////////////////////////////
// d_o = d_o + d_b * c
//////////////////////////////////////////////////////////////////////////////////
    void Add_MulC_I(cplVector3DArray& d_o,
                    const cplVector3DArray& d_b, float c, uint n, cudaStream_t stream)
    {
        if (is1D(d_o, d_b, n)){
            cplVectorOpers::Add_MulC_I(d_o.x, d_b.x, c, 3 * n, stream);
        }else {
            cplVectorOpers::Add_MulC_I(d_o.x, d_b.x, c, n, stream);
            cplVectorOpers::Add_MulC_I(d_o.y, d_b.y, c, n, stream);
            cplVectorOpers::Add_MulC_I(d_o.z, d_b.z, c, n, stream);
        }

    }

    void Add_MulC(cplVector3DArray& d_o,
                  const cplVector3DArray& d_a, const cplVector3DArray& d_b, float c, uint n, cudaStream_t stream)
    {
        if (is1D(d_o, d_a, d_b, n)){
            cplVectorOpers::Add_MulC(d_o.x, d_a.x, d_b.x, c, 3 * n, stream);
        }
        else {
            cplVectorOpers::Add_MulC(d_o.x, d_a.x, d_b.x, c, n, stream);
            cplVectorOpers::Add_MulC(d_o.y, d_a.y, d_b.y, c, n, stream);
            cplVectorOpers::Add_MulC(d_o.z, d_a.z, d_b.z, c, n, stream);
        }
    }

/////////////////////////////////////////////////////////////////////////////////
// d_o = d_o + d_a * d_b
//////////////////////////////////////////////////////////////////////////////////
    __global__ void Add_Mul_I_kernel(float* d_o_x, float* d_o_y, float* d_o_z,
                                     const float* d_a_x, const float* d_a_y, const float* d_a_z,
                                     const float* d_b, uint n){
        uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
        uint  id = threadIdx.x + blockId * blockDim.x;
        if (id < n){
            float b = d_b[id];
            d_o_x[id] += d_a_x[id] * b;
            d_o_y[id] += d_a_y[id] * b;
            d_o_z[id] += d_a_z[id] * b;
        }
    }

    __global__ void Add_Mul_I_kernel(float* d_o_x, uint noAlign,
                                     const float* d_a_x, uint naAlign,
                                     const float* d_b, uint n)
    {
        uint blockId = get_blockID();
        uint  id = get_threadID(blockId);
  
        if (id < n){
            float b = d_b[id];
            d_o_x[id]      += d_a_x[id     ] * b;
            d_o_x[id+noAlign]  += d_a_x[id+naAlign ] * b;
            d_o_x[id+2*noAlign] += d_a_x[id+2*naAlign] * b;
        }
    }

    void Add_Mul_I(cplVector3DArray& d_o, const cplVector3DArray& d_a, const float* d_b, uint n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);

        Add_Mul_I_kernel<<<grids, threads, 0, stream>>>(d_o.x, d_o.nAlign,
                                                        d_a.x, d_a.nAlign,
                                                        d_b, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// d_o = d_a + d_b * d_c
//////////////////////////////////////////////////////////////////////////////////
    __global__ void Add_Mul_kernel(float* d_o_x, uint noAlign,
                                   const float* d_a_x, uint naAlign,
                                   const float* d_b_x, uint nbAlign,
                                   const float* d_c, uint n)
    {
        uint blockId = get_blockID();
        uint  id = get_threadID(blockId);
  
        if (id < n){
            float c = d_c[id];
            d_o_x[id]      = d_a_x[id     ] + d_b_x[id     ] * c;
            d_o_x[id+noAlign]  = d_a_x[id+naAlign ] + d_b_x[id+nbAlign ] * c;
            d_o_x[id+2*noAlign] = d_a_x[id+2*naAlign] + d_b_x[id+2*nbAlign] * c;
        }
    }

    void Add_Mul(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, const float* d_c, uint n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);

        Add_Mul_kernel<<<grids, threads, 0, stream>>>(d_o.x, d_o.nAlign,
                                                      d_a.x, d_a.nAlign,
                                                      d_b.x, d_b.nAlign,
                                                      d_c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// d_o = (d_i + a) * b
//////////////////////////////////////////////////////////////////////////////////
    void AddCMulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& a, const Vector3Df& b, int n, cudaStream_t stream)
    {
        cplVectorOpers::AddCMulC(d_o.x, d_i.x, a.x, b.x, n, stream);
        cplVectorOpers::AddCMulC(d_o.y, d_i.y, a.y, b.y, n, stream);
        cplVectorOpers::AddCMulC(d_o.z, d_i.z, a.z, b.z, n, stream);
    }
/////////////////////////////////////////////////////////////////////////////////
// d_o = (d_o + a) * b
//////////////////////////////////////////////////////////////////////////////////
    void AddCMulC_I(cplVector3DArray& d_o, const Vector3Df& a, const Vector3Df& b, int n, cudaStream_t stream)
    {
        cplVectorOpers::AddCMulC_I(d_o.x, a.x, b.x, n, stream);
        cplVectorOpers::AddCMulC_I(d_o.y, a.y, b.y, n, stream);
        cplVectorOpers::AddCMulC_I(d_o.z, a.z, b.z, n, stream);
    }

/////////////////////////////////////////////////////////////////////////////////
// d_o = d_i * a + b
//////////////////////////////////////////////////////////////////////////////////
    void MulCAddC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& a, const Vector3Df& b, int n, cudaStream_t stream)
    {
        cplVectorOpers::MulCAddC(d_o.x, d_i.x, a.x, b.x, n, stream);
        cplVectorOpers::MulCAddC(d_o.y, d_i.y, a.y, b.y, n, stream);
        cplVectorOpers::MulCAddC(d_o.z, d_i.z, a.z, b.z, n, stream);
    }

/////////////////////////////////////////////////////////////////////////////////
// d_o = d_o * a + b
//////////////////////////////////////////////////////////////////////////////////
    void MulCAddC_I(cplVector3DArray& d_o, const Vector3Df& a, const Vector3Df& b, int n, cudaStream_t stream)
    {
        cplVectorOpers::MulCAddC_I(d_o.x, a.x, b.x, n, stream);
        cplVectorOpers::MulCAddC_I(d_o.y, a.y, b.y, n, stream);
        cplVectorOpers::MulCAddC_I(d_o.z, a.z, b.z, n, stream);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply Vector3 array with ant and add the second cplVector3DArray
//////////////////////////////////////////////////////////////////////////////////
    void MulCAdd(cplVector3DArray& d_o, const cplVector3DArray& d_a, float delta, const cplVector3DArray& d_b, uint n, cudaStream_t stream){
        if (is1D(d_o, d_a, d_b, n))
            cplVectorOpers::MulCAdd(d_o.x, d_a.x, delta, d_b.x, 3 * n, stream);
        else {
            cplVectorOpers::MulCAdd(d_o.x, d_a.x, delta, d_b.x, n, stream);
            cplVectorOpers::MulCAdd(d_o.y, d_a.y, delta, d_b.y, n, stream);
            cplVectorOpers::MulCAdd(d_o.z, d_a.z, delta, d_b.z, n, stream);
        }
    }

    void MulCAdd_I(cplVector3DArray& d_o, float c, const cplVector3DArray& d_b, uint n, cudaStream_t stream){
        if (is1D(d_o, d_b, n)){
            cplVectorOpers::MulCAdd_I(d_o.x, c, d_b.x, 3 * n, stream);
        }else {
            cplVectorOpers::MulCAdd_I(d_o.x, c, d_b.x, n, stream);
            cplVectorOpers::MulCAdd_I(d_o.y, c, d_b.y, n, stream);
            cplVectorOpers::MulCAdd_I(d_o.z, c, d_b.z, n, stream);
        }
    }

    void MulC_Add_MulC(cplVector3DArray& d_o,
                       const cplVector3DArray& d_a, float ca,
                       const cplVector3DArray& d_b, float cb,
                       uint n, cudaStream_t stream){
        if (is1D(d_o, d_a, d_b, n)){
            cplVectorOpers::MulC_Add_MulC(d_o.x, d_a.x, ca, d_b.x, cb, 3 * n, stream);
        }else {
            cplVectorOpers::MulC_Add_MulC(d_o.x, d_a.x, ca, d_b.x, cb, n, stream);
            cplVectorOpers::MulC_Add_MulC(d_o.y, d_a.y, ca, d_b.y, cb, n, stream);
            cplVectorOpers::MulC_Add_MulC(d_o.z, d_a.z, ca, d_b.z, cb, n, stream);
        }
    }

    void MulC_Add_MulC_I(cplVector3DArray& d_o, float co,
                         const cplVector3DArray& d_b, float cb,
                         uint n, cudaStream_t stream){
        if (is1D(d_o, d_b, n)){
            cplVectorOpers::MulC_Add_MulC_I(d_o.x, co, d_b.x, cb, 3 * n, stream);
        }else {
            cplVectorOpers::MulC_Add_MulC_I(d_o.x, co, d_b.x, cb, n, stream);
            cplVectorOpers::MulC_Add_MulC_I(d_o.y, co, d_b.y, cb, n, stream);
            cplVectorOpers::MulC_Add_MulC_I(d_o.z, co, d_b.z, cb, n, stream);
        }
    }

  
    void AddCMulCAddC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& a, const Vector3Df& b, const Vector3Df& c, int n, cudaStream_t stream)
    {
        cplVectorOpers::AddCMulCAddC(d_o.x, d_i.x, a.x, b.x, c.x, n, stream);
        cplVectorOpers::AddCMulCAddC(d_o.y, d_i.y, a.y, b.y, c.y, n, stream);
        cplVectorOpers::AddCMulCAddC(d_o.z, d_i.z, a.z, b.z, c.z, n, stream);
    }

    void AddCMulCAddC_I(cplVector3DArray& d_o, const Vector3Df& a, const Vector3Df& b, const Vector3Df& c, int n, cudaStream_t stream)
    {
        cplVectorOpers::AddCMulCAddC_I(d_o.x, a.x, b.x, c.x, n, stream);
        cplVectorOpers::AddCMulCAddC_I(d_o.y, a.y, b.y, c.y, n, stream);
        cplVectorOpers::AddCMulCAddC_I(d_o.z, a.z, b.z, c.z, n, stream);
    }

    void AddCMulCAddC(cplVector3DArray& d_o, const cplVector3DArray& d_i, float a, float b, float c, int n, cudaStream_t stream)
    {
        if (is1D(d_o, d_i, n)){
            cplVectorOpers::AddCMulCAddC(d_o.x, d_i.x, a, b, c, 3*n, stream);
        }
        else {
            cplVectorOpers::AddCMulCAddC(d_o.x, d_i.x, a, b, c, n, stream);
            cplVectorOpers::AddCMulCAddC(d_o.y, d_i.y, a, b, c, n, stream);
            cplVectorOpers::AddCMulCAddC(d_o.z, d_i.z, a, b, c, n, stream);
        }
    }

    void AddCMulCAddC_I(cplVector3DArray& d_o, float a, float b, float c, int n, cudaStream_t stream)
    {
        if (is1D(d_o, n)){
            cplVectorOpers::AddCMulCAddC_I(d_o.x, a, b, c, n * 3, stream);
        } else {
            cplVectorOpers::AddCMulCAddC_I(d_o.x, a, b, c, n, stream);
            cplVectorOpers::AddCMulCAddC_I(d_o.y, a, b, c, n, stream);
            cplVectorOpers::AddCMulCAddC_I(d_o.z, a, b, c, n, stream);
        }
    }


    void AddMulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const cplVector3DArray& d_a, float c, uint n, cudaStream_t stream)
    {
        if (is1D(d_o, d_i, n)){
            cplVectorOpers::AddMulC(d_o.x, d_i.x, d_a.x, c, n * 3, stream);
        } else {
            cplVectorOpers::AddMulC(d_o.x, d_i.x, d_a.x, c, n, stream);
            cplVectorOpers::AddMulC(d_o.y, d_i.y, d_a.y, c, n, stream);
            cplVectorOpers::AddMulC(d_o.z, d_i.z, d_a.z, c, n, stream);
    
        }
    }

    void AddMulC_I(cplVector3DArray& d_o, const cplVector3DArray& d_a, float c, uint n, cudaStream_t stream)
    {
        if (is1D(d_o, d_a, n)){
            cplVectorOpers::AddMulC_I(d_o.x, d_a.x, c, n * 3, stream);
        } else {
            cplVectorOpers::AddMulC_I(d_o.x, d_a.x, c, n, stream);
            cplVectorOpers::AddMulC_I(d_o.y, d_a.y, c, n, stream);
            cplVectorOpers::AddMulC_I(d_o.z, d_a.z, c, n, stream);
        }
    }

    void MulMulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const cplVector3DArray& d_a, float c, uint n, cudaStream_t stream)
    {
        if (is1D(d_o, d_i, d_a, n))
            cplVectorOpers::MulMulC(d_o.x, d_i.x, d_a.x, c, n * 3, stream);
        else {
            cplVectorOpers::MulMulC(d_o.x, d_i.x, d_a.x, c,  n, stream);
            cplVectorOpers::MulMulC(d_o.y, d_i.y, d_a.y, c, n, stream);
            cplVectorOpers::MulMulC(d_o.z, d_i.z, d_a.z, c, n, stream);
        }
    }

    void MulMulC_I(cplVector3DArray& d_o, const cplVector3DArray& d_a, float c, uint n, cudaStream_t stream)
    {
        if (is1D(d_o, d_a, n))
            cplVectorOpers::MulMulC_I(d_o.x, d_a.x, c, n * 3, stream);
        else {
            cplVectorOpers::MulMulC_I(d_o.x, d_a.x, c, n, stream);
            cplVectorOpers::MulMulC_I(d_o.y, d_a.y, c, n, stream);
            cplVectorOpers::MulMulC_I(d_o.z, d_a.z, c, n, stream);
        }
    }


    __global__ void MulMulC_kernel(float* d_o, const float* d_i, const float* d_a, float c, uint n, uint nAlign)
    {
        uint blockId = get_blockID();
        uint id   = get_threadID(blockId);

        if (id < n){
            float a = d_a[id] * c;
            d_o[id]              = d_i[id] * a;
            d_o[id + nAlign]     = d_i[id + nAlign] * a;
            d_o[id + 2 * nAlign] = d_i[id + 2 * nAlign] * a;
        }
    }


    void MulMulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const float* d_a, float c, uint n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);

        MulMulC_kernel<<<grids, threads, 0, stream>>>(d_o.x, d_i.x, d_a, c, n, d_o.nAlign);
    }

    __global__ void MulMulC_I_kernel(float* d_o, const float* d_a, float c,
                                     uint n, uint nAlign)
    {
        uint blockId = get_blockID();
        uint id   = get_threadID(blockId);

        if (id < n){
            float a = d_a[id] * c;
            d_o[id]              *= a;
            d_o[id + nAlign]     *= a;
            d_o[id + 2 * nAlign] *= a;
        }
    }

    void MulMulC_I(cplVector3DArray& d_o, const float* d_a, float c, uint n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);

        MulMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o.x, d_a, c, n, d_o.nAlign);
    }


////////////////////////////////////////////////////////////////////////////////
// d_o = (d_i + a) * b
////////////////////////////////////////////////////////////////////////////////
    void AddCMulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& a, const Vector3Df& b, uint n, cudaStream_t stream)
    {
        cplVectorOpers::AddCMulC(d_o.x, d_i.x, a.x, b.x, n, stream);
        cplVectorOpers::AddCMulC(d_o.y, d_i.y, a.y, b.y, n, stream);
        cplVectorOpers::AddCMulC(d_o.z, d_i.z, a.z, b.z, n, stream);
    }

////////////////////////////////////////////////////////////////////////////////
// d_o = (d_o + a) * b
////////////////////////////////////////////////////////////////////////////////
    void AddCMulC_I(cplVector3DArray& d_o, const Vector3Df& a, const Vector3Df& b, uint n, cudaStream_t stream)
    {
        cplVectorOpers::AddCMulC_I(d_o.x, a.x, b.x, n, stream);
        cplVectorOpers::AddCMulC_I(d_o.y, a.y, b.y, n, stream);
        cplVectorOpers::AddCMulC_I(d_o.z, a.z, b.z, n, stream);
    }

////////////////////////////////////////////////////////////////////////////////
// d_o = d_i * a + b
////////////////////////////////////////////////////////////////////////////////
    void MulCAddC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& a, const Vector3Df& b, uint n, cudaStream_t stream)
    {
        cplVectorOpers::MulCAddC(d_o.x, d_i.x, a.x, b.x, n, stream);
        cplVectorOpers::MulCAddC(d_o.y, d_i.y, a.y, b.y, n, stream);
        cplVectorOpers::MulCAddC(d_o.z, d_i.z, a.z, b.z, n, stream);
    }

////////////////////////////////////////////////////////////////////////////////
// d_o = d_o * a + b
////////////////////////////////////////////////////////////////////////////////
    void MulCAddC_I(cplVector3DArray& d_o, const Vector3Df& a, const Vector3Df& b, uint n, cudaStream_t stream)
    {
        cplVectorOpers::MulCAddC_I(d_o.x, a.x, b.x, n, stream);
        cplVectorOpers::MulCAddC_I(d_o.y, a.y, b.y, n, stream);
        cplVectorOpers::MulCAddC_I(d_o.z, a.z, b.z, n, stream);
    }


    void Add_MulMulC(cplVector3DArray& d_o,
                     const cplVector3DArray& d_a,
                     const cplVector3DArray& d_b,
                     const float* d_c,
                     float d, int n, cudaStream_t stream)
    {
        cplVectorOpers::Add_MulMulC(d_o.x, d_a.x, d_b.x, d_c, d, n, stream);
        cplVectorOpers::Add_MulMulC(d_o.y, d_a.y, d_b.y, d_c, d, n, stream);
        cplVectorOpers::Add_MulMulC(d_o.z, d_a.z, d_b.z, d_c, d, n, stream);
    }

    void Add_MulMulC_I(cplVector3DArray& d_a,
                       const cplVector3DArray& d_b,
                       const float* d_c,
                       float d, int n, cudaStream_t stream)
    {
        cplVectorOpers::Add_MulMulC_I(d_a.x, d_b.x, d_c, d, n, stream);
        cplVectorOpers::Add_MulMulC_I(d_a.y, d_b.y, d_c, d, n, stream);
        cplVectorOpers::Add_MulMulC_I(d_a.z, d_b.z, d_c, d, n, stream);
    }


    void Add_MulMulC(cplVector3DArray& d_o,
                     const cplVector3DArray& d_a,
                     const cplVector3DArray& d_b,
                     const cplVector3DArray& d_c,
                     float d, int n, cudaStream_t stream)
    {
        if (is1D(d_o, d_a, d_b, d_c, n))
            cplVectorOpers::Add_MulMulC(d_o.x, d_a.x, d_b.x, d_c.x, d, 3 * n, stream);
        else {
            cplVectorOpers::Add_MulMulC(d_o.x, d_a.x, d_b.x, d_c.x, d, n, stream);
            cplVectorOpers::Add_MulMulC(d_o.y, d_a.y, d_b.y, d_c.y, d, n, stream);
            cplVectorOpers::Add_MulMulC(d_o.z, d_a.z, d_b.z, d_c.z, d, n, stream);
        }
    }

    void Add_MulMulC_I(const cplVector3DArray& d_o,
                       const cplVector3DArray& d_b,
                       const cplVector3DArray& d_c,
                       float d, int n, cudaStream_t stream)
    {
        if (is1D(d_o, d_b, d_c, n)){
            cplVectorOpers::Add_MulMulC_I(d_o.x, d_b.x, d_c.x, d, 3* n, stream);
        }
        else {
            cplVectorOpers::Add_MulMulC_I(d_o.x, d_b.x, d_c.x, d, n, stream);
            cplVectorOpers::Add_MulMulC_I(d_o.y, d_b.y, d_c.y, d, n, stream);
            cplVectorOpers::Add_MulMulC_I(d_o.z, d_b.z, d_c.z, d, n, stream);
        }
    }

    void FixedToFloating(cplVector3DArray& d_o, uint n, cudaStream_t stream)
    {
        if (is1D(d_o, n)){
            cplVectorOpers::FixedToFloating(d_o.x, (int*)d_o.x, 3 * n, stream);
        } else {
            cplVectorOpers::FixedToFloating(d_o.x, (int*)d_o.x, n, stream);
            cplVectorOpers::FixedToFloating(d_o.y, (int*)d_o.y, n, stream);
            cplVectorOpers::FixedToFloating(d_o.z, (int*)d_o.z, n, stream);
        }
    }

    __global__ void fixedToFloatingNormalized_kernel(float* d_fX, float* d_fY, float* d_fZ,
                                                     int* d_iX, int* d_iY, int* d_iZ,
                                                     int n)
    {
        uint blockId = get_blockID();
        uint id   = get_threadID(blockId);

        if (id < n){
            float3 v;
            v.x = S2n20(d_iX[id]);
            v.y = S2n20(d_iY[id]);
            v.z = S2n20(d_iZ[id]);

            v = normalize(v);

            d_fX[id] = v.x;
            d_fY[id] = v.y;
            d_fZ[id] = v.z;
        }
    }


    void FixedToFloatingNormalize(cplVector3DArray& d_o, uint n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);

        fixedToFloatingNormalized_kernel<<<grids, threads, 0, stream>>>(d_o.x, d_o.y, d_o.z, 
                                                                        (int*)d_o.x, (int*)d_o.y,(int*)d_o.z, n);
    }

    void EpsUpdate(cplVector3DArray& d_data, cplVector3DArray d_var, float eps, int n, cudaStream_t stream){
        if (is1D(d_data, d_var, n)){
            cplVectorOpers::EpsUpdate(d_data.x, d_var.x, eps, 3 * n, stream);
        }else {
            cplVectorOpers::EpsUpdate(d_data.x, d_var.x, eps, n, stream);
            cplVectorOpers::EpsUpdate(d_data.y, d_var.y, eps, n, stream);
            cplVectorOpers::EpsUpdate(d_data.z, d_var.z, eps, n, stream);
        }
    }
        
}
