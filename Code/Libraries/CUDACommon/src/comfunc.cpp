#include <cutil.h>
#include <cutil_comfunc.h>

void cpuAdd_I(float* h_a, float* h_b, int n)
{
    for (int i=0; i< n; ++i)
        h_a[i] += h_b[i];
}

void cpuAdd(float** h_avgL, unsigned int nImgs, unsigned int nVox)
{
    if (nImgs == 1)
        return;
    
    unsigned int half = nextPowerOf2(nImgs) >> 1;
    for (unsigned int i=0; i< half; ++i){
        if (i + half < nImgs)
            cpuAdd_I(h_avgL[i], h_avgL[i+half], nVox);
    }

    half >>= 1;
    while (half > 0){
        for (unsigned int i=0; i< half; ++i)
            cpuAdd_I(h_avgL[i], h_avgL[i+half], nVox);
        half >>=1;
    }
}

/*
bool isMatchH2D(float* h_i, float* d_i1, unsigned int n)
{
    float* h_i1= new  float [n];
    cudaMemcpy(h_i1, d_i1, n * sizeof(float), cudaMemcpyDeviceToHost);
    bool result = isMatchH2H(h_i, h_i1, 1e-5,n);
    delete []h_i1;
    return result;
}

bool isMatchD2D(float* d_i, float* d_i1, unsigned int n)
{
    float* h_i = new  float [n];
    float* h_i1= new  float [n];
    cudaMemcpy(h_i, d_i, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_i1, d_i1, n * sizeof(float), cudaMemcpyDeviceToHost);
    bool result = isMatchH2H(h_i, h_i1, 1e-5,n);
    delete []h_i;
    delete []h_i1;
    return result;
}
*/

void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
    { 
        a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
    }
}

void makeRandomUintVector(uint2 *a, unsigned int numElements, unsigned int keybits)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
    { 
        a[i].x = ((rand() & keyshiftmask)<<16) | (rand() & keymask);
        a[i].y = i;
    }
}

template<class T>
inline T get_value(T* a, int i, int j, int k, int w, int h, int l){
    return a[i + j * w + k * w * h];
}

/**
 * @brief Function to perform trilinear interpolation 
 * @param[in]  vold  The input volume
 *             w, h, l size of input volume 
 * @param[out] Interpolation value
 * 
 */
inline float interpolate(float x, float y, float z,
                         float* vol,
                         const int w, const int h, const int l){

    const int wh = w * h;

    int   xInt, yInt, zInt, xIntP, yIntP, zIntP;
    float dx, dy, dz;
    float dxy, oz;

    xInt = int(x);
    yInt = int(y);
    zInt = int(z);

    xIntP = xInt + 1;
    if (xIntP == w) xIntP = xInt;
    
    yIntP = yInt + 1;
    if (yIntP == h) yIntP = yInt;
        
    zIntP = zInt + 1;
    if (zIntP == l) zIntP = zInt;
    
    dx = x - xInt;
    dy = y - yInt;
    dz = z - zInt;

    dxy = dx * dy;
    oz = 1.f - dz;

    float x0y0z0, x1y0z0, x0y1z0, x0y0z1,x1y1z0, x0y1z1, x1y0z1, x1y1z1;
    int index = xInt  + yInt  * w + zInt  * wh;

    int dfx = xIntP - xInt;
    int dfy = yIntP - yInt;
    int dfz = zIntP - zInt;
            
    x0y0z0 = vol[index];

    x1y0z0 = vol[index +             dfx];
    x0y1z0 = vol[index + dfy * w        ];
    x1y1z0 = vol[index + dfy * w +   dfx];
    x0y0z1 = vol[index + dfz * wh       ];
    x1y0z1 = vol[index + dfz * wh+   dfx];
    x0y1z1 = vol[index + dfy*w + dfz*wh ];
    x1y1z1 = vol[index +dfz*wh+dfy*w+dfx];
   
    float b1 = (x1y0z0* (dx - dxy) + x0y1z0 * (dy - dxy) + x1y1z0*dxy + x0y0z0 * (1-dy -(dx-dxy)));
    float b2 = (x1y0z1* (dx - dxy) + x0y1z1 * (dy - dxy) + x1y1z1*dxy + x0y0z1 * (1-dy -(dx-dxy)));


    return (b1 * oz + b2 * dz);

}

/**
 * @brief Function to perform resampling function that change the size of output
 * @param[in]  in  The input volume
 *             s_w, s_h, s_l size of input volume 
 * @param[out] out  The output volume
 *             d_w, d_h, d_l size of output volume 
 */

void resampling(float* in, float* out, int s_w, int s_h, int s_l, int d_w, int d_h, int d_l){

    int i, j, k;
    
    float ws = (float) d_w / s_w;
    float hs = (float) d_h / s_h;
    float ls = (float) d_l / s_l;
    fprintf(stderr, "Scale %g %g %g \n", ws, hs, ls);

    for (k=0; k < d_l ; ++k)
        for (j=0; j < d_h ; ++j)
            for (i=0; i < d_w ; ++i){
                float z = (float)k / ls;
                float y = (float)j / hs;
                float x = (float)i / ws;
                out[i + j * d_w + k * d_w * d_h] = interpolate(x, y, z, in, s_w, s_h, s_l);
            }
    
}

/**
 * @brief Function to perform downsamling one level
 * @param[in]  in  The input array
 *             w, h, l size of input volume 
 * @param[out] out  Half sampling the input  
 * 
 */

void halfSamplingInput(float* out, float* in, int w, int h, int l){
    int h_w = w /2;
    int h_h = h /2;
    int h_l = l /2;

    int destIdx = 0;
    for (int k=0; k < h_l ; ++k)
        for (int j=0; j < h_h ; ++j)
            for (int i=0; i < h_w ; ++i, ++destIdx){
                out[destIdx] = (get_value(in, 2 * i , 2 * j, 2 * k, w, h, l) +
                                get_value(in, 2 * i + 1 , 2 * j, 2 * k, w, h, l) +
                                get_value(in, 2 * i , 2 * j + 1, 2 * k, w, h, l) + 
                                get_value(in, 2 * i , 2 * j, 2 * k + 1, w, h, l) +
                                get_value(in, 2 * i +1, 2 * j+1, 2 * k, w, h, l) +
                                get_value(in, 2 * i +1, 2 * j, 2 * k+1, w, h, l) +
                                get_value(in, 2 * i , 2 * j+1, 2 * k+1, w, h, l) +
                                get_value(in, 2 * i+1,2 * j+1, 2 * k+1, w, h, l)) * (1.f /8);
            }
}

void upSamplingTransformation(float* out, float* in, int w, int h, int l){
    const int h_w = w /2;
    const int h_h = h /2;
    const int h_l = l /2;

    int destIdx = 0;
    for (int k=0; k < l ; ++k)
        for (int j=0; j < h ; ++j)
            for (int i=0; i < w ; ++i, ++destIdx){

                int i1 = (i -1) / 2;
                int j1 = (j -1) / 2;
                int k1 = (k -1) / 2;

                if (i1 <0 || j1 < 0 || k1 < 0 ||
                    i > w - 2 || j > h - 2 || k  > l - 2)
                    out[destIdx] = 0;
                else {
                    float off_x, off_y, off_z;
                    if ( i & 1) off_x = 0.25f;
                    else off_x = 0.75f;
                
                    if ( j & 1) off_y = 0.25f;
                    else off_y = 0.75f;

                    if ( k & 1) off_z = 0.25f;
                    else off_z = 0.75f;

                    float x, y, z;
                    x = i1 + off_x;
                    y = j1 + off_y;
                    z = k1 + off_z;

                    out[destIdx] = interpolate(x, y, z, in, h_w, h_h, h_l);
                }
            }
}

