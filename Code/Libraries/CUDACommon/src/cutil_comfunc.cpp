#include <cutil_comfunc.h>
#include <assert.h>

bool AlmostEqual2sComplement(float A, float B, int maxUlps)
{
    // Make sure maxUlps is non-negative and small enough that the
    // default NAN won't compare as equal to anything.
    assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
    int aInt = *(int*)&A;
    // Make aInt lexicographically ordered as a twos-complement int
    if (aInt < 0)
        aInt = 0x80000000 - aInt;
    // Make bInt lexicographically ordered as a twos-complement int
    int bInt = *(int*)&B;
    if (bInt < 0)
        bInt = 0x80000000 - bInt;
    int intDiff = abs(aInt - bInt);
    if (intDiff <= maxUlps)
        return true;
    return false;
}

static float meanSquareError(float* ref, float* a, int n)
{
    double diff;
    double refMean;

    for (int i=0; i< n; ++i){
        diff += (ref[i] - a[i]) *  (ref[i] - a[i]);
        refMean += ref[i] * ref[i];
    }

    diff /= n;
    refMean /=n;
    return diff /refMean;
}

static bool errorReport(float* h_ref, float* h_a, float eps,  int n)
{
    int first = -1;
    for (int i=0; i < n; ++i)
        if (fabs(h_ref[i] - h_a[i]) > eps) {
            first = i;
            break;
        }
    int   max_i = first;
    float max_e = fabs(h_ref[first]-h_a[first]);
    if (first != -1)
    {
        for (int i=first; i < n; ++i)
            if (fabs(h_ref[i] - h_a[i]) > max_e) {
                max_e = fabs(h_ref[i] - h_a[i]);
                max_i = i;
            }
        fprintf(stderr, "Error first at %d ref %f read %f \n", first, h_ref[first], h_a[first]);
        fprintf(stderr, "Maximum error %f at  %d ref %f read %f \n", max_e, max_i, h_ref[max_i], h_a[max_i]);
        return false;
    }
    return true;
}

int testErrorUlps(float* h_ref, float* d_a, int maxUlps, int n, const char* msg)
{
    float* h_a = new float [n];
    cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool r = true;
    int i =0;
    for (;i< n, r==true; ++i)
        r &= AlmostEqual2sComplement(h_ref[i], h_a[i], maxUlps);

    
    if (r)
        fprintf(stderr, "Test %s PASSED \n", msg);
    else {
        fprintf(stderr, "Test %s FAILED at %d ref %f read %f \n", msg, i-1, h_ref[i-1], h_a[i-1]);
    }

    delete []h_a;
    return r;
}


int testError(float* h_ref, float* d_a, float eps, int n, const char* msg){
    float* h_a = new float [n];
    cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);

    fprintf(stderr, "Testing %s \n", msg);
    bool r = errorReport(h_ref, h_a, eps, n);
    if (r)
        fprintf(stderr, "Test PASSED \n");
    else
        fprintf(stderr, "Test FAILED \n");

    delete []h_a;
    return r;
}

template<class T>
int testError(T* h_ref, T* d_a, int n, const char* msg){
    T* h_a     = new T [n];

    int r = 1;
    cudaMemcpy(h_a, d_a, n * sizeof(T), cudaMemcpyDeviceToHost);

    fprintf(stderr, "Testing %s ...", msg);
    for (int i=0; i < n; ++i) 
        if (h_ref[i] != h_a[i]){
            fprintf(stderr, "Error pos %d: ref=%d value=%d \n", i, h_ref[i], h_a[i]);
            r = 0;
            break;
        }
    if (r)
        fprintf(stderr, "PASSED \n");
    else
        fprintf(stderr, "FAILED \n");

    delete []h_a;
    
    return r;
}

template int testError(int* h_ref, int* d_a, int n, const char* msg);
template int testError(uint* h_ref, uint* d_a, int n, const char* msg);
    
/**
 * @brief Comprare two array and answer if these two are match
 *
 * @param[in]  ref   The reference array
 *             a     The computed array
 *             eps   Allowable error range 
 *             n     The number of elements
 * @returns 1  : the two are the same 
 *          0  : the two are different
 */
bool isMatchH2H(float* ref, float* a, float eps, int n){
    for (int i=0; i< n; ++i)
        if (fabs(ref[i] - a[i]) > eps)
            return false;
    return true;
}
