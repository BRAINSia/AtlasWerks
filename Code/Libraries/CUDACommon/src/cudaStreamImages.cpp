#include <cudaStream.h>
#include <cudaStreamImages.h>
#include <cudaReduceStream.h>
#include <cpl.h>

// Multi-Vector Operations
namespace cplMVOpers{
////////////////////////////////////////////////////////////////////////////////
///
////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Max(T* d_o, std::vector<T* >& h_Imgs, int n, int nImgs,
             cplReduceS* rd, T* d_scratchI[])
    {
        assert(nImgs <= h_Imgs.size());
        copyArrayToDevice(d_scratchI[0], h_Imgs[0], n);

        int i=1;
        for (; i< h_Imgs.size(); ++i){
            copyArrayToDeviceAsync(d_scratchI[i&1], h_Imgs[i], n, STM_H2D);
            (i==1) ? rd->Max(d_o, d_scratchI[(i-1)&1], n, STM_D2D)
                   : rd->MaxA(d_o, d_scratchI[(i-1)&1], n, STM_D2D);
            cudaThreadSynchronize();
        }
        rd->MaxA(d_o, d_scratchI[(i-1)&1], n);
    }
    template void Max(float* d_o, std::vector<float* >& h_Imgs, int n, int nImgs,cplReduceS* rd, float* d_scratchI[]);
    template void Max(int* d_o, std::vector<int* >& h_Imgs, int n, int nImgs, cplReduceS* rd, int* d_scratchI[]);
    template void Max(uint* d_o, std::vector<uint* >& h_Imgs, int n, int nImgs, cplReduceS* rd, uint* d_scratchI[]);

////////////////////////////////////////////////////////////////////////////////
///
////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Sum(T* d_o, std::vector<T* >& h_Imgs,
                 int n, int nImgs,          
                 cplReduceS* rd, T* d_scratchI[])
    {
        assert(nImgs <= h_Imgs.size());
        copyArrayToDevice(d_scratchI[0], h_Imgs[0], n);
        int i=1;
        for (; i< h_Imgs.size(); ++i){
            copyArrayToDeviceAsync(d_scratchI[i&1], h_Imgs[i], n, STM_H2D);
            (i==1) ? rd->Sum(d_o, d_scratchI[(i-1)&1], n, STM_D2D)
                : rd->SumA(d_o, d_scratchI[(i-1)&1], n, STM_D2D);
            cudaThreadSynchronize();
        }
        rd->SumA(d_o, d_scratchI[(i-1)&1], n);
    }

    template void Sum(float* d_o, std::vector<float* >& h_Imgs, int n, int nImgs, cplReduceS* rd, float* d_scratchI[]);
    template void Sum(int* d_o, std::vector<int* >& h_Imgs, int n, int nImgs, cplReduceS* rd, int* d_scratchI[]);
    template void Sum(uint* d_o, std::vector<uint* >& h_Imgs, int n, int nImgs, cplReduceS* rd, uint* d_scratchI[]);

////////////////////////////////////////////////////////////////////////////////
///
////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    void Min(T* d_o, std::vector<T* >& h_Imgs,
                 int n, int nImgs,          
                 cplReduceS* rd, T* d_scratchI[])
    {
        assert(nImgs <= h_Imgs.size());
        copyArrayToDevice(d_scratchI[0], h_Imgs[0], n);
        int i=1;
        for (; i< h_Imgs.size(); ++i){
            copyArrayToDeviceAsync(d_scratchI[i&1], h_Imgs[i], n, STM_H2D);
            (i==1) ? rd->Min(d_o, d_scratchI[(i-1)&1], n, STM_D2D)
                : rd->MinA(d_o, d_scratchI[(i-1)&1], n, STM_D2D);
            cudaThreadSynchronize();
        }
        rd->MinA(d_o, d_scratchI[(i-1)&1], n);
    }

    template
    void Min(float* d_o, std::vector<float* >& h_Imgs, int n, int nImgs, cplReduceS* rd, float* d_scratchI[]);

    template
    void Min(int* d_o, std::vector<int* >& h_Imgs, int n, int nImgs, cplReduceS* rd, int* d_scratchI[]);

    template
    void Min(uint* d_o, std::vector<uint* >& h_Imgs, int n, int nImgs, cplReduceS* rd, uint* d_scratchI[]);

////////////////////////////////////////////////////////////////////////////////
///
////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    void MulC_I(std::vector<T* >& h_Imgs, T c,
                                int n, int nImgs, T* d_scratchI[])
    {
#if 0
        assert(nImgs < h_Imgs.size());
        // load the first image
        copyArrayToDevice(d_scratchI[0], h_Imgs[0], n);
        cplVectorOpers::MulC_I(d_scratchI[0], c, n);

        // load the second image
        if (nImgs > 1)
            copyArrayToDevice(d_scratchI[1], h_Imgs[1], n);

        int i=2;
        for (; i < nImgs; ++i){
            copyArrayToDeviceAsync(d_scratchI[i % 3], h_Imgs[i], n, STM_H2D);
            cplVectorOpers::MulC_I(d_scratchI[(i-1) % 3], c, n, STM_D2D);
            copyArrayFromDeviceAsync(h_Imgs[i-2], d_scratchI[(i-2)%3], n, STM_D2H);
            cudaThreadSynchronize();
        }
    
        if (nImgs> 1)
            cplVectorOpers::MulC_I(d_scratchI[(i-1) % 3], c, n, STM_D2D);
        copyArrayFromDeviceAsync(h_Imgs[i-2], d_scratchI[(i-2) % 3], n, STM_D2H);
        cudaThreadSynchronize();
    
        ++i;
        if (nImgs> 1)
            copyArrayFromDeviceAsync(h_Imgs[i-2], d_scratchI[(i-2) % 3], n, STM_D2H);
#else
        for (int i=0; i < nImgs + 2; ++i) {
            if (i < nImgs)
                copyArrayToDeviceAsync(d_scratchI[i % 3], h_Imgs[i], n, STM_H2D);

            if ((i >=1) && ((i-1) < nImgs))
                cplVectorOpers::MulC_I(d_scratchI[(i-1) % 3], c, n, STM_D2D);

            if ((i >=2) && ((i-2) < nImgs))
                copyArrayFromDeviceAsync(h_Imgs[i-2], d_scratchI[(i-2)%3], n, STM_D2H);
        
            cudaThreadSynchronize();
        }
#endif
    }

    template void MulC_I(std::vector<float* >& h_Imgs, float c, int n, int nImgs, float* d_scratchI[]);
    template void MulC_I(std::vector<int* >& h_Imgs, int c, int n, int nImgs, int* d_scratchI[]);
    template void MulC_I(std::vector<uint* >& h_Imgs, uint c, int n, int nImgs, uint* d_scratchI[]);

    template<typename T>
    void MulC_I(std::vector<T* >& h_Imgs, std::vector<T* >& d_Imgs, T c, int n, int nImgs)
    {
        assert(nImgs < h_Imgs.size());
        for (int i=0; i< nImgs; ++i)
            copyArrayToDeviceAsync(d_Imgs[i], h_Imgs[i], n, getStream(i));
        for (int i=0; i< nImgs; ++i)
            cplVectorOpers::MulC_I(d_Imgs[i], c, n, getStream(i));
        for (int i=0; i< nImgs; ++i)
            copyArrayFromDeviceAsync(h_Imgs[i], d_Imgs[i], n, getStream(i));
    }

    template void MulC_I(std::vector<float* >& h_Imgs, std::vector<float* >& d_Imgs, float c, int n, int nImgs);
    template void MulC_I(std::vector<int* >& h_Imgs, std::vector<int* >& d_Imgs, int c, int n, int nImgs);
    template void MulC_I(std::vector<uint* >& h_Imgs, std::vector<uint* >& d_Imgs, uint c, int n, int nImgs);
};


