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

#ifndef __CUDA_LOPERATOR_H
#define __CUDA_LOPERATOR_H

#include <cuda_runtime.h>
#include <Vector3D.h>

class cplReduce;
class cplReduceS;
class cplVector3DArray;

void createTest3D(float *f, int n);
void createTest3D(float *f, Vector3Di& size);

// b = Ax
template<class T>
void matrixMulVector(float* b, T& A, float* x, cudaStream_t stream=NULL);
// r = b - Ax
template<class T>
void computeResidual(float* r, float* b, T& A, float* x, cudaStream_t stream=NULL);


struct poisonMatrix1D{
    poisonMatrix1D(int a_n):n(a_n){

    }

    int getNumElements(){
        return n;
    }

    int n;
};

struct poisonMatrix2D{
    poisonMatrix2D(int a_m, int a_n):m(a_m), n(a_n){

    }
    
    int getNumElements(){
        return n * m;
    }

    int m,n;
};

struct poisonMatrix3D{
    poisonMatrix3D(int a_w, int a_h, int a_l):size(a_w, a_h, a_l){
    }
    
    int getNumElements(){
        return size.productOfElements();
    }

    void print() {
        fprintf(stderr, "Size [%d %d %d]\n", size.x, size.y, size.z);
    }

    Vector3Di size;
};

void matrixMulVector(float* b, poisonMatrix1D& A, float* x, cudaStream_t stream=NULL);
void matrixMulVector(float* b, poisonMatrix2D& A, float* x, cudaStream_t stream=NULL);
void matrixMulVector(float* b, poisonMatrix3D& A, float* x, cudaStream_t stream=NULL);

void createPoison1D(float* a, int n);
void createPoison2D(float* a, int n, int m);
void createPoison3D(float* a, int w, int h, int l);
void createHelmhotlz3D(float* a, int w, int h, int l, float alpha, float gamma);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
struct helmholtzMatrix3D{
    helmholtzMatrix3D(){
    }
    
    helmholtzMatrix3D(int a_w, int a_h, int a_l,float a_alpha, float a_gamma)
        :size(a_w, a_h, a_l), alpha(a_alpha), gamma(a_gamma)   {   }

    void setParams(int a_w, int a_h, int a_l, float a_alpha, float a_gamma){
        size  = Vector3Di(a_w, a_h, a_l); alpha = a_alpha; gamma = a_gamma;
    }

    void setParams(const Vector3Di& s, float a_alpha, float a_gamma){
        size  = s; alpha = a_alpha;   gamma = a_gamma;
    }

    void setSize(int a_w, int a_h, int a_l) { Vector3Di(a_w, a_h, a_l); };
    void setSize(const Vector3Di& s) { size = s; };
    void setDiffParams(float a_alpha, float a_gamma) { alpha = a_alpha; gamma = a_gamma;};

    const Vector3Di& getSize() { return size;}
    void getSolverParams(float& a_alpha, float& a_gamma) { a_alpha = alpha; a_gamma = gamma; };
    unsigned int getNumElements(){
        return size.productOfElements();
    }

    void print() {
        fprintf(stderr, "Size [%d %d %d] alpha %f gamma %f \n", size.x, size.y, size.z, alpha, gamma);
    }
    
    Vector3Di size;
    float     alpha, gamma;
};

// b = Ax
void matrixMulVector(float* b, helmholtzMatrix3D& A, float* x, cudaStream_t stream=NULL);

// r = b - Ax
void computeResidual(float* r, float* b, helmholtzMatrix3D& A, float* x, cudaStream_t stream=NULL);


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
struct helmholtzMatrix3D_cyclic{
    helmholtzMatrix3D_cyclic(){};
    
    helmholtzMatrix3D_cyclic(int a_w, int a_h, int a_l,
                             float a_alpha, float a_gamma)
        :size(a_w, a_h, a_l), alpha(a_alpha), gamma(a_gamma)
        {

        }

    void setParams(int a_w, int a_h, int a_l, float a_alpha, float a_gamma){
        size  = Vector3Di(a_w, a_h, a_l); alpha = a_alpha; gamma = a_gamma;
    }

    void setParams(const Vector3Di& s, float a_alpha, float a_gamma){
        size  = s; alpha = a_alpha;   gamma = a_gamma;
    }

    void setSize(int a_w, int a_h, int a_l) { Vector3Di(a_w, a_h, a_l); };
    void setSize(const Vector3Di& s) { size = s; };
    void setDiffParams(float a_alpha, float a_gamma) { alpha = a_alpha; gamma = a_gamma;};

    const Vector3Di& getSize() { return size;}
    void getSolverParams(float& a_alpha, float& a_gamma) { a_alpha = alpha; a_gamma = gamma; };
    unsigned int getNumElements(){
        return size.productOfElements();
    }

    void print() {
        fprintf(stderr, "Size [%d %d %d] alpha %f gamma %f \n", size.x, size.y, size.z, alpha, gamma);
    }

    Vector3Di size;
    float     alpha, gamma;
};

// b = Ax
void matrixMulVector(float* b, helmholtzMatrix3D_cyclic& A, float* x, cudaStream_t stream=NULL);

// r = b - Ax
void computeResidual(float* r, float* b, helmholtzMatrix3D_cyclic& A, float* x, cudaStream_t stream=NULL);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

struct tridiagonalMatrix{
    float *d_a;
    float *d_a_m;
    float *d_a_p;

    int n;
    int poff, moff;

    tridiagonalMatrix(int a_n, int a_poff, int a_moff):
        d_a(NULL), d_a_m(NULL), d_a_p(NULL),
        n(a_n), poff(a_poff), moff(a_moff){
    }

    int getSize(){
        return n;
    }

    void init();
    void clean();
};

void matrixMulVector(float* b, tridiagonalMatrix& A, float* x, cudaStream_t stream=NULL);

template<class T>
void CG_impl(float* d_b, T& d_A, float* d_x, int imax,
             cplReduce* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream=NULL);

template<class T>
void CG(float* d_b, T& d_A, float* d_x, int imax,
        cplReduce* rd = NULL, float* d_r=NULL, float* d_d=NULL, float* d_q=NULL, cudaStream_t stream=NULL);

template<class T>
void CG(cplVector3DArray& d_b, T& d_A, cplVector3DArray& d_x, int imax,
        cplReduce* rd = NULL, float* d_r=NULL, float* d_d=NULL, float* d_q=NULL, cudaStream_t stream=NULL);


template<class T>
void CG_stream(float* d_b, T& d_A, float* d_x, int imax,
               cplReduceS* rd, float* d_r, float* d_d, float* d_q, float* d_cTemp3, cudaStream_t stream=NULL)    ;

template<class T>
void CG_stream(cplVector3DArray& d_b, T& d_A, cplVector3DArray& d_x, int imax,
               cplReduceS* rd, float* d_r, float* d_d, float* d_q, float* d_cTemp3, cudaStream_t stream=NULL);

//CPU Testing function
void matrixMulVector_cpu(float* b, float* a, float* x, int m, int n);
void computeResidual_cpu(float* r, float* b, float* a, float* x, int m, int n);

void runPoissonResidualTest( int argc, char** argv);
void runPoissionSolverTest(int argc, char** argv);
void runPoissionSolverStreamTest(int argc, char** argv);
void runHelmHoltzTest( int argc, char** argv);
void runPoissionSolverPerformanceTest(int argc, char** argv);

class CGSolverPlan
{
public:
    CGSolverPlan(){};
    void setParams(const Vector3Di& size, float alpha, float gamma);
    void solve(float* d_x,  float* d_b,
               float alpha, float gamma,  const Vector3Di& size, int nIter, 
               cplReduce* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream);
    void solve(cplVector3DArray& d_x, cplVector3DArray& d_b,
               float alpha, float gamma, const Vector3Di& size, int nIter,
               cplReduce* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream);

    const Vector3Di& getSize() { return d_A.getSize();}
private:
    helmholtzMatrix3D d_A;
};

class cplCGSolverStream
{
public:
    cplCGSolverStream():d_cTemp3(NULL){};
    ~cplCGSolverStream();
    void setParams(const Vector3Di& size, float alpha, float gamma);
    void solve(float* d_x,  float* d_b, int nIter, 
               cplReduceS* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream);
    void solve(cplVector3DArray& d_x, cplVector3DArray& d_b, int nIter,
               cplReduceS* rd, float* d_r, float* d_d, float* d_q, cudaStream_t stream);

    void init();
    const Vector3Di& getSize() { return d_A.getSize();}
    void getSolverParams(float& a_alpha, float& a_gamma) { d_A.getSolverParams(a_alpha, a_gamma); };
private:
    helmholtzMatrix3D d_A;
    float* d_cTemp3;
};

#endif
