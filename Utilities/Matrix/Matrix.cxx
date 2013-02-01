//////////////////////////////////////////////////////////////////////
//
// File: Matrix.cxx
//
// General C++ matrix library that uses CLAPACK.  The datatype used
// is double. The original form of the library is due to Paul
// Yushkevich (see below).  Some modifications by P. Lorenzen. 
//
//////////////////////////////////////////////////////////////////////

/******************************************************************
 * MATRIX Library                                                 *
 ******************************************************************
 * Author:		Paul Yushkevich
 *
 * Date:		Apr 1, 1999
 *
 * Description	        Basic and numerical matrix operations   
 *			See http://www.cs.unc.edu/~pauly/matrix
 *									
 *	Sources:	Uses my compilation of CLAPACK
 *									
 * Dependencies:				CLAPACK
 ******************************************************************
 * matrix.cpp
 *	-----------
 * Definitions for matrix and vector classes
 ******************************************************************/
#include <memory.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdarg.h>

#include "Matrix.h"

#ifdef WIN32

extern "C" {
   int dposv_(char *uplo, integer *n, integer *nrhs, doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *info);

   int dgesv_(integer *n, integer *nrhs, doublereal *a, integer *lda, integer *ipiv, doublereal *b, integer *ldb, integer *info);

   int dgesvd_(char *jobu, char *jobvt, integer *m, integer *n, doublereal *a, integer *lda, doublereal *s, doublereal *u, integer *
        ldu, doublereal *vt, integer *ldvt, doublereal *work, integer *lwork, integer *info);

   int dsyev_(char *jobz, char *uplo, integer *n, doublereal *a,integer *lda, doublereal *w, doublereal *work, integer *lwork, integer *info);
   
   int dgetri_(integer *n, doublereal *a, integer *lda, integer *ipiv, doublereal *work, integer *lwork, integer *info);

   int dgeev_(char *jobvl, char *jobvr, integer *n, doublereal *a, integer *lda, doublereal *wr, doublereal *wi, doublereal *vl, integer *ldvl, doublereal *vr, integer *ldvr, doublereal *work, integer *lwork, integer *info);
}

#else
extern "C" {
   int dposv_(char *uplo, integer *n, integer *nrhs, doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *info);
   int dgesv_(integer *n, integer *nrhs, doublereal *a, integer *lda, integer *ipiv, doublereal *b, integer *ldb, integer *info);
   int dgesvd_(char *jobu, char *jobvt, integer *m, integer *n, doublereal *a, integer *lda, doublereal *s, doublereal *u, integer *
        ldu, doublereal *vt, integer *ldvt, doublereal *work, integer *lwork, integer *info);
   int dsyev_(char *jobz, char *uplo, integer *n, doublereal *a,integer *lda, doublereal *w, doublereal *work, integer *lwork, integer *info);
   int dgetri_(integer *n, doublereal *a, integer *lda, integer *ipiv, doublereal *work, integer *lwork, integer *info);
   int dgeev_(char *jobvl, char *jobvr, integer *n, doublereal *a, integer *lda, doublereal *wr, doublereal *wi, doublereal *vl, integer *ldvl, doublereal *vr, integer *ldvr, doublereal *work, integer *lwork, integer *info);
 }
#endif

void Matrix::initData(double *inData,bool columnMajorOrder) 
{
    if(inData) {
        if(columnMajorOrder) {
            memcpy(data,inData,sizeof(double)*nCells);
        }
        else {
            for(int iRow=0;iRow<nRows;iRow++) {
                for(int iCol=0;iCol<nCols;iCol++) {
                    cols[iCol][iRow] = *inData;
                    inData++;
                }
            }
        }
    }
    else {
        memset(data,0,sizeof(double)*nCells);
    }
}

// Initializes a matrix to new size
void Matrix::initStorage(int rows,int columns) {
   nRows = rows;
   nCols = columns;
   nCells = nRows * nCols;
   data = new double[nCells];
   cols = new double*[nCols];

   cols[0] = data;
   for(int i=1;i<nCols;i++) {
      cols[i] = cols[i-1] + nRows;
   }
}

Matrix::Matrix(int rows,int columns,double firstValue...) {
   initStorage(rows,columns);

   // Read in values.  Values are in column major order
   va_list ap;
   va_start(ap, firstValue);

   for(int iCol=0;iCol<nCols;iCol++) {
      for(int iRow=0;iRow<nRows;iRow++) {
         if(iCol==0 && iRow==0)
            cols[iCol][iRow] = firstValue;
         else
            cols[iCol][iRow] = va_arg(ap,double);
      }
   }

   va_end(ap);
}


Matrix Matrix::operator*(const Matrix &A) const {
   dassert(nCols == A.nRows);

   register int i,j,k;

   // Initialize destination matrix with all zeros
   Matrix C(nRows,A.nCols);

   // Compute product
   for(i=0;i<C.nRows;i++) {
      for(j=0;j<C.nCols;j++) {
         for(k=0;k<nCols;k++) {
            C.cols[j][i]  += cols[k][i] * A.cols[j][k];
         }
      }
   }

   return C;
}


/**
 * Get transpose of a matrix
 */
Matrix Matrix::t() const {
   Matrix T(nCols,nRows);

   for(int iRow=0;iRow<T.nRows;iRow++) {
      for(int iCol=0;iCol<T.nCols;iCol++) {
         T.cols[iCol][iRow] = cols[iRow][iCol];
      }
   }

	return T;
}

int Matrix::factorGenericLU(Matrix &LU,integer pivot[]) {
   // Copy our contents to the LU matrix
   LU = *this;

   // Data used in call
   integer n = nRows;
   integer nrhs = 0;
   double *a = LU.data;
   integer lda = n;   
   double *b = NULL;
   integer ldb = n;
   integer info = 0;

   // Perform LU call
   dgesv_(&n,&nrhs,a,&lda,pivot,b,&ldb,&info);

   return (int) info;
}

int Matrix::factorSPDLU(Matrix &LU) {
   // Copy our contents to the LU matrix
   LU = *this;

   // Data used in call
   integer n = nRows;
   integer nrhs = 0;
   double *a = data;
   integer lda = n;   
   double *b = NULL;
   integer ldb = n;
   integer info = 0;

   // Perform LU call
   char uplo = 'U';
   dposv_(&uplo,&n,&nrhs,a,&lda,b,&ldb,&info);

   return (int) info;
}

/**
 * LU Factorization of a matrix - 'public' call
 */
int Matrix::factorLU(Matrix &L, Matrix &U, Matrix &P,int type) {
   int info;

   P.setSize(nRows,nRows);
   P = 1;

   Matrix LU;

   if(type==SPD) {
      info = factorSPDLU(LU);      
   }

   else {
      integer *pivot = new integer[nRows];
      info = factorGenericLU(LU,pivot);

      // Create a permutation matrix
      for(int iRow=0;iRow<nRows;iRow++) {
         if(pivot[iRow] && pivot[iRow]-1!=iRow) {
            P.swapColumns(iRow,pivot[iRow]-1);
         }
      }

      delete [] pivot;
   }

   // Construct L, U
   L = LU;
   U = LU;

   for(int iRow=0;iRow<nRows;iRow++) {
      for(int iCol=0;iCol<iRow;iCol++) {
         U.cols[iCol][iRow] = 0;
         L.cols[iRow][iCol] = 0;
      }
      L.cols[iRow][iRow] = 1;
   }

   return (int) info;
}

/**
 * Perform Gaussian elimination, get back LU decomp.
 */
int Matrix::solveGE(const Matrix &B,Matrix &X,int type) {
   dassert(nRows == nCols);
   dassert(B.nRows == nRows);

   // Create a copy of ourselves
   Matrix LU;
   LU = *this;

   // Create a copy of B
   X = B;

   // Data used in call
   integer n = nRows;
   integer nrhs = B.nCols;
   double *a = data;
   integer lda = n;   
   double *b = X.data;
   integer ldb = n;
   integer info = 0;
   
   // Call to gaussian eliminator
   if(type==SPD) {
      char uplo = 'U';
      dposv_(&uplo,&n,&nrhs,a,&lda,b,&ldb,&info);
   }
   else {
      integer *ipiv = new integer[n];
      dgesv_(&n,&nrhs,a,&lda,ipiv,b,&ldb,&info);
      delete [] ipiv;
   }

   return (int)info;
}


/**
 * Compute matrix determinant - Ill posed problem
 */
double Matrix::det(int type) {
   dassert(nRows == nCols);
   double det=1;

   Matrix LU;
   
   if(type==SPD) {
      factorSPDLU(LU);
   }
   else {
      integer *pivot = new integer[nRows];
      factorGenericLU(LU,pivot);

      // Compute sign of determinant 
      for(int iRow=0;iRow<nRows;iRow++) {
         if(pivot[iRow] != iRow+1)
            det = 0 - det;
      }
      delete [] pivot;
   }

   for(int iRow=0;iRow<nRows;iRow++) {
      det*=LU(iRow,iRow);
   }

   return det;
}


//////////////////////////////////////////////////////////////////////
//
// Matrix::equal
//
//////////////////////////////////////////////////////////////////////
bool
Matrix::equal( const Matrix& matrix, double tolerance )
{
  if( (nRows != matrix.rows()) ||
      (nCols != matrix.columns()) ) {
    return( false );
  }

  double difference;
  for( int iRow = 0; iRow < nRows; iRow++ ) {
    for( int iCol = 0; iCol < nCols; iCol++ ) {

      difference = fabs( matrix.cols[iCol][iRow] - cols[iCol][iRow] );

      if( difference > tolerance ) {
	return( false );
      }
    }
  }  

  return( true );
} // Matrix::equal


//////////////////////////////////////////////////////////////////////
//
// Matrix::readFromFile
//
//////////////////////////////////////////////////////////////////////
bool
Matrix::readFromFile( char* fileName )
{
  std::ifstream inStream( fileName, std::ios::in );
  
  if( !inStream ) {
   std::cerr << "[Matrix::readFromFile] error: could not read file ";
   std::cerr << fileName <<std::endl;
    return( false );
  }

  /////////////////////
  // Read dimensions //
  /////////////////////
  int numRows;
  int numCols;
  inStream >> numRows >> numCols;

  /////////////////////
  // Allocate memory //
  /////////////////////
  setSize( numRows, numCols );

  /////////////////////////
  // Populate the matrix //
  /////////////////////////
  for( int iRow = 0; iRow < numRows; iRow++ ) {
    for( int iCol = 0; iCol < numCols; iCol++ ) {
      inStream >> (*this)(iRow, iCol);
    }
  }

  inStream.close();

  return( true );
} // Matrix::readFromFile


//////////////////////////////////////////////////////////////////////
//
// Matrix::writeToFile
//
//////////////////////////////////////////////////////////////////////
bool
Matrix::writeToFile( char* fileName )
{
  return( true );
} // Matrix::writeToFile


//////////////////////////////////////////////////////////////////////
//
// Matrix::print
//
//////////////////////////////////////////////////////////////////////
void 
Matrix::print()
{
  for(int iRow=0;iRow<nRows;iRow++) {
    for(int iCol=0;iCol<nCols;iCol++) {
      std::cout << (*this)(iRow,iCol) << "\t";
    }
    std::cout <<std::endl;
  }
  std::cout <<std::endl;
} // Matrix::print

/**
 * Insert matrix into this matrix with top left of insreted matrix at row,col
 */
void Matrix::insertMatrix(int row,int col,const Matrix &A) {
	dassert(row + A.nRows <= nRows);
   dassert(col + A.nCols <= nCols);

   for(int iRow=row;iRow<row + A.nRows;iRow++) {
      for(int iCol=col;iCol<col + A.nCols;iCol++) {
         cols[iCol][iRow] = A.cols[iCol-col][iRow-row];
      }
   }
}

// Extracts a sub-matrix
void Matrix::extractMatrix(int row,int col,Matrix &A) {
	dassert(row + A.nRows <= nRows);
   dassert(col + A.nCols <= nCols);

   for(int iRow=row;iRow<row + A.nRows;iRow++) {
      for(int iCol=col;iCol<col + A.nCols;iCol++) {
         A.cols[iCol-col][iRow-row] = cols[iCol][iRow];
      }
   }
}

/**
 * Get column vector
 */
Vector Matrix::getColumn(int col) {
	dassert(col < nCols);

	Vector v(nRows);

   for(int iRow=0;iRow<nRows;iRow++) {
		v.data[iRow] = cols[col][iRow];
   }

   return v;
}

/**
 * Get row vector
 */
Vector Matrix::getRow(int row) {
	dassert(row < nRows);

	Vector v(nCols);

   for(int iCol=0;iCol<nCols;iCol++) {
		v.data[iCol] = cols[iCol][row];
   }

   return v;
}

/**
 * Swap two rows
 */
void Matrix::swapRows(int r1,int r2) {
	dassert(r1 < nRows);
   dassert(r2 < nRows);

   for(int iCol=0;iCol<nCols;iCol++) {
      double tmp = cols[iCol][r1];
      cols[iCol][r1] = cols[iCol][r2];
      cols[iCol][r2] = tmp;
   }
}

/**
 * Swap two columns
 */
void Matrix::swapColumns(int c1,int c2) {
	dassert(c1 < nCols);
   dassert(c2 < nCols);

   for(int iRow=0;iRow<nRows;iRow++) {
      double tmp = cols[c1][iRow];
      cols[c1][iRow] = cols[c2][iRow];
      cols[c2][iRow] = tmp;
   }
}

/**
 * SVD Factorization

    The SVD is written

         A = U * SIGMA * transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
    V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are double and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note that the routine returns transpose(V), not V.
 */
int Matrix::factorSVD(Matrix &U,Matrix &Vt,Vector &sv,bool economy,int type) {

   // Have to create a copy of A so it does not get destroyed
   Matrix A(*this);

   // Compute the smaller of two dimensions
   int nsv = (nRows < nCols) ? nRows : nCols;

   // Input to svd method (get all rows/cols of u,vt)
   char jobu = 'A';
   char jobvt = 'A';
   integer ldu = nRows;
   integer ldvt = nCols;

   // Set size of U,Vt
   if(economy){
     U.setSize(nRows,nsv);
     Vt.setSize(nsv,nCols);
     jobu = 'S';
     jobvt = 'S';
     ldu = nsv;
     ldvt = nsv;
   }else{
     U.setSize(nRows,nRows);
     Vt.setSize(nCols,nCols);
   }
   sv.setSize(nsv);

   integer m = nRows;
   integer n = nCols;
   double *a = A.data;
   integer lda = m;
   double *s = sv.data;
   double *u = U.data;
   //integer ldu = m;
   double *vt = Vt.data;
   //integer ldvt = n;
   integer lwork = 6*(m+n);
   double *work = new double[lwork];
   integer info;

   // Call svd method
   dgesvd_(&jobu,&jobvt,&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,work,&lwork,&info);

   delete [] work;

   return (int) info;
}

/**
 * Last biggie - eigenvectors/eigenvalues
 * A = V * LAMBDA * V.t
 * LAMBDA contains eigenvalues of A on it diagonals
 * A must be symmetrical.
 */
int Matrix::factorEV(Vector &l,Matrix &V,int type) {
   dassert(nCols == nRows);

   V = *this;
   l.setSize(nCols);
   
   // Parameters
   char jobz = 'V';
   char uplo = 'U';
   integer n = nCols;
   double *a = V.data;
   integer lda = n;
   double *w = l.data;
   integer lwork = 6*n;
   double *work = new double[lwork];
   integer info = 0;

   dsyev_(&jobz, &uplo, &n, a,
          &lda, w, work, &lwork, 
          &info);
 
   delete [] work;
   return (int)info;
}

/**
 * er and el hold the real and imaginary parts of the eigenvalues,
 * respectively, while VL and VR hold the left and right eigenvectors,
 * respectively, in their columns.
 */
int Matrix::factorEVgen(Vector &el,Vector &er,Matrix &VL,Matrix &VR,int type) 
{
   dassert(nCols == nRows);

   Matrix V = *this;
   el.setSize(nCols);
   er.setSize(nCols);
  
   // Parameters

   // compute left eigenvectors
   char jobvl = 'V';
   // compute right eigenvectors
   char jobvr = 'V';
   integer n = nCols;
   // input array
   double *a = V.data;
   integer lda = n;
   // real part of computed eigenvalues
   double *wr = er.data;
   // imaginary part of computed eigenvalues
   double *wl = el.data;
   // output of the right eigenvectors.
   double *vr = VR.data;
   // output of the left eigenvectors
   double *vl = VL.data;
   // work array
   integer lwork = 6*n;
   double *work = new double[lwork];
   integer info = 0;

   dgeev_(&jobvl, &jobvr, &n, a,
          &lda, wr, wl,vl, &n, vr, &n, work, &lwork,
          &info);
 
   delete [] work;
   return (int)info;
}


/**
 * Oops - one more - inverse calc.
 */
int Matrix::inverse(Matrix &Ainv,int type) {
   dassert(nCols == nRows);
   
   // Do LU factorization
   
   integer n = nCols;
   integer lda = n;
   integer *ipiv = new integer[n];
   integer lwork = 4 * n;
   double *work = new double[lwork];
   integer info;

   factorGenericLU(Ainv,ipiv);
   double *a = Ainv.data;

   // Compute inverse
   dgetri_(&n,a,&lda,ipiv,work,&lwork,&info);

   // Delete trash
   delete [] ipiv;
   delete [] work;

   return (int)info;
}

// Norms of the matrix
double Matrix::oneNorm() const {
   double rtn = 0;
   for(int i=0;i<nCells;i++) 
      rtn += fabs(data[i]);
   return rtn;
}

double Matrix::infinityNorm() const {
   double rtn = 0;
   for(int i=0;i<nCells;i++) {
      double val = fabs(data[i]);
      rtn = (val > rtn) ? val : rtn;
   }
   return rtn;
}

double Matrix::twoNorm() const {
   double rtn = 0;
   for(int i=0;i<nCells;i++) 
      rtn += data[i] * data[i];
   return sqrt(rtn);
}

double Matrix::pNorm(double p) const {
   double rtn = 0;
   for(int i=0;i<nCells;i++) 
      rtn += pow(fabs(data[i]),p);
   return pow(rtn,1/p);
}

double Matrix::frobeniusNorm() const {
  Matrix temp(*this);
  temp = temp * temp.t();
  return(sqrt(temp.trace()));
}

double Matrix::trace() const {
  double rtn = 0;

  for(int iCol=0;iCol<nCols;iCol++) {
    rtn += cols[iCol][iCol];
  }

  return(rtn);
}

Vector::Vector(int rows,double firstValue,...) : Matrix(rows,1) {
   // Read in values.  Values are in column major order
   va_list ap;
   va_start(ap, firstValue);

   data[0] = firstValue;
   for(int iRow=1;iRow<nRows;iRow++) {
      data[iRow] = va_arg(ap,double);
   }

   va_end(ap);
}

/*
Vector::Vector(int rows,double commonValue) : Matrix(rows,1) {
   // Read in values.  Populates all entries by a common value in column major order
   
   for(int iRow=0;iRow<nRows;iRow++) {
      data[iRow] = commonValue;
   }   
}
*/
/* (C) Copr. 1986-92 Numerical Recipes Software 42,. */

/************************************************************************/
/* unif_rand_dbl.C -- contains routines to generate uniform random      */
/*      floats or reals in range [0.0, 1.0], and uniform random       */
/*      integers in the range [0, n], for n specified.                  */
/*                                                                      */
/*      Contents:  rand3() -- uniform random floats in range [0.0, 1.0] */
/*                 unif_rand_dbl() -- as above, but returns reals     */
/*                 randint() -- returns uniform random ints,            */
/*                      in range [0, n] for n specified.                */
/*                                                                      */
/*      Author:  rand3() is from Numerical Recipes in C, 2nd ed.        */
/*              mods and other code by A. Thall                         */
/*      Date:  2. Feb. 1997                                             */
/************************************************************************/
#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC (1.0/MBIG)

// formerly, ran3(), from numeric recipes.

float rand3(long *idum)
{
   static int inext,inextp;
   static long ma[56];
   static int iff=0;
   long mj,mk;
   int i,ii,k;

   if (*idum < 0 || iff == 0) {
      iff=1;
      mj=MSEED-(*idum < 0 ? -*idum : *idum);
      mj %= MBIG;
      ma[55]=mj;
      mk=1;
      for (i=1;i<=54;i++) {
         ii=(21*i) % 55;
         ma[ii]=mk;
         mk=mj-mk;
         if (mk < MZ) mk += MBIG;
         mj=ma[ii];
      }
      for (k=1;k<=4;k++)
         for (i=1;i<=55;i++) {
            ma[i] -= ma[1+(i+30) % 55];
            if (ma[i] < MZ) ma[i] += MBIG;
         }
      inext=0;
      inextp=31;
      *idum=1;
   }
   if (++inext == 56) inext=1;
   if (++inextp == 56) inextp=1;
   mj=ma[inext]-ma[inextp];
   if (mj < MZ) mj += MBIG;
   ma[inext]=mj;
   return (float) mj*FAC;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
{
    if (matrix.rows() > 10 || matrix.columns() > 10) {
        os << "[" << matrix.columns()
           << "x" << matrix.rows()
           << " matrix]";
    } else {
        for (int row = 0; row < matrix.rows(); ++row) {
            for (int col = 0; col < matrix.columns(); ++col) {
                os << matrix(row, col) << " ";
            }
            os << '\n';
        }
    }
    return os;
}

/************************************************************************/
/* unif_rand_dbl() returns a uniform random variate between [0.0, 1.0]. */
/*      (based on NumRec routine rand3(), based itself on this-or-that  */
/*      from Knuth.  Not a linear congruence generator.                 */
/* To initialize/reinitialize, pass it a negative long int; it has a    */
/*      memory, so passing it the same initializer multiple times       */
/*      during a run of the program will produce different values.      */
/************************************************************************/
double unif_rand_dbl(long *idum)
{
   double highorder = (double) rand3(idum);
   double loworder = (double) rand3(idum);

   return highorder + loworder*FAC;
}

/************************************************************************/
/* randint() -- returns a uniformly distributed random integer in the   */
/*      range [0, thismax], based on a scaled unif_rand_dbl() value.    */
/************************************************************************/
int randint(int thismax, long *seed)
{
    double scaleval = (thismax + 1)*unif_rand_dbl(seed);
    return (int) scaleval;
}

#undef MBIG
#undef MSEED
#undef MZ
#undef FAC


Matrix *allocMatrixArray(int nMatrices,int rows,int columns) {
   Matrix *array = new Matrix[nMatrices];
   for(int i=0;i<nMatrices;i++) {
      array[i].setSize(rows,columns);
   }
   return array;
}


Vector *allocVectorArray(int nVectors,int size) {
   Vector *array = new Vector[nVectors];
   for(int i=0;i<nVectors;i++) {
      array[i].setSize(size);
   }
   return array;
}

