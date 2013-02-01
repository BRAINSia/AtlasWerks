//////////////////////////////////////////////////////////////////////
//
// File: Matrix.h
//
// General C++ matrix library that uses CLAPACK.  The datatype used
// is double. The original form of the library is due to Paul
// Yushkevich (see below).  Some modifications by P. Lorenzen. 
//
//////////////////////////////////////////////////////////////////////

/******************************************************************
 * MATRIX Library                                                 *
 ******************************************************************
 * Author:                 Paul Yushkevich
 *
 * Date:                   Apr 1, 1999
 *
 * Description             Basic and numerical matrix operations   
 *	                        See http://www.cs.unc.edu/~pauly/matrix
 *									
 *	Sources:                Uses my compilation of CLAPACK
 *									
 * Dependencies:           CLAPACK
 ******************************************************************
 * matrix.h
 *	---------
 * Declarations for matrix and vector classes
 ******************************************************************/
#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>

typedef long int integer;
typedef double doublereal;
typedef long int longint;

class Vector;
class Matrix;

/**
 * Debug macros
 */
#ifdef _MATRIX_BOUNDS_CHECK
#define dassert(a) assert(a);
#else
#define dassert(a) ;
#endif

// Types of possible matrices
#define GENERAL      0
#define TRIANGULAR   -1
#define SPD          -2

/**
 * Matrix of double
 */
class Matrix {
protected:
   //////////////////////////////////////////////
   // Number of rows and columns in the matrix //
   //////////////////////////////////////////////
   int nRows,nCols,nCells;
   
   ////////////////////
   // The data array //
   ////////////////////
   double *data;
   
   ///////////////////////////////////////////////////////
   // The columns index into the data - for fast access //
   ///////////////////////////////////////////////////////
   double **cols;

   ///////////////////////////
   // A couple init methods //
   ///////////////////////////
   void initStorage(int nRows,int nCols);
   void initData(double *data = NULL,bool columnMajorOrder = true);

   int factorGenericLU(Matrix &LU,integer pivot[]);
   int factorSPDLU(Matrix &LU);

public:
   ////////////////////////////////////////////////////////////////////
   // This constructor initializes the matrix with no data           //
   // Do not attempt matrix operations with such matrices - they wll //
   // crash (assertions will fail).                                  //
   ////////////////////////////////////////////////////////////////////
   Matrix() : nRows(0),nCols(0),nCells(0),data(NULL),cols(NULL) {
   }
   
   ///////////////////////////////////////////////////////////////////////////
   // This is the right way to initialize the matrix.  Data contains values //
   // in column major order, unless the flag is changed                     //
   // This is also a default constructor                                    //
   ///////////////////////////////////////////////////////////////////////////
   Matrix(int rows,int columns,double *data=NULL,double **cols=NULL,bool columnMajorOrder=true) {
     initStorage(rows,columns);
     initData(data,columnMajorOrder);
   }

   /////////////////////////////////////////////////////////////////
   // Creates a matrix and reads in values in column major order  //
   // Make sure that constants are specified correctly,           //
   // e.g. MatrixTmp<double> (1,2,3,3) will                       //
   // cause problems but MatrixTmp<double> (1,2,3.0,3.0) will not //
   /////////////////////////////////////////////////////////////////
   Matrix(int rows,int columns,double firstValue,...);
   
   //////////////////////
   // Copy constructor //
   //////////////////////
   Matrix(const Matrix &m) {
     if (m.nRows * m.nCols == 0)
       {
	 nRows = m.nRows;
	 nCols = m.nCols;
	 nCells = 0;
	 data = 0;
	 cols = 0;
       }
     else
       {
	 initStorage(m.nRows,m.nCols);
	 initData(m.data);
       }
   }

   ////////////////
   // Destructor //
   ////////////////
   virtual ~Matrix() {
      if(data) {
         delete[] data;
		 data = NULL;
	  }
	  if (cols) {
		  delete[] cols;
		  cols = NULL;
	  }
   }

   //////////////////////////////////////////////////////
   // Read only member access - inline for extra speed //
   //////////////////////////////////////////////////////
   const double operator() (int row, int col) const {
     dassert(data && cols);
     dassert(row >= 0 && row < nRows);
     dassert(col >= 0 && col < nCols);
     return( cols[col][row] );
   }

   ///////////////////////////////////////////////////////
   // Read-Write member access - inline for extra speed //
   ///////////////////////////////////////////////////////
   double& operator() (int row,int col) {
     dassert(data && cols);
     dassert(row >= 0 && row < nRows);
     dassert(col >= 0 && col < nCols);
     return( cols[col][row] );
   }

   //////////////////////////////////////////////////////////////////
   // Set all elements of a matrix to a number; different from M=I //
   /////////////////////// ///////////////////////////////////////////
   void setAll(double k) {
     for(int i=0;i < nCells;i++) {
       data[i] = k;
     }
   }

   ////////////////////
   // Resizes matrix //
   ////////////////////
   void setSize(int rows,int columns) {
     if(rows != nRows || columns != nCols) {
       if(data) {
	 delete[] data;
		data = NULL;
	   }
	   if(cols) {
		   delete[] cols;
		   cols = NULL;
	   }
     
       initStorage(rows,columns);
     }
     initData(NULL);      
   }

   //////////////
   // Equality //
   //////////////
   bool equal( const Matrix& matrix, double tolerance );

   /////////////////////
   // Matrix addition //
   /////////////////////
   virtual void operator += (const Matrix &A) {
      dassert(nRows == A.nRows && nCols == A.nCols);
      for(int i=0;i < nCells;i++)
         data[i] += A.data[i];
   }

   virtual Matrix operator +(const Matrix &A) const { 
      dassert(nRows == A.nRows && nCols == A.nCols);
      Matrix C;
      C.initStorage(nRows,nCols);

      for(int i=0;i < nCells;i++) {
	C.data[i] = data[i] + A.data[i];
      }

      return C;
   }

   /////////////////////////
   // Matrix subtraction  //
   /////////////////////////
   virtual void operator -= (const Matrix &A) {
     dassert(nRows == A.nRows && nCols == A.nCols);
     for(int i=0;i < nCells;i++) {
       data[i] -= A.data[i];
     }
   }   

   virtual Matrix operator -(const Matrix &A) const { 
      dassert(nRows == A.nRows && nCols == A.nCols);
      Matrix C;
      C.initStorage(nRows,nCols);

      for(int i=0;i < nCells;i++)
         C.data[i] = data[i] - A.data[i];
      
      return C;
   }

   // Matrix multiplication
   virtual void operator *= (const Matrix &A) {
      *this = *this * A;
   }

   virtual Matrix operator * (const Matrix &A) const;

   // Multiplication by constant
   virtual void operator *= (const double k) {
      for(int i=0;i < nCells;i++)
         data[i] *= k;
   }

   virtual Matrix operator *(const double k) const { 
      Matrix C;
      C.initStorage(nRows,nCols);

      for(int i=0;i < nCells;i++)
         C.data[i] = data[i] * k;
      
      return C;
   }

   virtual void operator /= (const double k) {
      for(int i=0;i < nCells;i++)
         data[i] /= k;
   }

   virtual Matrix operator /(const double k) const { 
      Matrix C;
      C.initStorage(nRows,nCols);

      for(int i=0;i < nCells;i++)
         C.data[i] = data[i] / k;
      
      return C;
   }

   // Negation
   virtual void negate() {
      for(int i=0;i < nCells;i++)
         data[i] = -data[i];
   }

   virtual Matrix operator -() const { 
      Matrix C;
      C.initStorage(nRows,nCols);

      for(int i=0;i < nCells;i++)
         C.data[i] = -data[i];
      
      return C;
   }

	// This assignment operator assign the matrix k*I to a square matrix
   Matrix& operator= (const double k) {
      dassert(nRows==nCols);
      initData(NULL);
      for(int i=0;i<nRows;i++)
         cols[i][i] = k;
      return *this;
   }

   // Copy operator
   Matrix& operator= (const Matrix &A) {
      if(nRows != A.nRows || nCols != A.nCols) {
         if(data) {
            delete[] data;
			data = NULL;
		 }
		 if (cols) {
            delete[] cols;
			cols = NULL;
         }
         initStorage(A.nRows,A.nCols);
      }
      initData(A.data);
      return *this;
   }

   // Transpose operator - returns the transpose
   virtual Matrix t() const;

   // Compute the determinant
   double det(int type=GENERAL);

   // Solve A*x = B system
   int solveGE(const Matrix &A,Matrix &C,int type=GENERAL);

   // LU Factorization
   int factorLU(Matrix &L, Matrix &U, Matrix &P,int type=GENERAL);

   // SVD Factorization
   int factorSVD(Matrix &U,Matrix &Vt,Vector &sv,bool economy=false,int type=GENERAL);

   // Eigenvalue analysis
   int factorEV(Vector &l,Matrix &V,int type=GENERAL);

   int factorEVgen(Vector &el,Vector &er,Matrix &VL,Matrix &VR,
		   int type=GENERAL);
   // Get inverse
   int inverse(Matrix &Ainv,int type=GENERAL);

   // get/extract submatrices
   virtual void insertMatrix(int row,int col,const Matrix &m);
   virtual void extractMatrix(int row,int col,Matrix &m);
   virtual Vector getColumn(int col);
   virtual Vector getRow(int row);

   int rows() const {
      return nRows;
   }

   int columns() const {
      return nCols;
   }

   /////////////////////////
   // Norms of the matrix //
   /////////////////////////
   double oneNorm() const ;
   double infinityNorm() const;
   double pNorm(double p) const;
   double twoNorm() const;
   double frobeniusNorm() const;

   double trace() const;

   ///////////////////////
   // Swap rows/columns //
   ///////////////////////
   virtual void swapRows(int r1,int r2);
   virtual void swapColumns(int c1,int c2);

   /////////
   // I/O //
   /////////
   bool readFromFile( char* fileName );
   bool writeToFile( char* fileName );
   void print();

}; // class Matrix

// Backwards multiplication operator
inline Matrix operator * (const double k,const Matrix &A) {
   return A*k;
}


/**
 * Generic vector class - a Nx1 matrix
 */
class Vector : public Matrix
{
protected:
public:
  Vector() : Matrix() {};

  // Creates a vector, feeds it some data
  Vector(int dim,double *data=NULL) : Matrix(dim,1,data) {};

  // Creates a vector, may be initialized with doubles (!!!)
  Vector(int dim,double firstValue,...);
  /*
  // Creates a vector, all entries initialized with a common double value
  Vector(int dim, double value);
  */
  Vector(const Matrix &m) : Matrix(m) {
    dassert(m.columns() == 1);
  }

  // Read only access operator
  double operator() (int row) const {
    return data[row];
  }

  // Read-write access operator
  double &operator() (int row) {
    return data[row];
  }

  // Set size
  void setSize(int rows) {
    Matrix::setSize(rows,1);
  }

  // Cross product
  virtual Vector cross (const Vector& v) {
    dassert(nRows == v.nRows && nRows == 3);
    Vector result(nRows);

    result.data[0] = data[1]*v.data[2] - data[2]*v.data[1];
    result.data[1] = data[2]*v.data[0] - data[0]*v.data[2];
    result.data[2] = data[0]*v.data[1] - data[1]*v.data[0];

    return result;
  }

  // Dot product
  double dotProduct(const Vector &v){
    dassert(nRows == v.nRows);

    double dp = 0.0;
    for(int i=0;i<nRows;i++)
      dp += data[i] * v.data[i];
    return dp;
  }

  // Normalizes the vector
  virtual void normalize() {
    double len = twoNorm();
    if(len != 0)
      (*this) /= len;
  }

  // Return number of rows
  virtual int size() const {
    return rows();
  }

};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

Matrix *allocMatrixArray(int nMatrices,int rows,int columns);
Vector *allocVectorArray(int nVectors,int size);

double unif_rand_dbl(long *idum);
int randint(int thismax, long *seed);

#endif

