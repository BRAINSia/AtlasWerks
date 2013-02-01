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

#ifndef MATRIX3DUTILS_H
#define MATRIX3DUTILS_H

#include "Matrix3D.h"
#include "Vector3D.h"

#include "Matrix.h"

//-------------------------------------------------------------------------
// Conversion with Yuskevich's Matrix class.

template<typename T>
Matrix 
castMatrix3DToMatrix(const Matrix3D<T>& matrix)
{
  Matrix result(3, 3);
  for( size_t j = 0; j < 3; ++j ) {
    for( size_t i = 0; i < 3; ++i ) {
      result(i, j) = static_cast<double>(matrix(i, j));
    }
  }
  return result;
}

template<typename T>
Vector 
castVector3DToVector(const Vector3D<T>& v)
{
  Vector result(3);
  for( size_t i = 0; i < 3; ++i ) {
    result(i) = static_cast<double>(v[i]);
  }
  return result;
}

Vector3D<double>
castVectorToVector3D(const Vector& vector)
{
    assert(vector.size() == 3);
    return(Vector3D<double>(vector(0), vector(1), vector(2)));
}

Matrix3D<double>
castMatrixToMatrix3D(const Matrix& matrix)
{
    assert(matrix.rows() == 3 && matrix.columns() == 3);
    Matrix3D<double> result;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            result(i, j) = matrix(i, j);
        }
    }
    return result;
}

#endif  // ndef MATRIX3DUTILS_H
