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

#ifndef AFFINETRANSFORM_TXX
#define AFFINETRANSFORM_TXX

// Included at end of AffineTransform3D.h
#ifndef SWIG

#include <AtlasWerksException.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <itkTransformFileReader.h>
#include <itkTransformBase.h>
#include <itkAffineTransform.h>

#endif // !SWIG

//------------------------------------------------------------------------
// Public methods

template<class T>
AffineTransform3D<T>& AffineTransform3D<T>::
operator*=( const AffineTransform3D<T>& rhs )
{
    matrix *= rhs.matrix;
    vector = *this * rhs.vector;
    return *this;
}

// If w = Mv + T, then M^-1 w - M^-1 T = v.
template<class T> 
bool AffineTransform3D<T>::
invert()
{
    if (!matrix.invert()) return false;
    vector *= static_cast<T>(-1);
    vector = matrix * vector;
    return true;
}

// Composes two trasformations.  (A*B)*v = A*(B*v).
template<class T> 
AffineTransform3D<T> AffineTransform3D<T>::
operator*( const AffineTransform3D<T>& other ) const
{ return AffineTransform3D<T>(*this, other); }

// Note that v*A is undefined.
template<class T> 
Vector3D<T>
AffineTransform3D<T>::operator*( const Vector3D<T>& v ) const
{
    Vector3D<T> result;
    transformVector( v, result );
    return result;
}

template<class T> 
inline void
AffineTransform3D<T>::transformVector( Vector3D<T>& v ) const
{
    Vector3D<T> in = v;
    transformVector( in, v );
}

// Apply the transformation (*this) to 'in', yielding 'out'.  Avoids
// copying the data structure.
template<class T> 
inline void
AffineTransform3D<T>::transformVector( const Vector3D<T>& in,
                                       Vector3D<T>& out ) const
{
    transformCoordinates( in[0], in[1], in[2], out[0], out[1], out[2] );
}

// template<class T> 
// inline void AffineTransform3D<T>::
// transformCoordinates( const T& xIn, const T& yIn, const T& zIn, 
//                       T& xOut, T& yOut, T& zOut ) const 
// {
//     xOut = xIn*matrix.a[0] + yIn*matrix.a[1] + zIn*matrix.a[2] + vector[0];
//     yOut = xIn*matrix.a[3] + yIn*matrix.a[4] + zIn*matrix.a[5] + vector[1];
//     zOut = xIn*matrix.a[6] + yIn*matrix.a[7] + zIn*matrix.a[8] + vector[2];
// }

template<class T> bool AffineTransform3D<T>::
operator==( const AffineTransform3D<T> t ) const
{
    return ( matrix == t.matrix && vector == t.vector );
}

template <class T>
void
AffineTransform3D<T>
::writePLUNCStyle(const char* filename) const
{
    std::ofstream output(filename);
    if (output.fail() || output.bad())
    {
        throw std::runtime_error("Could not open file.");
    }
    output << "matrix" << std::endl;
    output << matrix(0) << " " << matrix(3) << " " << matrix(6) << " " << 0 
           << std::endl;
    output << matrix(1) << " " << matrix(4) << " " << matrix(7) << " " << 0 
           << std::endl;
    output << matrix(2) << " " << matrix(5) << " " << matrix(8) << " " << 0 
           << std::endl;
    output << vector.x  << " " << vector.y  << " " << vector.z  << " " << 1
           << std::endl;
  
    if (output.bad())
    {
        throw std::runtime_error("Could not write file.");
    }

    output.close();
}


template <class T>
void
AffineTransform3D<T>
::readPLUNCStyle(const char* filename)
{
    std::ifstream input(filename);
    if (input.fail() || input.bad())
    {
        throw std::runtime_error("Could not open file.");
    }

    std::string key;
    std::getline(input, key, '\n');
    key.erase(key.begin()+6, key.end());
    if (key != ("matrix"))
    {
      std::cerr << "WARNING: Unexpected matrix header, could be bad file format" << std::endl;
    }

    T dummy1, dummy2, dummy3, dummy4;

    input >> matrix(0) >> matrix(3) >> matrix(6) >> dummy1;
    input >> matrix(1) >> matrix(4) >> matrix(7) >> dummy2;
    input >> matrix(2) >> matrix(5) >> matrix(8) >> dummy3;
    input >> vector.x  >> vector.y  >> vector.z  >> dummy4;
  
    if (input.fail())
    {
        throw std::runtime_error("Could not read file.");
    }

    input.close();
}

template <class T>
void
AffineTransform3D<T>
::readITKStyle(const char* filename)
{
  #if (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 5) || ITK_VERSION_MAJOR> 4
  itk::TransformFileReaderTemplate< float >::Pointer transformreader = itk::TransformFileReaderTemplate< float >::New();
  #else
  itk::TransformFileReader::Pointer transformreader = itk::TransformFileReader::New();
  #endif
  // read ITK-style transform
  transformreader->SetFileName(filename);
  try{
    transformreader->Update();
    itk::AffineTransform<float,3>::Pointer itkTransform = 
      dynamic_cast<itk::AffineTransform<float,3>*>((*transformreader->GetTransformList()->begin()).GetPointer());
    if(!itkTransform)
      {
	throw AtlasWerksException(__FILE__, __LINE__, 
			       "Could not read transform into an itk::AffineTransform<float, 3>");
      }

    // set the center to zero
//     itk::AffineTransform<float,3>::InputPointType center;
//     center[0] = center[1] = center[2] = 0.0f;
//     itkTransform->SetCenter(center);

    // convert to RealAffineTransform format
    
    itk::Matrix<float,3,3> matrix = itkTransform->GetMatrix();
    itk::FixedArray<float, 3> vector = itkTransform->GetOffset();
    for(int row = 0; row<3; row++){
      for(int col = 0; col<3; col++){
	this->matrix(row, col) = matrix(row, col);
      }
    }
    
    this->vector.x = vector[0];
    this->vector.y = vector[1];
    this->vector.z = vector[2];
    
  }catch (std::exception* e){
    throw AtlasWerksException(__FILE__, __LINE__, "Error reading ITK transform", e);
  }
  
}
//------------------------------------------------------------------------
// Private methods

// Constructs by composing two others.  Makes it easier for the *
// operator to optimize away the copy when the value is returned (I
// think).  Not really appropriate for user code; just use the *
// operator.
template<class T> AffineTransform3D<T>::
AffineTransform3D( const AffineTransform3D<T>& first,
		   const AffineTransform3D<T>& second )
  : matrix( first.matrix * second.matrix ), vector( first * second.vector ) {}

#endif  // ndef AFFINETRANSFORM_TXX
