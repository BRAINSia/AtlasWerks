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

#include <complex>

#include "AtlasWerksTypes.h"

// Real part
Real real(Real r){ return r;}
Real real(Complex c){ return c.real();}

// Imaginary part
Real imag(Real r){ return 0.0f;}
Real imag(Complex c){ return c.imag();}

// Complex conjugate
Real conj(Real r){ return r;}
Complex conj(Complex c){ return std::conj(c);}

// Check if either part is NaN
bool isnan(Real r){ return std::isnan(r);}
bool isnan(Complex c){ return std::isnan(c.real()) || std::isnan(c.imag());}

// Check if either part is infinity
bool isinf(Real r){ return std::isinf(r);}
bool isinf(Complex c){ return std::isinf(c.real()) || std::isinf(c.imag());}
