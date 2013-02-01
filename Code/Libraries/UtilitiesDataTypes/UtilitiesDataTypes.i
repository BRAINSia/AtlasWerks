%module UtilitiesDataTypes

%import "DataTypes.i"
%include "typemaps.i"

%feature("autodoc","1");

%{
#include "HField3DUtils.h"
#include "Array3DUtils.h"
#include "ImageUtils.h"
%}

%include "HField3DUtils.h"

//
// This is necessary to correctly handle Array3DUtils::getMinMax
//
%apply float& OUTPUT { float &MIN_OUTPUT };
%apply float& OUTPUT { float &MAX_OUTPUT };

//
// Need a version of trilerp that returns a value
//

%extend HField3DUtils {
  static Vector3D<float> 
    trilerp(const Array3D<Vector3D<float> >&h,
	    Vector3D<float> in,
	    HField3DUtils::VectorBackgroundStrategy backgroundStrategy =
	    HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID )
  {
    Vector3D<float> out;
    HField3DUtils::trilerp(h, in.x, in.y, in.z, out.x, out.y, out.z, backgroundStrategy);
    return out;
  }
};

%include "Array3DUtils.h"

%include "ImageUtils.h"
