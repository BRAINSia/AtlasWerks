%module Algorithms

%include "std_string.i"
%include "std_vector.i"
%include "std_complex.i"

%import "Base.i"
%import "DataTypes.i"
%import "UtilitiesDataTypes.i"

%{
#include "DiffOper.h" 
#include "GreedyDeformationData.h"
#include "GreedyIteratorCPU.h"
#include "GreedyWarp.h"
#include "XORParam.h"
#include "MultiscaleManager.h"
#ifdef CUDA_ENABLED
#include "GreedyIteratorGPU.h"
#include "GreedyDeformationDataGPU.h"
#endif
%}

%feature("autodoc","1");

typedef float Real;

%include "DiffOper.h"

%template(DiffOper) DiffOperT<float>;
%template(dDiffOper) DiffOperT<double>;

%include "MultiscaleManager.h"

%include "GreedyIteratorCPU.h"
%include "GreedyWarp.h"
%template(GreedyWarpCPU) GreedyWarp<GreedyIteratorCPU>;
