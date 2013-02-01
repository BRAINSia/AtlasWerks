%module Base

%include "std_string.i"
%include "std_complex.i"

%import "DataTypes.i"

%{

#include "AtlasWerksTypes.h"

#include "ApplicationUtils.h"

#include "ParamBase.h"
#include "ValueParam.h"
#include "CompoundParam.h"
#include "MultiParam.h"
#include "XORParam.h"

#include "WeightedImageSet.h"

%}

%feature("autodoc","1");

%exception LoadImageITK<float>{
  try {
    $action
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_IOError, const_cast<char*>(e.what()));
    return NULL;
  }
}

%exception SaveImageITK<float>{
  try {
    $action
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_IOError, const_cast<char*>(e.what()));
    return NULL;
  }
}

%exception LoadHFieldITK<float>{
  try {
    $action
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_IOError, const_cast<char*>(e.what()));
    return NULL;
  }
}

%exception SaveHFieldITK<float>{
  try {
    $action
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_IOError, const_cast<char*>(e.what()));
    return NULL;
  }
}

%include "AtlasWerksTypes.h"

%include "ApplicationUtils.h"

%define ParamPrintExtension(type)
%extend type {
  char *__str__() {
    static char *tmp = NULL;
    static unsigned long tmpSz = 0;
    std::string indent = "";
    std::stringstream os;
    $self->Output(os, indent);
    if(os.str().size() > tmpSz){
      if(tmp){
        delete [] tmp;
      }
      tmpSz = os.str().size();
      tmp = new char[tmpSz+1];
    }
    strncpy(tmp,os.str().c_str(),tmpSz);
    return tmp;
  }
};
%enddef

%include "ParamBase.h"
%include "ValueParam.h"
ParamPrintExtension(ValueParam)

%include "CompoundParam.h"
ParamPrintExtension(CompoundParam)

%include "MultiParam.h"
ParamPrintExtension(MultiParam)

%include "XORParam.h"
ParamPrintExtension(XORParam)

%include "WeightedImageSet.h"
