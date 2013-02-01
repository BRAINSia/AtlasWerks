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


#ifndef __PARAM_EXCEPTION_H__
#define __PARAM_EXCEPTION_H__

#include "AtlasWerksException.h"

class ParamException : public AtlasWerksException
{
public:
  ParamException(const char *file="unknown file", int line=0,
		 const std::string& text = "undefined exception",
		 const std::exception *cause = NULL) :
    AtlasWerksException(file, line, text, cause)
  {
    mTypeDescription = "ParamException";
  }
  
  virtual ~ParamException() throw() {}
};

#endif

