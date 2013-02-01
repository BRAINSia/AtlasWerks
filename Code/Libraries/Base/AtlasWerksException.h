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


#ifndef __ATLASWERKS_EXCEPTION_H__
#define __ATLASWERKS_EXCEPTION_H__

#ifndef SWIG

#include <string>
#include <exception>

#endif // SWIG

#include "StringUtils.h"

/**
 * Class representing an error thrown by the AtlasWerks libraries
 */
class AtlasWerksException : public std::exception
{
public:
  AtlasWerksException(const char *file="unknown file", int line=0,
		   const std::string& text = "undefined exception",
		   const std::exception *cause = NULL) :
    mTypeDescription("AtlasWerksException")
  {
    mText = text;
    if(cause){
      mText = mText + "\n   Caused by:\n" + cause->what();
    }
    mLocation = StringUtils::strPrintf("%s:%d", file, line);
  }

  virtual ~AtlasWerksException() throw() {}

  std::string Text() const { return mText; }
  std::string Location() const { return mLocation; }
  std::string TypeDescription() const { return mTypeDescription; }

  virtual const char* what() const throw (){
    static std::string ex; 
    ex = TypeDescription() + " : From " + Location() + " : " + Text();
    return ex.c_str();
  }
  
protected:

  std::string mText;
  std::string mLocation;
  std::string mTypeDescription;

};

#endif
