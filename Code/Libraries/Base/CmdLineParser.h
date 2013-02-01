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

#ifndef __CMD_LINE_PARSER_H__
#define __CMD_LINE_PARSER_H__

#include <cstdlib>
#include "ParamBase.h"
#include "ValueParam.h"
#include "MultiParam.h"
#include "CompoundParam.h"
#include "XORParam.h"

/**
 * Class to parse a ParamBase structure from a command line.  For
 * applications that use this to parse their command lines, running:
 *
 * > MyApp -h
 *
 * will give a full set of options.  To generate a sample parameter
 * file, run:
 *
 * > MyApp -g test.xml
 *
 * To run the application with a parameter file, use:
 *
 * > MyApp -f param.xml
 *
 * The available parameters with descriptions can be printed to the
 * screen with:
 *
 * > MyApp -u
 * 
 * In addition, individual top-level parameters can be set by
 * specifying them on the command line:
 *
 * > MyApp -f param.xml --ExtraParam extraVal --ExtraParam2 extraVal2
 *
 * Parameters specified on the command line will override settings
 * from the parameter file.
 */
class CmdLineParser {
public:

  /**
   * Create a command line parser that will parse the given parameter
   * file
   * \param topLevel the parameter to be parsed
   */
  CmdLineParser(ParamBase &topLevel);
  
  /**
   * Parse the given command line, reading parameter file and parsing
   * args as necessary
   */
  void Parse(unsigned int argc, char **argv);

  /**
   * Generate a sample parameter file from the mRootParam parameter
   */
  void GenerateFile(const char *fname);

private:
  
  std::string mProgName;
  
  ParamBase &mRootParam;

};

/**
 * Function for quickly extracting the value of a child ValueParam from a CompoundParam
 */
template <typename TChildVal>
TChildVal GetChildVal(const CompoundParam *parent, const char *childName){
  const ParamBase* base = parent->GetChild(childName);
  if(!base){
    std::string err = StringUtils::strPrintf("Error, no child named %s in CompoundParam %s", 
					     childName, parent->GetName().c_str());
    throw(ParamException(__FILE__, __LINE__, err.c_str()));
  }
  const ValueParam<TChildVal>* param =
    dynamic_cast<const ValueParam<TChildVal>* >(base);
  if(!param){
    std::string err = StringUtils::strPrintf("Error, cannot cast %s to correct ValueParam type", 
					     childName);
    throw(ParamException(__FILE__, __LINE__, err.c_str()));
  }
  return param->Value();
}

#endif // __CMD_LINE_PARSER_H__
