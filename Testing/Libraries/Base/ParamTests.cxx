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

#include <iostream>
#include <sstream>

#include "TestUtils.h"
#include "ValueParam.h"
#include "CompoundParam.h"
#include "MultiParam.h"
#include "tinyxml.h"

int runTests1(const char *testingDataDir){

  std::istringstream is;

  char buff[1024];
  sprintf(buff, "%s/%s/XMLParseTest.xml", 
	  testingDataDir,
	  "Tests/Libraries/Base");
  std::cout << "Reading XML file: " << buff << std::endl;
  
  CompoundParam topLevel("parameters", "Top-level node containing all parameters", PARAM_REQUIRED);
  ValueParam<bool> boolParam("SomeBoolVal", "A boolean value", PARAM_COMMON, false);
  topLevel.AddChild(boolParam);
  CompoundParam compoundParam("SimpleCompoundParam", "A single compound parameter", PARAM_REQUIRED);
  compoundParam.AddChild(ValueParam<float>("alpha", "some alpha value", PARAM_COMMON, -1.0));
  compoundParam.AddChild(ValueParam<float>("beta", "some beta value", PARAM_COMMON, -1.0));
  compoundParam.AddChild(ValueParam<float>("gamma", "some gamma value", PARAM_COMMON, -1.0));
  compoundParam.AddChild(ValueParam<float>("omega", "some omega value", PARAM_COMMON, -1.0));
  topLevel.AddChild(compoundParam);
  MultiParam<CompoundParam> multiParam(CompoundParam("MultiCompoundParam", "A muli-valued compound parameter", PARAM_COMMON));
  multiParam.GetTemplateParam().AddChild(ValueParam<float>("alpha", "some alpha value", PARAM_COMMON, -1.0));
  multiParam.GetTemplateParam().AddChild(ValueParam<float>("beta", "some beta value", PARAM_COMMON, -1.0));
  topLevel.AddChild(multiParam);

  TiXmlDocument doc(buff);
  doc.LoadFile();

  std::cout << "Root element name: " << doc.RootElement()->Value() << std::endl;
  
  try{
    topLevel.Parse(doc.RootElement());
    topLevel.Print("");
  }catch(ParamException e){
    std::cout << e.what() << std::endl;
  }

  std::cout << "XML Document: " << std::endl;
  doc.Print();
  std::cout << std::endl;

  


  return TEST_PASS;
}

int runTests2(){
  // set up the stringstream
  std::stringstream sstr;
  sstr << "<SingleFloatArg>12.34</SingleFloatArg>\n";
  sstr << "<SingleIntArg>56</SingleIntArg>\n";
  
  ValueParam<float> floatParam("SingleFloatArg", "A single float parameter", PARAM_COMMON, 0.0);
  ValueParam<int> intParam("SingleIntArg", "A single integer parameter", PARAM_COMMON, 0);

  return TEST_PASS;
}


int main(int argc, char *argv[]){
  if(runTests1(argv[1]) != TEST_PASS){
    return -1;
  }
  return 0;
}
