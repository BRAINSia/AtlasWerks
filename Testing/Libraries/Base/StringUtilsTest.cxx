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

// Test strPrintf
int runTests1(){

  // short test

  std::string str = "003 is the same as 3";
  std::string fmt = "%03d is the same as %d";
  std::string test = StringUtils::strPrintf(fmt.c_str(), 3, 3);
  if(str != test){
    std::cerr << str << " != " << test << std::endl;
    return TEST_FAIL;
  }

  // loooong test, force it to allocate from heap

  std::stringstream strstr;
  std::stringstream fmtstr;
  std::string sample = "Beer is living proof that God loves us and wants us to be happy.  ";
  sample = sample + sample;
  sample = sample + sample;
  sample = sample + sample;
  for(int i=0;i<4;i++){
    strstr << sample;
    fmtstr << "%s";
  }
  std::cout << "Creating a string " << 4*sample.length() << " characters long" << std::endl;
  test = StringUtils::strPrintf(fmtstr.str().c_str(), sample.c_str(), sample.c_str(), sample.c_str(), sample.c_str());
  if(strstr.str() != test){
    std::cerr << "Original string of length " << strstr.str().length() << " != new string of length " << test.length() << std::endl;
    return TEST_FAIL;
  }
 
  return TEST_PASS;
}

int main(int argc, char *argv[]){
  if(runTests1() != TEST_PASS){
    return -1;
  }
  return 0;
}
