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

#include "CmdLineParser.h"
#include "CompoundParam.h"
#include "ApplicationUtils.h"
#include "log.h"

#include <tclap/CmdLine.h>

#include <sstream>

CmdLineParser::
CmdLineParser(ParamBase &topLevel)
  : mRootParam(topLevel)
{
}

void 
CmdLineParser::
Parse(unsigned int argc, char **argv)
{

  mProgName = argv[0];
  try
    {
      
      TCLAP::CmdLine cmd(mProgName);
      
      TCLAP::SwitchArg
	usageSwitch("u","usage",
		    "Print out the usage, ie all XML elements",
		    cmd, false);
    
      TCLAP::ValueArg<std::string>
	xmlFileArg("f","paramFile",
		   "XML file to parse args from",
		   false,"","XML file", cmd);
    
      TCLAP::ValueArg<std::string>
	genFileArg("g","generate",
		   "Generate a sample XML file",
		   false,"","XML file", cmd);

      TCLAP::ValueArg<std::string>
	outFileArg("o","parsedParamFile",
		   "XML file to write parsed args to",
		   false,"","XML file", cmd);
    
      // get the rest of the arguments
      TCLAP::UnlabeledMultiArg<std::string>
	xmlElements("elements", "program-specific XML elements specified on the command line", false, "--XMLElement [value]", cmd);

      cmd.parse(argc, argv);

      if(usageSwitch.isSet()){
	mRootParam.Print("");
	std::exit(0);
      }else if(genFileArg.isSet()){
	LOG(logINFO) << "Generating sample config file...";
	GenerateFile(genFileArg.getValue().c_str());
	std::exit(0);
      }

      // setup the argsToParse from the UnlabeledMultiArg
      std::vector<ParamNameValPair> argsToParse;
      std::vector<std::string> cmdLineElements = xmlElements.getValue();
      unsigned int curElIdx = 0;
      while(curElIdx < cmdLineElements.size()){
	std::string& curEl = cmdLineElements[curElIdx];
	std::string elementName = curEl.substr(2,curEl.size()-2);
	ParamNameValPair curPair(elementName, "");
	curElIdx++;
	if(curElIdx < cmdLineElements.size()){
	  std::string& curVal = cmdLineElements[curElIdx];
	  if(curVal.substr(0,2) != "--"){
	    curPair.Val() = curVal;
	    curElIdx++;
	  }
	}
	LOGNODE(logDEBUG1) << "Parsing command line element " << curPair.Name() << " = " << curPair.Val();
	argsToParse.push_back(curPair);
      }
    
      //
      // Parse XML elements specified on command line
      //
      unsigned int curArg = 0;
      // we have to wait until the end to mark parsed args as 'Ignore()'
      // because of MultiParam args (otherwise only the first value would
      // be parsed)
      std::vector<ParamBase*> toIgnore;
      while(curArg < argsToParse.size()){
	// if this is a compound param, parse children
	CompoundParam *root = dynamic_cast<CompoundParam*>(&mRootParam);
	if(root){
	  ParamNameValPair *curPair = &argsToParse[curArg];
	  ParamBase *child = root->GetChild(curPair->Name());
	  if(!child){
	    throw( ParamException(__FILE__, __LINE__, 
				  "Cannot parse command line pair " + curPair->Name() + ":" + curPair->Val()));
	  }
	  // this may throw a parameter exception
	  child->Parse(curArg, argsToParse);
	  // now this node will not be parsed by the param file
	  toIgnore.push_back(child);
	}else{
	  LOGNODE(logERROR) << "Error, top-level param is not a compound param, but parsable elements specified on command line";
	}
      }
      // tell args to ignore values that might be specified in the XML file
      for(unsigned int i=0;i<toIgnore.size();i++){
	toIgnore[i]->Ignore() = true;
      }

      //
      // Parse the XML file
      //
      if(xmlFileArg.isSet()){
	std::string paramFileName = xmlFileArg.getValue();
	LOGNODE(logDEBUG) << "Parsing parameter file " << paramFileName;
	TiXmlDocument doc(paramFileName);
	doc.LoadFile();
	if(doc.Error()){
	  throw ParamException(__FILE__, __LINE__, "An error occurred parsing file " + paramFileName + ": " + doc.ErrorDesc());
	}
	// this may throw a ParameterException
	mRootParam.Parse(doc.RootElement());
        LOGNODE(logDEBUG) << "finished parsing file";
      }else{
	// we still need to check that all required arguments were set
	CompoundParam *root = dynamic_cast<CompoundParam*>(&mRootParam);
	if(root){
	  const ParamBase *unsetRequiredParam = NULL;
	  if(!root->RequiredChildrenSet(unsetRequiredParam)){
	    std::string unsetParamName = "UNKNOWN PARAM";
	    if(unsetRequiredParam){
	      unsetParamName = unsetRequiredParam->GetName();
	    }
	    throw( ParamException(__FILE__, __LINE__, 
				  "Required parameter " + unsetParamName + " not set!"));
	  }
	  LOGNODE(logDEBUG) << "Passed required parameter check";
	}else{
	  LOGNODE(logERROR) << "Error, top-level param is not a compound param, but parsable elements specified on command line";
	}
      }

      //
      // Write out the parsed params to a file
      //
      std::string parsedXMLFileName;
      if(outFileArg.isSet()){
	parsedXMLFileName = outFileArg.getValue();
      }else{
	// parse a pathname
	std::string path, nameBase, nameExt;
	ApplicationUtils::SplitName(mProgName.c_str(), path, nameBase, nameExt);
	parsedXMLFileName = nameBase + "ParsedOutput.xml";
      }

      LOGNODE(logDEBUG) << "Writing parsed parameters to " << parsedXMLFileName;
      GenerateFile(parsedXMLFileName.c_str());
    }
  catch (TCLAP::ArgException &e)
    {
      std::stringstream ss;
      ss << "TCLAP parameter exception: " << e.error() << " caused by parameter " << e.argId();
      throw ParamException(__FILE__, __LINE__, ss.str());
    }
}

void
CmdLineParser::
GenerateFile(const char *fname)
{
  
  TiXmlDocument *doc = new TiXmlDocument();
  mRootParam.GenerateXML(doc, PARAM_RARE, true);
  doc->SaveFile(fname);
  
}
