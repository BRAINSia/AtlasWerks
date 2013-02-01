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


#ifndef __ParamBase_H__
#define __ParamBase_H__

#ifndef SWIG

#include <vector>
#include <sstream>

#include "tinyxml.h"
#include "ParamException.h"
#include "ParamConstraint.h"
#include "StringUtils.h"
#include "log.h"

#endif

enum ParamLevel { PARAM_REQUIRED,
		  PARAM_COMMON,
		  PARAM_RARE,
		  PARAM_DEBUG };

class ParamConstraint;

/**
 * Simple name-value pair.  A vector of these is used when parsing command line arguments.
 */
class ParamNameValPair {
protected:
  std::string mName;
  std::string mVal;
public:
  ParamNameValPair(const std::string& name, const std::string& val)
    : mName(name),
      mVal(val)
  {}
  
  ParamNameValPair(const char *name, const char *val)
    : mName(name),
      mVal(val)
  {}
  
  ParamNameValPair()
    : mName(""),
      mVal("")
  {}
  
  std::string &Name(){ return mName; }
  const std::string &Name() const { return mName; }
  std::string &Val(){ return mVal; }
  const std::string &Val() const { return mVal; }
};

/**
 * Define a macro to enable easy subclassing of ParamBase
 */
#define CopyFunctionMacro(Class)        \
  virtual ParamBase *Copy() const {     \
    return new Class(*this);            \
  }                                                          

/**
 * Base type of all parameter classes.  Virtual, cannot be instantiated.
 */
class ParamBase {
protected:
  /**
   * Name of the parameter used to identify it in the parameter file
   */
  std::string mName;

  /**
   * Description of this argument, used for generating usage/help
   */
  std::string mDescription;

  /**
   * The importance level of this param, used to check if param is
   * required, too
   */
  ParamLevel mParamLevel;

  /**
   * Has this parameter been set?
   */
  bool mAlreadySet;

  /**
   * Should we ignore occurrences of this param?
   */
  bool mIgnore;

  /**
   * A constraint, checked after parsing
   */
  ParamConstraint *mConstraint;

  /**
   * A set of aliases for this parameter.  Calling MatchesName(...)
   * with a name that matches one of the aliases will return 'true'.
   * This is generally used for backwards-compatibility when renaming
   * parameters.
   */
  std::vector<std::string> mAlias;

  /**
   * This constructor should never be called by users.  It is only to
   * be called by subclasses.
   *
   * \param name - The name identifying the argument.  
   *
   * \param description - The description of the argument, used in the
   * usage.
   *
   * \param level - importance level of the parameter.  Set to
   * PARAM_REQUIRED for required parameters.
   */
  ParamBase(const std::string& name, 
	    const std::string& description, 
	    ParamLevel level,
	    ParamConstraint *constraint);

  void
  PrintFormatted(std::ostream& os,
		 const std::string &str, 
		 std::string indent="", 
		 unsigned int width=80) const;

public:
  
  virtual ~ParamBase();

  /**
   * Return the name used to identify this parameter
   */
  const std::string& GetName() const;
  
  /**
   * Does this name match our name? (allows case-insensitivity, etc)
   */
  virtual bool MatchesName(const char* nameToMatch) const;
  virtual bool MatchesName(const std::string &nameToMatch) const;

  /**
   * Return a description of this parameter
   */
  const std::string& GetDescription() const;
  
  /**
   * Is this parameter required to be set?
   */
  bool Required() const;

  /**
   * Return the parameter's level
   */
  ParamLevel Level() const;

  /**
   * Has this parameter been set?
   */
  bool IsSet() const;

  /**
   * Set this parameter as not having been set yet
   */
  virtual void Unset();

  /**
   * Should we ignore requests to parse this param?
   */
  bool &Ignore() { return mIgnore; }
  const bool &Ignore() const { return mIgnore; }

  /**
   * Add an alias for this parameter (will parse correctly using this
   * name in the XML file)
   */
  void AddAlias(std::string alias);
  void AddAlias(const char *alias);

  /**
   * Set the parameter constraint
   */
  void SetConstraint(ParamConstraint *constraint){
    mConstraint = constraint;
  }

  /**
   * Create a deep copy of the object
   */
  virtual ParamBase *Copy() const = 0;

  /**
   * Parse an XML node.  Will throw an exception on an
   * error. Internally calls DoParse() and check constraint
   */
  void Parse(const TiXmlElement *element);

  /**
   * Attempt to parse a set of name/value pairs.  Used for command line parsing.
   */
  void Parse(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec);

  /**
   * Virtual function to do actual parsing
   */
  virtual void DoParse(const TiXmlElement *element) = 0;

  /**
   * Virtual function to do actual parsing.  Subclasses are not
   * required to implement this, it's okay to throw an error saying
   * that the arg can't be parsed in this manner (this is the default
   * implementation).
   */
  virtual TiXmlElement *ParseToXML(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec) const;

  /**
   * Print this node
   */
  void Print(std::string indent, bool brief=false) const;

  virtual void Output(std::ostream& os, std::string indent, bool brief=false) const = 0;

  std::string GetXMLString() const;

  /**
   * Create an XML element for this node, and link it to the parent
   * node.  Only generate XML if the level of this node is <=
   * maxLevel.  Include parameter description in XML comments if
   * includeComments is true.
   */
  virtual void GenerateXML(TiXmlNode *parent, ParamLevel maxLevel, bool includeComments) const = 0;

};

// ########### End Class Definition ########### //
// ######## Begin Class Implementation ######## //

inline
ParamBase::
ParamBase(const std::string& name, 
	  const std::string& description, 
	  ParamLevel level,
	  ParamConstraint *constraint) :
  mName(name),
  mDescription(description),
  mParamLevel(level),
  mAlreadySet(false),
  mIgnore(false),
  mConstraint(constraint)
{}

inline ParamBase::~ParamBase(){}

inline const 
std::string& 
ParamBase::
GetName() const
{
  return mName;
}
  
inline const 
std::string& 
ParamBase::
GetDescription() const
{
  return mDescription;
}

inline 
bool 
ParamBase::
Required() const
{
  return (mParamLevel == PARAM_REQUIRED);
}

inline 
ParamLevel
ParamBase::
Level() const
{
  return mParamLevel;
}

inline 
bool 
ParamBase::
IsSet() const
{
  return mAlreadySet;
}

inline
void 
ParamBase::
Unset()
{
  mAlreadySet = false;
}

inline
void 
ParamBase::
AddAlias(const char* alias)
{
  this->AddAlias(std::string(alias));
}

inline
void 
ParamBase::
AddAlias(std::string alias)
{
  mAlias.push_back(alias);
}

inline
bool
ParamBase::
MatchesName(const std::string &nameToMatch) const
{
  return MatchesName(nameToMatch.c_str());
}
  
inline
bool
ParamBase::
MatchesName(const char* nameToMatch) const
{
  std::string myNameUpper = StringUtils::toUpper(GetName());
  std::string otherNameUpper = StringUtils::toUpper(std::string(nameToMatch));
  if(myNameUpper == otherNameUpper) return true;
  for(unsigned int i=0;i<mAlias.size();i++){
    myNameUpper = StringUtils::toUpper(mAlias[i]);
    if(myNameUpper == otherNameUpper) return true;
  }
  return false;
}

inline 
void
ParamBase::
Parse(const TiXmlElement *element)
{
  if(!mIgnore){
    DoParse(element);
    if(mConstraint){
      if(!mConstraint->Check(this)){
	throw ParamException(__FILE__, __LINE__,
			     GetName() + " violated constraint " + mConstraint->Description());
      }
    }
  }
}

inline 
void
ParamBase::
Parse(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec)
{
  TiXmlElement *element = ParseToXML(curIdx, nameValVec);
  Parse(element);
  delete element;
}

inline 
TiXmlElement *
ParamBase::
ParseToXML(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec) const
{
  throw ParamException(__FILE__, __LINE__,
		       GetName() + " does not support XML generation from name-value pairs");
  //return static_cast<TiXmlElement*>(NULL);
}

inline
std::string
ParamBase::
GetXMLString() const
{
  TiXmlDocument *doc = new TiXmlDocument();
  this->GenerateXML(doc, PARAM_RARE, true);
  std::stringstream ss;
  TiXmlNode *child;
  for( child = doc->FirstChild(); child; child = child->NextSibling() ){
    ss << (*child) << std::endl;
  }
  return ss.str();
}

inline
void
ParamBase::
Print(std::string indent, bool brief) const
{
  this->Output(std::cout, indent, brief);
}

inline
void
ParamBase::
PrintFormatted(std::ostream& os,
	       const std::string &str, 
	       std::string indent, 
	       unsigned int width) const
{
  std::stringstream output;
  size_t curpos = 0;
  size_t linelen = width-indent.size();
  while(curpos < str.size()){
    // add indention to the line
    output << indent;
    size_t breakpos = str.size();
    // if we have to split this over multiple lines
    if(str.size()-curpos > linelen){
      // try to split line at natural break
      breakpos = str.find_last_of(" \t|-",curpos+linelen);
      if(str[breakpos] == '-' || str[breakpos] == '|') breakpos++;
      if(breakpos == std::string::npos){
	// if we can't, just break at maximum position
	breakpos = curpos+linelen;
      }
    }
    //std::cout << "curpos = " << curpos << ", breakpos = " << breakpos << std::endl;
    output << str.substr(curpos,breakpos-curpos) << std::endl;
    // add extra indention for lines after the first
    if(curpos == 0){
      indent += "  ";
      linelen -= 2;
    }
    curpos += breakpos-curpos;
  }
  os << output.str();
}


#endif // __ParamBase_H__















