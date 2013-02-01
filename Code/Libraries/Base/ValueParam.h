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


#ifndef __VALUE_PARAM_H__
#define __VALUE_PARAM_H__

#ifndef SWIG

#include "ParamBase.h"
#include "ParamException.h"
#include <sstream>

#endif // !SWIG


/**
 * This class is used to parse a parameter value.  It is used because
 * we need a special implementation to deal with std::string (spaces
 * cause problems with << operator). Will throw an exception if there
 * is a problem parsing the parameter.
 */
template<class T> class ParamParser 
{
public:
  static
  T parse( const std::string& val ) 
  {
    
    std::istringstream is(val);
    
    T value;
    
    int valuesRead = 0;
    while ( is.good() ) 
      {
	if ( is.peek() != EOF )
	  is >> std::boolalpha >> value;
	else
	  break;
	
	valuesRead++;
      }
    
    if ( is.fail() ) 
      throw( ParamException(__FILE__, __LINE__,
			    "Error parsing value from: " + val));
	     
    if ( valuesRead > 1 )
      throw( ParamException(__FILE__, __LINE__,
			    "Multiple values parsed from: " + val));
    
    return value;
  }

  static
  std::string toString(const T& val){
    std::ostringstream os;
    os << std::boolalpha << val;
    return os.str();
  }
};

/**
 * Specialization for string.  This is necessary because istringstream
 * operator>> is not able to ignore spaces.
 */
template<> class ParamParser<std::string>
{
public:
  static
  std::string parse( const std::string& val ) 
  {
    std::string value = val;
    return value;
  }

  static
  std::string toString(const std::string& val){
    std::string str = val;
    return str;
  }
};

/**
 * Specialization for bool, we want to be able to parse "True", "true", TRUE, 1, etc.
 */
template<> class ParamParser<bool>
{
public:
  static
  bool parse( const std::string& val ) 
  {
    return StringUtils::toBool(val);
  }

  static
  std::string toString(const bool& val){
    std::ostringstream os;
    os << std::boolalpha << val;
    return os.str();
  }
};

template<class T>
class ValueParam : public ParamBase
{
  
public:
  
  ValueParam( const std::string& name, 
	      const std::string& desc, 
	      ParamLevel level, 
	      T defaultVal,
	      ParamConstraint *constraint = NULL);

  virtual T& Value();
  virtual const T& Value() const;
  
  /**
   * Create an XML node from a name-value pair, used for parsing.
   */
  virtual TiXmlElement *ParseToXML(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec) const;


  /**
   * Parse a name-value pair.  Used for command line parsing.
   */
  virtual void DoParse(const TiXmlElement *element);

  CopyFunctionMacro(ValueParam)

  /**
   * Print this node
   */
  virtual void Output(std::ostream& os, std::string indent, bool brief=false) const;

  virtual void GenerateXML(TiXmlNode *parent, ParamLevel maxLevel, bool includeComments) const;

protected:

  /** base-type value for this parameter*/
  T mValue;

  /**
   * Extracts the value from the string.
   * Attempts to parse string as type T, if this fails an exception
   * is thrown.
   * \param val - value to be parsed. 
   */
  virtual void parseValue( const std::string& val );

};

// ########### End Class Definition ########### //
// ######## Begin Class Implementation ######## //

#ifndef SWIG

template<class T>
ValueParam<T>::
ValueParam( const std::string& name, 
	    const std::string& desc, 
	    ParamLevel level, 
	    T defaultVal,
	    ParamConstraint *constraint)
  : ParamBase(name, desc, level, constraint),
    mValue(defaultVal)
{
}

template<class T>
T& 
ValueParam<T>::
Value() 
{ return mValue; }

template<class T>
const T& 
ValueParam<T>::
Value() const 
{ return mValue; }

template<class T>
TiXmlElement *
ValueParam<T>::
ParseToXML(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec) const
{
  const ParamNameValPair &curPair = nameValVec[curIdx];
  TiXmlElement *element = new TiXmlElement(curPair.Name());
  element->SetAttribute("val", curPair.Val());

  // update curIdx
  curIdx++;

  return element;
}

template<class T>
void 
ValueParam<T>::
DoParse(const TiXmlElement *element)
{
  if(this->IsSet()){
    throw( ParamException(__FILE__, __LINE__,
			  "Value node " + this->GetName() + " is being set twice."));
  }

  LOGNODE(logDEBUG3) << "Parsing ValueParam " << GetName();

  // this is a value node, we shouldn't have any children
  if(element->FirstChildElement() != NULL){
    throw( ParamException(__FILE__, __LINE__,
			  "Value node " + this->GetName() + " has child elements"));
  }
  
  // this will hold the text
  std::string v;

  // look for the 'val' attribute
  bool hasValAttr = false;
  const TiXmlAttribute *attr = element->FirstAttribute();
  if(attr && strcmp(attr->Name(),"val") == 0){
    v = attr->ValueStr();
    hasValAttr = true;
  }

  // look for text
  const char *text = element->GetText();

  // test that we only have text from one or the other
  if(hasValAttr && text){
    throw( ParamException(__FILE__, __LINE__, 
			  "Value node " + this->GetName() + " has both 'val' attribute and encloses text"));
  }

  // test that we got something from at least one
  if(!hasValAttr && !text){
    throw( ParamException(__FILE__, __LINE__, 
			  "Value node " + this->GetName() + " has neither 'val' attribute nor encloses text"));
  }

  if(text != NULL) v = text;

  this->parseValue(v);

}

template<class T>
void 
ValueParam<T>::
parseValue( const std::string& val )
{
  mValue = ParamParser<T>::parse(val);
  mAlreadySet = true;
}

template<class T>
void
ValueParam<T>::
Output(std::ostream& os, std::string indent, bool brief) const {
  std::stringstream out;
  out << GetName();
  if(Required()) out << " (required) ";
  if(IsSet()) out << " (set) ";
  out << " = " << mValue << " ";
  if(!brief){
    out << GetDescription();
  }
  this->PrintFormatted(os, out.str(), indent);
}

template<class T>
void
ValueParam<T>::
GenerateXML(TiXmlNode *parent, ParamLevel maxLevel, bool includeComments) const
{
 
  if(this->Level() > maxLevel){
    return;
  }

 if(includeComments){
    // add the comment
    TiXmlComment *comment = new TiXmlComment();
    comment->SetValue(GetDescription());
    parent->LinkEndChild(comment);
  }

  // add the element
  TiXmlElement *element = new TiXmlElement(this->GetName());
  std::string str = ParamParser<T>::toString(Value());
  element->SetAttribute("val", str);
  parent->LinkEndChild(element);
}

#endif // !SWIG

#endif // __VALUE_PARAM_H__
