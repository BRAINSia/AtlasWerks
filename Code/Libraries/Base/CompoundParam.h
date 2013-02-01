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


#ifndef __COMPOUND_PARAM_H__
#define __COMPOUND_PARAM_H__

#include "ParamBase.h"
#include "ParamException.h"

#ifndef SWIG

#include <map>

#endif

/**
 * Macros to make easy accessor functions for children.  Extra
 * checking is done to make sure something silly wasn't done -- this
 * encurs some overhead, but we aren't too concerned with performance
 * for parameter parsing, and we'd rather not have unexpected and
 * nasty bugs.  Throws exception on error
 *
 * \param ParamType the full type of the parameter to be accessed
 *
 * \param ParamName the string name of the child parameter (not in
 * quotes) -- will also be the name of the accessor function
 */
#define ParamAccessorMacro(PType, PName)                              \
  PType &PName(){						      \
    ParamBase *baseParam = GetChild(#PName);                          \
    if(!baseParam){						      \
      throw(                                                          \
        ParamException(                                               \
          __FILE__, __LINE__,                                         \
          "ParamAccessorException : No Child Named " #PName " -- "    \
          #PType "::" #PName "() (Macro from CompoundParam.h)"));     \
    }                                                                 \
    PType *param = dynamic_cast<PType*>(baseParam);                   \
    if(!param){							      \
      throw(                                                          \
        ParamException(                                               \
          __FILE__, __LINE__,                                         \
          "ParamAccessorException : Cannot cast "                     \
          #PName " to " #PType "* --"                                 \
          #PType "::" #PName "() (Macro from CompoundParam.h)"));     \
    }                                                                 \
    return *param;                                                    \
  }                                                                   \
  const PType &PName() const {                                        \
    const ParamBase *baseParam = GetChild(#PName);                    \
    if(!baseParam){						      \
      throw(                                                          \
        ParamException(                                               \
          __FILE__, __LINE__,                                         \
          "ParamAccessorException : No Child Named " #PName " -- "    \
          #PType "::" #PName "() (Macro from CompoundParam.h)"));     \
    }                                                                 \
    const PType *param = dynamic_cast<const PType*>(baseParam);	      \
    if(!param){							      \
      throw(                                                          \
        ParamException(                                               \
          __FILE__, __LINE__,                                         \
          "ParamAccessorException : Cannot cast "                     \
          #PName " to " #PType "* -- "                                \
          #PType "::" #PName "() (Macro from CompoundParam.h)"));     \
    }                                                                 \
    return *param;                                                    \
  }                                                                   \
  PType Get##PName(){						      \
    PType rtn = this->PName();                                        \
    return rtn;                                                       \
  }                                                                   \
  void Set##PName(const PType &param){                                \
    this->PName() = param;                                            \
  }                                                                   

#define ValueParamAccessorMacro(ValueType, PName)                     \
  ValueType &PName(){						      \
    ParamBase *baseParam = GetChild(#PName);                          \
    if(!baseParam){						      \
      throw(                                                          \
        ParamException(                                               \
          __FILE__, __LINE__,                                         \
          "ValueParamAccessorException : No Child Named " #PName      \
          " -- ValueParam<" #ValueType ">::" #PName                   \
          "() (Macro from CompoundParam.h)"));                        \
    }                                                                 \
    ValueParam<ValueType> *param =                                    \
      dynamic_cast<ValueParam<ValueType>*>(baseParam);                \
    if(!param){							      \
      throw(                                                          \
        ParamException(                                               \
          __FILE__, __LINE__,                                         \
          "ValueParamAccessorException : Cannot cast "                \
          #PName " to ValueParam<" #ValueType "> * -- "               \
          "ValueParam<" #ValueType ">::" #PName                       \
          "() (Macro from CompoundParam.h)"));                        \
    }                                                                 \
    return param->Value();                                            \
  }                                                                   \
  const ValueType &PName() const {				      \
    const ParamBase *baseParam = GetChild(#PName);                    \
    if(!baseParam){						      \
      throw(                                                          \
        ParamException(                                               \
          __FILE__, __LINE__,                                         \
          "ValueParamAccessorException : No Child Named " #PName      \
          " -- ValueParam<" #ValueType ">::" #PName                   \
          "() (Macro from CompoundParam.h)"));                        \
    }                                                                 \
    const ValueParam<ValueType> *param =                              \
      dynamic_cast<const ValueParam<ValueType>*>(baseParam);	      \
    if(!param){							      \
      throw(                                                          \
        ParamException(                                               \
          __FILE__, __LINE__,                                         \
          "ValueParamAccessorException : Cannot cast "                \
          #PName " to ValueParam<" #ValueType "> * -- "               \
          "ValueParam<" #ValueType ">::" #PName                       \
          "() (Macro from CompoundParam.h)"));                        \
    }                                                                 \
    return param->Value();                                            \
  }                                                                   \
  ValueType Get##PName(){					      \
    ValueType rtn = this->PName();                                    \
    return rtn;                                                       \
  }                                                                   \
  void Set##PName(const ValueType &param){			      \
    this->PName() = param;                                            \
  }                                                                   

/**
 * CompoundParam represents a parameter with child parameters
 */
class CompoundParam : public ParamBase {

protected:
  typedef std::vector<ParamBase*> TChildVec;

public:
  /** needs copy constructor to copy children */
  CompoundParam(const CompoundParam &other);

  CompoundParam(const std::string& name, 
		const std::string& desc, 
		ParamLevel level,
		ParamConstraint *constraint = NULL);

  virtual ~CompoundParam();

  /**
   * This method links in the given param as a child of this
   * parameter.  This param will be automatically deleted when this
   * (parent) object is deleted.
   */
  void AddChild(ParamBase *param);

  /** 
   * This method adds a copy of param as a child of this parameter.
   * Returns a pointer to the copy.
   */
  ParamBase *AddChild(const ParamBase &param);

  /**
   * Return a pointer to the child with the given name.  Returns NULL
   * if no such child exists.
   */
  ParamBase *GetChild(const std::string &name);

  /**
   * Return a pointer to the child with the given name.  Returns NULL
   * if no such child exists.
   */
  const ParamBase *GetChild(const std::string &name) const;

  /**
   * Are all the required children of this param set?  Returns pointer
   * to unset required param if one is found
   */
  bool RequiredChildrenSet(const ParamBase*& unsetRequiredParam) const;

  /**
   * Overridden to unset children too
   */
  virtual void Unset();

  /**
   * Convert series of name-value pairs to parsable XML elements
   */
  virtual TiXmlElement *ParseToXML(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec) const;

  virtual void DoParse(const TiXmlElement *element);

  CopyFunctionMacro(CompoundParam)
  
  /**
   * Write string representation of this node (and children) to `os'
   */
  virtual void Output(std::ostream& os, std::string indent, bool brief=false) const;

  /**
   * assignment operator
   */
  CompoundParam &operator=(const CompoundParam &other);

  /**
   * Create a tinyxml structure representing this node and it's children
   */
  virtual void GenerateXML(TiXmlNode *parent, ParamLevel maxLevel, bool includeComments) const;

protected:
  /** list of child parameters */
  TChildVec mChildParams;

};

// ########### End Class Definition ########### //
// ######## Begin Class Implementation ######## //

#ifndef SWIG

// Copy Constructor
inline
CompoundParam::
CompoundParam(const CompoundParam &other)
  : ParamBase(other)
{
  TChildVec::const_iterator it;
  for(it = other.mChildParams.begin(); it != other.mChildParams.end(); ++it){
    this->mChildParams.push_back((*it)->Copy());
  }
}

inline
CompoundParam::
CompoundParam(const std::string& name, 
	      const std::string& desc, 
	      ParamLevel level,
	      ParamConstraint *constraint)
  : ParamBase(name, desc, level, constraint)
{}

inline
CompoundParam::
~CompoundParam()
{
  TChildVec::iterator it;
  for(it = mChildParams.begin(); it != mChildParams.end(); ++it){
    delete *it;
  }
}

inline
void
CompoundParam::
AddChild(ParamBase *param)
{
  const std::string &name = param->GetName();
  if(GetChild(name) != NULL){
    throw( ParamException(__FILE__, __LINE__,
			  "Attempted to add child param with conflicting name :" + name));
  }
  mChildParams.push_back(param);
}

inline
ParamBase *
CompoundParam::
AddChild(const ParamBase &param)
{
  ParamBase *copy = param.Copy();
  this->AddChild(copy);
  return copy;
}

inline
ParamBase *
CompoundParam::
GetChild(const std::string &name)
{
  TChildVec::iterator it = mChildParams.begin();
  for(;it != mChildParams.end(); ++it){
    if((*it)->MatchesName(name)){
      return *it;
    }
  }
  return NULL;
}

inline
const ParamBase *
CompoundParam::
GetChild(const std::string &name) const
{
  TChildVec::const_iterator it = mChildParams.begin();
  for(;it != mChildParams.end(); ++it){
    if((*it)->MatchesName(name)){
      return *it;
    }
  }
  return NULL;
}

inline
bool
CompoundParam::
RequiredChildrenSet(const ParamBase*& unsetRequiredParam) const
{
  TChildVec::const_iterator it;
  for(it = mChildParams.begin(); it != mChildParams.end(); ++it){
    if((*it)->Required() && !(*it)->IsSet()){
      unsetRequiredParam = *it;
      return false;
    }
  }
  return true;
}

inline
CompoundParam &
CompoundParam::
operator=(const CompoundParam &other)
{
  if(this != &other){
    this->ParamBase::operator=(other);

    // delete any current children
    TChildVec::iterator it;
    for(it = mChildParams.begin(); it != mChildParams.end(); ++it){
      delete *it;
    }
    mChildParams.clear();

    // deep copy children
    TChildVec::const_iterator c_it;
    for(c_it = other.mChildParams.begin(); c_it != other.mChildParams.end(); ++c_it){
      this->mChildParams.push_back((*c_it)->Copy());
    }
  }
  
  return *this;
}

inline
TiXmlElement *
CompoundParam::
ParseToXML(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec) const
{
  const ParamNameValPair *curPair = &nameValVec[curIdx];

  // this is just a container so there should be no value
  if(curPair->Val().size() != 0){
    throw( ParamException(__FILE__, __LINE__, 
			  "CompoundParam " + this->GetName() + " has value " + curPair->Val() 
			  + ", when it is a container and should be empty"));
  }

  curIdx++;

  TiXmlElement *element = new TiXmlElement(GetName());

  while(curIdx < nameValVec.size()){
    curPair = &nameValVec[curIdx];
    const ParamBase *child = GetChild(curPair->Name());
    // stop parsing when we get a name that isn't a child name
    if(!child) break;
    TiXmlElement *elChild = child->ParseToXML(curIdx, nameValVec);
    element->LinkEndChild(elChild);
  }

  return element;
}

inline
void 
CompoundParam::
DoParse(const TiXmlElement *element)
{
  if(this->IsSet()){
    throw( ParamException(__FILE__, __LINE__,
			  "Compound node " + this->GetName() + " is being set twice."));
  }

  // get child elements

  LOGNODE(logDEBUG3) << "Parsing CompoundParam " << GetName();

  const TiXmlNode *child = element->FirstChild();

  while(child){
    int t = child->Type();
    switch(t){
    case TiXmlNode::ELEMENT:
      {
	const TiXmlElement *childElement = child->ToElement();
	if(childElement){
	  ParamBase *childParam = this->GetChild(childElement->ValueStr());
	  if(childParam){
	    // have the child parameter parse the child element
	    childParam->Parse(childElement);
	  }else{
	    throw( ParamException(__FILE__, __LINE__,
				  "CompoundParam " + this->GetName() + " has no child named " + childElement->ValueStr()));
	  }
	}
      }
      break;
    case TiXmlNode::TEXT:
      throw( ParamException(__FILE__, __LINE__,
			    "CompoundParam " + this->GetName() 
			    + " should contain only children, but has text contents " 
			    + child->ToText()->ValueStr()));
    }
    child = child->NextSibling();

    mAlreadySet = true;
  }

  // now get the attributes

  const TiXmlAttribute *attr = element->FirstAttribute();
  while(attr){
    const char *attrName = attr->Name();
    ParamBase *childParam = this->GetChild(attrName);
    if(childParam){
      // create a simple element node for this attribute so that a ValueParam can parse it
      TiXmlElement tmp(attrName);
      tmp.SetAttribute("val",attr->Value());
      
      // have the child parameter parse the child element
      childParam->Parse(&tmp);
    }else{
      throw( ParamException(__FILE__, __LINE__,
			    "CompoundParam " + this->GetName() + " has no child named " + 
			    attrName + " (found as attribute of XML node)"));
    }
    attr = attr->Next();
  }

  // test that we set all the required children
  const ParamBase* unsetParam = NULL;
  if(!this->RequiredChildrenSet(unsetParam)){
    throw( ParamException(__FILE__, __LINE__,
			  "CompoundParam " + this->GetName() + ": Required Child " + unsetParam->GetName() + " was not set"));
  }
  
}

inline
void 
CompoundParam::
Unset()
{
  ParamBase::Unset();
  
  mAlreadySet = false;
  TChildVec::iterator it;
  for(it = mChildParams.begin(); it != mChildParams.end(); ++it){
    (*it)->Unset();
  }
}

inline
void 
CompoundParam::
Output(std::ostream& os, std::string indent, bool brief) const
{

  std::stringstream out;
  out << GetName();
  if(Required()) out << " (required) ";
  if(IsSet()) out << " (set) ";
  if(!brief){
    out << ": " << GetDescription();
    this->PrintFormatted(os, out.str(), indent);
  }
  
  indent = indent + "   ";

  TChildVec::const_iterator it;
  for(it = mChildParams.begin(); it != mChildParams.end(); ++it){
    (*it)->Output(os, indent, brief);
  }
}

inline
void
CompoundParam::
GenerateXML(TiXmlNode *parent, ParamLevel maxLevel, bool includeComments) const
{

  if(this->Level() > maxLevel){
    return;
  }

  if(includeComments){
    TiXmlComment *comment = new TiXmlComment();
    comment->SetValue(GetDescription());
    parent->LinkEndChild(comment);
  }
  TiXmlElement *element = new TiXmlElement(this->GetName());
  TChildVec::const_iterator it;
  for(it = mChildParams.begin(); it != mChildParams.end(); ++it){
    (*it)->GenerateXML(element, maxLevel, includeComments);
  }
  parent->LinkEndChild(element);
}

#endif // SWIG

#endif // __COMPOUND_PARAM_H__
