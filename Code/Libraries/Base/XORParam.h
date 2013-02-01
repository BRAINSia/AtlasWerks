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


#ifndef __XOR_PARAM_H__
#define __XOR_PARAM_H__

#include "CompoundParam.h"

class XORParam : public CompoundParam {
  
public:
  
  XORParam(const XORParam &other);
  
  XORParam(const std::string& name, 
	   const std::string& desc, 
	   ParamLevel level,
	   ParamConstraint *constraint = NULL);
  
  virtual ~XORParam();

  XORParam &operator=(const XORParam &other);

  virtual bool MatchesName(const std::string &nameToMatch) const;

  virtual bool MatchesName(const char* nameToMatch) const;

  virtual void SetParam(std::string name);

  virtual ParamBase *GetSetParam(){ return mSetParam; }

  virtual const ParamBase *GetSetParam() const { return mSetParam; }

  virtual TiXmlElement *ParseToXML(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec) const;

  virtual void DoParse(const TiXmlElement *element);

  virtual void Output(std::ostream& os, std::string indent, bool brief=false) const;

  virtual void GenerateXML(TiXmlNode *parent, ParamLevel maxLevel, bool includeComments) const;

  CopyFunctionMacro(XORParam)

protected:
  
  ParamBase *mSetParam;

};

// ########### End Class Definition ########### //
// ######## Begin Class Implementation ######## //

#ifndef SWIG

// Copy Constructor
inline
XORParam::
XORParam(const XORParam &other)
  : CompoundParam(other)
{
  if(other.mSetParam != NULL){
    mSetParam = GetChild(other.mSetParam->GetName());
  }else{
    mSetParam = NULL;
  }
}

inline
XORParam::
XORParam(const std::string& name,
	 const std::string& desc, 
	 ParamLevel level,
	 ParamConstraint *constraint)
  : CompoundParam(name, desc, level, constraint),
    mSetParam(NULL)
{}

inline
XORParam::
~XORParam()
{}

inline
XORParam &
XORParam::
operator=(const XORParam &other)
{
  if(this != &other){
    this->CompoundParam::operator=(other);
    
    if(other.mSetParam != NULL){
      mSetParam = GetChild(other.mSetParam->GetName());
    }else{
      mSetParam = NULL;
    }
    
  }
  
  return *this;
}

inline
bool
XORParam::
MatchesName(const std::string &nameToMatch) const
{
  return MatchesName(nameToMatch.c_str());
}
  
inline
bool
XORParam::
MatchesName(const char* nameToMatch) const
{
  std::string name(nameToMatch);
  // if the name matches the name of the XORParam itsself
  if(ParamBase::MatchesName(nameToMatch)) return true;
  // or if the name matches one of the XORed params
  if(GetChild(name) != NULL) return true;
  // otherwise return false
  return false;
}

inline
void
XORParam::
SetParam(std::string name)
{
  if(this->IsSet()){
    throw( ParamException(__FILE__, __LINE__, 
			  "XOR node " + this->GetName() + " is being set twice (by " +
			  mSetParam->GetName() + " and " + name));
  }

  ParamBase *param = GetChild(name);
  if(param == NULL){
    throw( ParamException(__FILE__, __LINE__, 
			  "XORParam " + this->GetName() + " has no child named " + name));
  }

  mSetParam = param;

  mAlreadySet = true;
}

inline
TiXmlElement *
XORParam::
ParseToXML(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec) const
{
  
  const ParamNameValPair *curPair = &nameValVec[curIdx];

  const ParamBase *param = GetChild(curPair->Name());

  if(param == NULL){
    throw( ParamException(__FILE__, __LINE__, 
			  "XORParam " + this->GetName() + " has no child named " + curPair->Name()));
  }
  
  return param->ParseToXML(curIdx, nameValVec);
}

inline
void 
XORParam::
DoParse(const TiXmlElement *element)
{

  std::string paramName = element->ValueStr();
  
  if(this->IsSet()){
    throw( ParamException(__FILE__, __LINE__, 
			  "XOR node " + this->GetName() + " is being set twice (by " +
			  mSetParam->GetName() + " and " + paramName));
  }

  // get child elements

  LOGNODE(logDEBUG3) << "Parsing XORParam " << GetName() << " (with " <<  paramName << ")";

  
  ParamBase *param = GetChild(paramName);
  if(param == NULL){
    throw( ParamException(__FILE__, __LINE__, 
			  "XORParam " + this->GetName() + " has no child named " + paramName));
  }

  // parse one of the child params
  try{
    param->Parse(element);
  }catch(ParamException e){
    std::stringstream ss;
    ss << "Recieved exception while parsing XOR element " << this->GetName();
    throw(ParamException(__FILE__, __LINE__, ss.str(), &e));
  }

  mSetParam = param;

  mAlreadySet = true;

}

inline
void 
XORParam::
Output(std::ostream& os, std::string indent, bool brief) const
{
  std::stringstream out;
  out << "XOR " << GetName();
  if(!IsSet()) out << " (set one of the following) ";
  if(Required()) out << " (required) ";
  if(IsSet()) out << " (set) ";
  if(!brief){
    out << " " << GetDescription();
  }
  this->PrintFormatted(std::cout, out.str(), indent);
  
  indent = indent + "   ";
  if(!IsSet()){
    TChildVec::const_iterator it;
    for(it = mChildParams.begin(); it != mChildParams.end(); ++it){
      (*it)->Print(indent, brief);
    }
  }else{
    mSetParam->Print(indent, brief);
  }
}

inline
void
XORParam::
GenerateXML(TiXmlNode *parent, ParamLevel maxLevel, bool includeComments) const
{

  if(this->Level() > maxLevel){
    return;
  }

  if(includeComments){
    TiXmlComment *comment = new TiXmlComment();
    comment->SetValue("XOR param default");
    parent->LinkEndChild(comment);
  }
  if(mChildParams.size() == 0){
    throw( ParamException(__FILE__, __LINE__, 
			  "XORParam " + this->GetName() + " has no children"));
  }
  mChildParams[0]->GenerateXML(parent, maxLevel, includeComments);
}

#endif // !SWIG

#endif // __XOR_PARAM_H__
