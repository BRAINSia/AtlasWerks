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


#ifndef __MULTI_PARAM_H__
#define __MULTI_PARAM_H__

#ifndef SWIG

#include "ParamBase.h"

#endif // !SWIG

template<class TParamType> 
class MultiParam : public ParamBase {

public:
  
  MultiParam(TParamType param,
	     ParamConstraint *constraint = NULL);
  MultiParam(const MultiParam<TParamType> &other);
  ~MultiParam();
  
  /**
   * Return a reference to the template parameter
   */
  TParamType &GetTemplateParam();

  unsigned int GetNumberOfParsedParams() const { return mParsedParams.size(); }
  TParamType &GetParsedParam(int i) { return *mParsedParams[i]; }
  void AddParsedParam(const TParamType &param) { mParsedParams.push_back(new TParamType(param)); }
  const TParamType &GetParsedParam(int i) const { return *mParsedParams[i]; }
  std::vector<TParamType*> &GetParsedParamVec() { return mParsedParams; }

  /** Access a parsed param, equivalent to GetParsedParam(i) */
  TParamType &operator[](int i){ return GetParsedParam(i); }
  const TParamType &operator[](int i) const { return GetParsedParam(i); }
  /** Equivalent to GetNumberOfParsedParams() */
  size_t size() const { return GetNumberOfParsedParams(); }

  virtual TiXmlElement *ParseToXML(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec) const;

  virtual void DoParse(const TiXmlElement*);

  CopyFunctionMacro(MultiParam)

  MultiParam<TParamType> &operator=(const MultiParam<TParamType> &other);

  virtual void Output(std::ostream& os, std::string indent, bool brief=false) const;

  virtual void GenerateXML(TiXmlNode *parent, ParamLevel maxLevel, bool includeComments) const;

  virtual bool MatchesName(const char* nameToMatch) const;
  virtual bool MatchesName(const std::string &nameToMatch) const;
  
protected:
  /** template child parameter */
  TParamType mTemplateParam;
  /** list of child parameters */
  std::vector<TParamType*> mParsedParams;
};

// ########### End Class Definition ########### //
// ######## Begin Class Implementation ######## //

#ifndef SWIG

template<class TParamType>
MultiParam<TParamType>::
MultiParam(TParamType param,
	   ParamConstraint *constraint)
  : ParamBase(param.GetName(), param.GetDescription(), param.Level(), constraint),
    mTemplateParam(param)
{}

template<class TParamType>
MultiParam<TParamType>::
MultiParam(const MultiParam<TParamType> &other)
  : ParamBase(other),
    mTemplateParam(other.mTemplateParam)
{
  for(size_t i=0;i<other.mParsedParams.size();i++){
    mParsedParams.push_back(new TParamType(*(other.mParsedParams[i])));
  }
}

template<class TParamType>
MultiParam<TParamType>::
~MultiParam()
{
  for(size_t i=0;i<mParsedParams.size();i++){
    delete mParsedParams[i];
  }
}

template<class TParamType>
TParamType &
MultiParam<TParamType>::
GetTemplateParam()
{
  return mTemplateParam;
}

template<class TParamType>
bool 
MultiParam<TParamType>::
MatchesName(const char* nameToMatch) const
{
  return mTemplateParam.MatchesName(nameToMatch);
}

template<class TParamType>
bool 
MultiParam<TParamType>::
MatchesName(const std::string &nameToMatch) const
{
  return mTemplateParam.MatchesName(nameToMatch);
}

template<class TParamType>
TiXmlElement *
MultiParam<TParamType>::
ParseToXML(unsigned int &curIdx, const std::vector<ParamNameValPair> &nameValVec) const
{
  return mTemplateParam.ParseToXML(curIdx, nameValVec);
}

template<class TParamType>
void 
MultiParam<TParamType>::
DoParse(const TiXmlElement *element)
{
  LOGNODE(logDEBUG3) << "Parsing MultiParam " << GetName();

  TParamType *newParam = new TParamType(mTemplateParam);
  newParam->Parse(element);
  mParsedParams.push_back(newParam);
  std::cout << "mParsedParams.size() = " << mParsedParams.size() << std::endl;

  // copy newly-parsed values to template as new defaults
  mTemplateParam = *newParam;
  // but unset so as not to generate errors...
  mTemplateParam.Unset();

  mAlreadySet = true;
}

template<class TParamType>
MultiParam<TParamType> &
MultiParam<TParamType>::
operator=(const MultiParam<TParamType> &other)
{
  if(this != &other){
    this->ParamBase::operator=(other);

    // copy the template
    mTemplateParam = other.mTemplateParam;

    // delete any current parsed nodes
    for(size_t i=0;i<mParsedParams.size();i++){
      delete mParsedParams[i];
    }
    mParsedParams.clear();

    // deep copy parsed params
    for(size_t i=0;i<other.mParsedParams.size();i++){
      mParsedParams.push_back(new TParamType(*(other.mParsedParams[i])));
    }
  }
  return *this;
}

template<class TParamType>
void
MultiParam<TParamType>::
Output(std::ostream& os, std::string indent, bool brief) const
{
  std::stringstream out;
  out << "--MultiParam-- " << GetName();
  if(Required()) out << " (required) ";
  if(IsSet()) out << " (set) ";
  this->PrintFormatted(os, out.str(), indent);

  indent = indent + "   ";
  out.str("");
  
  out << "--Template--";
  this->PrintFormatted(os, out.str(), indent);

  indent = indent + "   ";
  out.str("");

  mTemplateParam.Output(os, indent, brief);
  if(mParsedParams.size()){
    out << "--Parsed Values--";
    this->PrintFormatted(std::cout, out.str(), indent);
    for(size_t i=0;i<mParsedParams.size();i++){
      mParsedParams[i]->Output(os, indent, true);
    }
  }
}

template<class TParamType>
void
MultiParam<TParamType>::
GenerateXML(TiXmlNode *parent, ParamLevel maxLevel, bool includeComments) const
{
  if(this->Level() > maxLevel){
    return;
  }

  if(includeComments){
    TiXmlComment *comment = new TiXmlComment();
    comment->SetValue("Multiple of the following can occur...");
    parent->LinkEndChild(comment);
  }
  unsigned int nParsed = GetNumberOfParsedParams();
  if(nParsed){
    for(size_t i=0;i<nParsed;i++){
      mParsedParams[i]->GenerateXML(parent, maxLevel, includeComments);
    }
  }else{
    mTemplateParam.GenerateXML(parent, maxLevel, includeComments);
  }
}

#endif // !SWIG

#endif // __MULTI_PARAM_H__
