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


#ifndef __SCALE_LEVEL_PARAM_ORDERING_CONSTRAINT_H__
#define __SCALE_LEVEL_PARAM_ORDERING_CONSTRAINT_H__

/**
 * Constraint with re-orders a MultiParam of CompoundParams (class
 * TParamType) by the downsample factor of their child ScaleLevel(),
 * largest to smallest, of the child ScaleLevelParam.  That is, the
 * largest downsample factor, which results in smallest image, is the
 * 0th element of the MultiParam, and the full-sized image (factor=1)
 * is the last element.  Also checks that downsample factors are
 * unique.
 *
 * To recap, this constraint should be added to a MultiParam of some
 * subclass of CompoundParam that has a child ScaleLevelParam called
 * ScaleLevel.
 */
template<class TParamType> 
class ScaleLevelParamOrderingConstraint : public ParamConstraint {
public:
  ScaleLevelParamOrderingConstraint()
    : ParamConstraint("Reorders a MultiParam of CompoundParams based on downsample factor, "
		      "largest to smallest, and ensures unique downsample factors")
  {}
  
  virtual bool Check(ParamBase* caller) const 
  {
    MultiParam<TParamType> *scaleLevelList = static_cast<MultiParam<TParamType>*>(caller);
    std::vector<TParamType*> &parsedParamVec = scaleLevelList->GetParsedParamVec();
    int nParsedParams = parsedParamVec.size();
    if(nParsedParams == 0){
      throw( ParamException(__FILE__, __LINE__,
			    "Parse constraint check called on MultiParam with no parsed values"));
    }
    
    // remove the last parsed level, we'll re-insert it in the correct location
    // (ordering by downsample factor, greatest to least)
    TParamType *curScaleLevel = parsedParamVec.back();
    parsedParamVec.pop_back();
    unsigned int downsampleFactor = curScaleLevel->ScaleLevel().DownsampleFactor();
    typename std::vector<TParamType*>::iterator it = parsedParamVec.begin();
    for( ; it != parsedParamVec.end(); ++it){
      if(downsampleFactor == (*it)->ScaleLevel().DownsampleFactor()){
	std::stringstream is;
	is << "Trying to add multiple scale levels with the same downsample value: downsample = " << downsampleFactor;
	throw( ParamException(__FILE__, __LINE__, is.str().c_str()));
      }
      if(downsampleFactor > (*it)->ScaleLevel().DownsampleFactor()){
	// we'll insert here
	break;
      }
    }
    parsedParamVec.insert(it,curScaleLevel);
    return true;
  }
};

#endif // __SCALE_LEVEL_PARAM_ORDERING_CONSTRAINT_H__
