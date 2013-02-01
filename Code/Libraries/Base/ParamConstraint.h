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


#ifndef __PARAM_CONSTRAINT_H__
#define __PARAM_CONSTRAINT_H__

class  ParamBase;

class ParamConstraint
{
protected:
  ParamConstraint(const std::string &desc) :
    mDescription(desc)
  {}

  std::string mDescription;

public:
  
  /**
   * Returns a description of the Constraint.
   */
  virtual std::string Description() const 
  { return mDescription; }

  /**
   * Check constraint after 'caller' has been parsed
   * \param caller - the param to be checked
   */
  virtual bool Check(ParamBase* caller) const =0;

  /**
   * Destructor.
   * Silences warnings about Constraint being a base class with virtual
   * functions but without a virtual destructor.
   */
  virtual ~ParamConstraint() { ; }
};

#endif
