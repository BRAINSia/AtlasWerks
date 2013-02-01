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

#ifndef __ENERGY_H__
#define __ENERGY_H__

#ifndef SWIG

#include <vector>
#include <iostream>

#include "tinyxml.h"
#include "AtlasWerksTypes.h"
#include "AtlasWerksException.h"

#endif // !SWIG

class Energy
{
public:
  
  Energy()
    : mEnergy(0.0)
  {}

  virtual ~Energy() {};

  virtual Energy* Duplicate() const { return new Energy(*this); }

  virtual Real GetEnergy() const { return mEnergy; }

  virtual void SetEnergy(Real energy){ mEnergy = energy; }

  /** Reset energy to initial state */
  virtual void Clear(){ mEnergy = 0.0; }

  /** Scale all energy by 'scale' */
  Energy operator*(float scale) const;
  /** divide all energy by 'div' */
  Energy operator/(float div) const;
  /** Add two energies */
  Energy operator+(const Energy &other) const;
  /** Add an energy to this energy */
  Energy& operator+=(const Energy &other);

  /** Print energy to os */
  virtual void Print(std::ostream& os) const;
  /** Create XML representation of this energy */
  virtual TiXmlElement *ToXML() const;

  friend std::ostream& operator<< (std::ostream& os, const Energy& energy);
  
  /** For use with MPI, get the size of the float buffer needed to
      hold serialized data */
  virtual unsigned int GetSerialSize() const;
  /** For use with MPI, serialize the state of this energy for
      summation across nodes */
  virtual void Serialize(Real*) const;
  /** For use with MPI, unserialize the state of this energy after
      summation across nodes */
  virtual void Unserialize(const Real*);

protected:
  Real mEnergy;
};

std::ostream& operator<< (std::ostream& os, const Energy& energy);

#endif // __ENERGY_H__
