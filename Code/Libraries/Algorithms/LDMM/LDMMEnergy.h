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

#ifndef __LDMM_ENERGY_H__
#define __LDMM_ENERGY_H__

#include <vector>
#include <iostream>

#include "AtlasWerksTypes.h"
#include "AtlasWerksException.h"
#include "Energy.h"

class LDMMEnergy : public Energy
{
public:
  LDMMEnergy(int nTimeSteps, Real sigma);
  LDMMEnergy(const LDMMEnergy &other);
  ~LDMMEnergy();

  virtual Energy* Duplicate() const { return new LDMMEnergy(*this); }

  /** Return the number of timesteps */
  Real NTimeSteps() const { return mNTimeSteps; }
  /** Return sigma */
  Real Sigma() const { return mSigma; }
  
  /** Return the total energy */
  Real GetEnergy() const;

  /** Return the image energy */
  Real ImageEnergy() const { return (1.0/(mSigma*mSigma))*mImageEnergy; }
  /** Return the vector energy */
  Real VecEnergy() const;
  /** Return the vector energy at a particular timestep */
  Real VecStepEnergy(unsigned int t) const;

  /** Set the vector energy for the next timestep */
  void SetVecStepEnergy(Real energy);
  /** Set the vector energy for a particular timestep */
  void SetVecStepEnergy(Real energy, unsigned int t);
  /** Set the image energy */
  void SetImageEnergy(Real energy);
  /** Reset energy to initial state */
  void Clear();

  /** Return whether this energy will output individual step energies
      (verbose mode) */
  bool Verbose(){ return mVerbose; }
  /** Set whether this energy will output individual step energies
      (verbose mode) */
  void Verbose(bool v){ mVerbose = v; }
  
  /** assignment operator */
  LDMMEnergy& operator=(const LDMMEnergy &other);
  /** Scale energy by 'scale' */
  LDMMEnergy operator*(float scale) const;
  /** divide energy by 'div' */
  LDMMEnergy operator/(float div) const;
  /** Add two energies */
  LDMMEnergy operator+(const LDMMEnergy &other) const;
  /** Add an energy to this energy */
  LDMMEnergy& operator+=(const LDMMEnergy &other);

  /** Print energy to os */
  virtual void Print(std::ostream& os) const;
  /** Create XML representation of this energy */
  virtual TiXmlElement *ToXML();
  
  /** For use with MPI, get the size of the float buffer needed to
      hold serialized data */
  unsigned int GetSerialSize() const;
  /** For use with MPI, serialize the state of this energy for
      summation across nodes */
  void Serialize(Real*) const;
  /** For use with MPI, unserialize the state of this energy after
      summation across nodes */
  void Unserialize(const Real*);

protected:
  unsigned int mNTimeSteps;
  Real mSigma;
  unsigned int mCurTimeStep;
  Real mImageEnergy;
  Real *mVecStepEnergy;
  bool mVerbose;

private:
  // purposefully hidden
  void SetEnergy(Real energy) {}

};

#endif // __LDMM_ENERGY_H__
