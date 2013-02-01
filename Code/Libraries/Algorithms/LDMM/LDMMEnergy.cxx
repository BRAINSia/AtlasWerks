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


#include "LDMMEnergy.h"

LDMMEnergy::
LDMMEnergy(int nTimeSteps, Real sigma)
  : mNTimeSteps(nTimeSteps),
    mSigma(sigma),
    mCurTimeStep(0),
    mImageEnergy(0.0),
    mVecStepEnergy(new Real[mNTimeSteps]),
    mVerbose(false)
{
  this->Clear();
}

LDMMEnergy::
LDMMEnergy(const LDMMEnergy &other)
{
  this->mNTimeSteps = other.mNTimeSteps;
  this->mVecStepEnergy = new Real[this->mNTimeSteps];
  for(unsigned int t=0;t<mNTimeSteps;t++){
    this->mVecStepEnergy[t] = other.mVecStepEnergy[t];
  }
  this->mSigma = other.mSigma;
  this->mCurTimeStep = other.mCurTimeStep;
  this->mImageEnergy = other.mImageEnergy;
  this->mVerbose = other.mVerbose;
}

LDMMEnergy::
~LDMMEnergy()
{
}

Real 
LDMMEnergy::
GetEnergy() const 
{
  return ImageEnergy() + VecEnergy();
}

Real 
LDMMEnergy::
VecEnergy() const
{
  Real val = 0.0f;
  for(unsigned int i=0;i<this->mNTimeSteps;i++){
    val += mVecStepEnergy[i];
  }
  return val/mNTimeSteps;
}

Real
LDMMEnergy::
VecStepEnergy(unsigned int t) const
{
  if(t >= mNTimeSteps){
    throw 
      AtlasWerksException(__FILE__, __LINE__, 
			  "Error, array bounds violation in mVecStepEnergy ");
  }
  return mVecStepEnergy[t];
}

void 
LDMMEnergy::
SetVecStepEnergy(Real energy)
{
  mVecStepEnergy[mCurTimeStep] = energy;
  mCurTimeStep++;
}

void 
LDMMEnergy::
SetVecStepEnergy(Real energy, unsigned int t)
{
  mVecStepEnergy[t] = energy;
}

void 
LDMMEnergy::
SetImageEnergy(Real energy)
{
  if(mImageEnergy != 0){
    throw 
      AtlasWerksException(__FILE__, __LINE__, 
		       "Error, setting image energy twice!");
  }
  mImageEnergy = energy;
}

void 
LDMMEnergy::
Clear()
{
  mCurTimeStep = 0;
  mImageEnergy = 0.f;
  memset(mVecStepEnergy, 0, mNTimeSteps*sizeof(Real));
}

LDMMEnergy&
LDMMEnergy::
operator=(const LDMMEnergy &other)
{
  if(this == &other) return *this;

  if(this->mNTimeSteps != other.mNTimeSteps){
    delete [] mVecStepEnergy;
    this->mVecStepEnergy = new Real[other.mNTimeSteps];
  }
  this->mNTimeSteps = other.mNTimeSteps;
  for(unsigned int t=0;t<mNTimeSteps;t++){
    this->mVecStepEnergy[t] = other.mVecStepEnergy[t];
  }
  this->mSigma = other.mSigma;
  this->mCurTimeStep = other.mCurTimeStep;
  this->mImageEnergy = other.mImageEnergy;
  this->mVerbose = other.mVerbose;

  return *this;
}

LDMMEnergy 
LDMMEnergy::
operator*(float scale) const
{
  LDMMEnergy rtn(mNTimeSteps, mSigma);
  rtn.mImageEnergy = this->mImageEnergy * scale;
  for(unsigned int i=0;i<this->mNTimeSteps;i++){
    rtn.mVecStepEnergy[i] = this->mVecStepEnergy[i] * scale;
  }
  return rtn;
}

LDMMEnergy 
LDMMEnergy::
operator/(float div) const
{
  LDMMEnergy rtn(mNTimeSteps, mSigma);
  rtn.mImageEnergy = this->mImageEnergy / div;
  for(unsigned int i=0;i<this->mNTimeSteps;i++){
    rtn.mVecStepEnergy[i] = this->mVecStepEnergy[i] / div;
  }
  return rtn;
}

LDMMEnergy 
LDMMEnergy::
operator+(const LDMMEnergy &other) const
{
  if(this->mNTimeSteps != other.mNTimeSteps){
    throw 
      AtlasWerksException(__FILE__, __LINE__, 
		       "Error, cannot add energies from iterators "
		       "with different nubmer of timesteps");
  }
  LDMMEnergy rtn(mNTimeSteps, mSigma);
  rtn.mImageEnergy = this->mImageEnergy + other.mImageEnergy;
  for(unsigned int i=0;i<this->mNTimeSteps;i++){
    rtn.mVecStepEnergy[i] = this->mVecStepEnergy[i] + other.mVecStepEnergy[i];
  }
  return rtn;
}

LDMMEnergy& 
LDMMEnergy::
operator+=(const LDMMEnergy &other)
{
  if(this->mNTimeSteps != other.mNTimeSteps){
    throw 
      AtlasWerksException(__FILE__, __LINE__, 
		       "Error, cannot add energies from iterators "
		       "with different nubmer of timesteps");
  }
  this->mImageEnergy += other.mImageEnergy;
  for(unsigned int i=0;i<this->mNTimeSteps;i++){
    this->mVecStepEnergy[i] += other.mVecStepEnergy[i];
  }
  return *this;
}

void 
LDMMEnergy::
Print(std::ostream& os) const {
  Energy::Print(os);
  os << " (total) = " 
     << this->ImageEnergy()  << " (image) + " 
     << this->VecEnergy()  << " (vec)";
  if(this->mVerbose){
    os << " ";
    for(unsigned int t=0;t<this->mNTimeSteps;t++){
      os << "[ Step " << t << " : " << this->mVecStepEnergy[t] << "] ";
    }
  }
}

TiXmlElement*
LDMMEnergy::
ToXML()
{
  // get element with total energy
  TiXmlElement *energyXML = Energy::ToXML();
  static char buff[256];
  // ImageEnergy
  TiXmlElement *element = new TiXmlElement("ImageEnergy");
  sprintf(buff, "%f", this->ImageEnergy());
  element->SetAttribute("val", buff);
  energyXML->LinkEndChild(element);
  // VecEnergy
  TiXmlElement *vecEnergyEl = new TiXmlElement("VecEnergy");
  sprintf(buff, "%f", this->VecEnergy());
  vecEnergyEl->SetAttribute("val", buff);
  sprintf(buff, "%d", mNTimeSteps);
  vecEnergyEl->SetAttribute("NTimeSteps", buff);
  energyXML->LinkEndChild(vecEnergyEl);
  // vector step energies
  for(unsigned int t=0; t < mNTimeSteps; ++t){
    element = new TiXmlElement("VecStepEnergy");
    sprintf(buff, "%f", this->VecStepEnergy(t));
    vecEnergyEl->SetAttribute("val", buff);
    sprintf(buff, "%d", t);
    vecEnergyEl->SetAttribute("t", buff);
  }
  return energyXML;
}

unsigned int 
LDMMEnergy::
GetSerialSize() const
{
  return 1 + mNTimeSteps;
}

void
LDMMEnergy::
Serialize(Real* buff) const
{
  unsigned int buffPos = 0;
  buff[buffPos] = mImageEnergy; 
  buffPos++;
  for(unsigned int i=0;i<mNTimeSteps;i++){
    buff[buffPos] = mVecStepEnergy[i];
    buffPos++;
  }
}
 
void 
LDMMEnergy::
Unserialize(const Real* buff)
{
  unsigned int buffPos = 0;
  mImageEnergy = buff[buffPos]; 
  buffPos++;
  for(unsigned int i=0;i<mNTimeSteps;i++){
    mVecStepEnergy[i] = buff[buffPos];
    buffPos++;
  }
}
