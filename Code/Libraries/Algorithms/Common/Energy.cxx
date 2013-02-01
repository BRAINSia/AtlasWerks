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

#include "Energy.h"

Energy 
Energy::
operator*(float scale) const
{
  Energy rtn;
  rtn.mEnergy = this->mEnergy * scale;
  return rtn;
}

Energy 
Energy::
operator/(float div) const
{
  Energy rtn;
  rtn.mEnergy = this->mEnergy / div;
  return rtn;
}

Energy 
Energy::
operator+(const Energy &other) const
{
  Energy rtn;
  rtn.mEnergy = this->mEnergy + other.mEnergy;
  return rtn;
}

Energy& 
Energy::
operator+=(const Energy &other)
{
  this->mEnergy += other.mEnergy;
  return *this;
}

void 
Energy::
Print(std::ostream& os) const
{
  os << "Energy = " << this->GetEnergy();
}

TiXmlElement*
Energy::
ToXML() const
{
  static char buff[256];
  TiXmlElement *element = new TiXmlElement("Energy");
  sprintf(buff, "%f", this->GetEnergy());
  element->SetAttribute("val", buff);
  return element;
}

unsigned int 
Energy::
GetSerialSize() const
{
  return 1;
}

void
Energy::
Serialize(Real* buff) const
{
  buff[0] = mEnergy; 
}
 
void 
Energy::
Unserialize(const Real* buff)
{
  mEnergy = buff[0]; 
}

std::ostream& 
operator<< (std::ostream& os, const Energy& e)
{
  e.Print(os);
  return os;
}
