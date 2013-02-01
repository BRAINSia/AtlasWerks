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

#include "Line3D.h"
#include <iostream>

bool 
operator<(const Line3D& lhs, const Line3D& rhs)
{
  if (lhs.p1 < rhs.p1) return true;
  if (rhs.p1 < lhs.p1) return false;
  if (lhs.p2 < rhs.p2) return true;
  return false;    
}

bool 
operator==(const Line3D& lhs, const Line3D& rhs)
{
  return ((lhs.p1 == rhs.p1) && (lhs.p2 == rhs.p2));
}

std::ostream&
operator<<(std::ostream& output, const Line3D& line)
{
  output << "(" << line.p1 << ", " << line.p2 << ")";
  return output;
}
