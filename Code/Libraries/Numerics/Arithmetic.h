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

#ifndef ARITHMETIC_H
#define ARITHMETIC_H

// Some implementations of modulo operator return negative values for
// negative arguments.  In such case we want to add the base again in
// order to get a result that is between 0 and b-1 inclusive
inline
int
safe_mod(int r, int b)
{
  // TODO: detect popular compilers and implement this specifically
  // for each, to avoid an if at every point (for speed)
  int m = r % b;

  if (m < 0)
    m += b;

  return m;
}

#endif // ARITHMETIC_H
