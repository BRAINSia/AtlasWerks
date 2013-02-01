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

#ifndef __SINGLETON_H
#define __SINGLETON_H

template <class T>
class Singleton
{
public:
  static T& Instance() {
    static T _instance;
    return _instance;
  }
private:
  Singleton();          // ctor hidden
  ~Singleton();          // dtor hidden
  Singleton(Singleton const&);    // copy ctor hidden
  Singleton& operator=(Singleton const&);  // assign op hidden
};

#endif
