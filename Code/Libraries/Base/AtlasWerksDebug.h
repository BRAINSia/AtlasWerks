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

#ifndef ATLASWERKS_DEBUG_H
#define ATLASWERKS_DEBUG_H

#include <iostream>

bool ask_question(std::string question, bool def);

#ifndef NDEBUG
  // declare functions used in debug mode
  void printPIDandStop(int sig);
  void setupSignalHandlers();
  #define ATLASWERKS_SIGHANDLERS setupSignalHandlers()

  // Set up a debug-only ostream, for debug-mode only output
  // note that you use this like DBOUT("some text" << otherstuff)
  // it's a little more cumbersome than being able to just do 
  // dbout << "some text"
  // but it's the only way I know to get it to compile to nothing in
  // Release mode
  #define DBOUT( x ) std::cout << x << std::endl;
#else
  #define DBOUT( x ) std::cout << x << std::endl;
  #define ATLASWERKS_SIGHANDLERS
#endif

#endif // ATLASWERKS_DEBUG_H
