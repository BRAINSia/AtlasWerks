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


/**
 * \page ImMap
 *
 * GUI frontend for image registration using 'greedy' fluid warping
 * algorithm.
 */

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include <iostream>

#include <exception>
#include "ImMap.h"
#include <FL/Fl.H>

#ifdef WIN32
//#include <winsock2.h>
#include <winsock.h>
#else
#include <unistd.h>
#endif

bool findValidKey();

int
main( int argc, char** argv )
{
  // first check for key file
  const bool require_key = false;
  if (require_key && !findValidKey())
    {
      exit(0);
    }
  
  //format the name of the presetfile
  int pos;
  std::string directory(argv[0]);  
  std::cerr << directory << std::endl;
#ifdef WIN32
  pos = directory.find_last_of("\\");
#else
  pos = directory.find_last_of("/");
#endif
  directory.erase(directory.begin() + pos + 1, directory.end());

  //format the presetfile name
  std::string presetFilename=directory;
  presetFilename.replace(presetFilename.end(),presetFilename.end()+1,"preset.dat");

  Fl::visual( FL_RGB );
  Fl::scheme("Plastic");
  
  ImMap imMap;
  imMap.show( 0, 0 );
  imMap.setPresetFilename(presetFilename);
  imMap.loadPresetFileCallback();

  //
  // load images from the command line
  //
  for (int i = 1; i < argc; ++i)
  {
      std::string arg = argv[i];
      if (StringUtils::getPathExtension(arg) == std::string("immap"))
      {
          try
          {
              imMap.runScript(arg);
          }
          catch(std::exception e)
          {
              std::cerr << "Error loading script: " << argv[i] << ": " 
                        << e.what() << std::endl;
          }
      }
      else
      {
          try
          {
              imMap.loadImage(argv[i]);
          }
          catch(std::exception e)
          {
              std::cerr << "Error loading file: " << argv[i] << ": " 
                        << e.what() << std::endl;
          }      
      }
  }

#ifdef FLRUN_NO_TRYCATCH
  // When running the program in the debugger, I don't want Fl::run()
  // to be in a try block, since I *want* an abnormal termination when
  // there is an exception during the event loop.
  Fl::run();
#else
  try 
    {
      Fl::run();
    }
  catch (std::exception e)
    {
	  std::cerr << "IV EXCEPTION: " << e.what() << std::endl;
    }
  catch (...)
    {
      std::cerr << "IV EXCEPTION: UNKNOWN" << std::endl;
    }
#endif
  return 0;

} // main

bool findValidKey()
{
  const std::string keyFilename = "ivkey.dat";
  unsigned long key;
  
#ifdef WIN32
  // 
  // all this is required for windows gethostname to work
  //
  WORD wVersionRequested;
  WSADATA wsaData;
  int err;
  
  wVersionRequested = MAKEWORD( 2, 2 );
  
  err = (int)WSAStartup( wVersionRequested, &wsaData );
  if ( err != 0 ) 
    {
      return false;
    }
#endif

  // get host name
  char nameBuffer[1024];
  gethostname(nameBuffer, 1024);
  std::string hostName(nameBuffer);

  // try to load key file
  std::ifstream input(keyFilename.c_str());
  if (input.fail())
    {
      // get key from user
      std::cerr << "enter key for host " << hostName << ": ";
      std::cin >> key;
    }
  else
    {
      // get key from file
      input >> key;
      input.close();
    }

  unsigned long asciiSum = 0;
  for (unsigned int i = 0; i < hostName.size(); i++)
    {
      asciiSum += static_cast<unsigned long>(nameBuffer[i]);
    }
  
  unsigned long magicNumber = 8675309;
  unsigned long product = asciiSum * magicNumber + 5;

  if (product != key)
    {
      std::cerr << "Invalid key." << std::endl;
      if (key == magicNumber)
	{
	  std::cerr << "correct key is: " << product << std::endl;	  
	}
      return false;
    }
  else
    {
      // write correct key to file
      std::ofstream output(keyFilename.c_str());
      if (output.fail())
	{
	  std::cerr << "Error opening key file for writing." << std::endl;
	  return true;
	}

      output << key;
      output.close();

      if (output.fail())
	{
	  std::cerr << "Error writing key file." << std::endl;
	}

      return true;
    }
}
