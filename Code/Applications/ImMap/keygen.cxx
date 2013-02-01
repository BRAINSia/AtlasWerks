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

#include <iostream>
#include <string>
#include <fstream>

int
main(int argc, char** argv)
{
  std::string hostName;
  std::string outFile;
  bool interactive = true;
  if (argc == 3)
    {
      hostName = argv[1];
      outFile = argv[2];
      interactive = false;
    }
  else
    {
      std::cerr << "Enter host name: ";
      std::cin  >> hostName;
    }

  unsigned long asciiSum = 0;
  for (unsigned int i = 0; i < hostName.size(); i++)
    {
      asciiSum += static_cast<unsigned long>(hostName[i]);
    }  
  unsigned long magicNumber = 8675309;
  unsigned long key = asciiSum * magicNumber + 5;

  if (interactive)
    {
      std::cerr << "Key: " << key << std::endl;
    }
  else
    {
      std::ofstream output(outFile.c_str());
      if (output.bad())
	{
	  std::cerr << "failed to open file for writing: " 
		    << outFile << std::endl;
	  return 1;
	}
      output << key;
      output.close();
    }
  return 0;
}
