#include "BasicFileParser.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "StringUtils.h"

BasicFileParser
::BasicFileParser()
  : _split('='), _comment('#')
{}

BasicFileParser
::BasicFileParser(char split, char comment)
  : _split(split), _comment(comment)
{}

void
BasicFileParser
::parseFile(const std::string& filename)
{
  // open file for reading
  std::ifstream input(filename.c_str());
  if (input.fail() || input.bad())
    {
      throw std::runtime_error("failed to open file");
    }

  // read in and parse one line at a time
  std::string line, key, value;
  while (true)
    {
      if (input.eof()) break;

      // read the next line
      std::getline(input, line);
      if (input.bad())
        {
          throw std::runtime_error("input failed");
        }

      // erase everything after the comment
      std::string::size_type commentPos = line.find(_comment);
      if (commentPos != std::string::npos) 
        {
          line.erase(commentPos, line.size() - commentPos);
        }

      // skip lines that dont split
      std::string::size_type splitPos = line.find(_split);
      if (splitPos == std::string::npos) continue;

      // get key and value from line
      key = StringUtils::trimWhitespace(line.substr(0, splitPos));
      value = StringUtils::trimWhitespace(line.erase(0, splitPos+1));

      // insert pair into map
      keyValuePairs.push_back(std::make_pair(key, value));
    }
  input.close();
}

void
BasicFileParser
::clearAll()
{
  keyValuePairs.clear();
}

void
BasicFileParser
::printPairs() const
{
  for (StrStrList::const_iterator p = keyValuePairs.begin(); p != keyValuePairs.end(); ++p)
    {
      std::cout << p->first << ", " << p->second << std::endl;
    }
}
