#ifndef FILEPARSER_H
#define FILEPARSER_H

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include <string>
#include <list>
#include <vector>

class BasicFileParser
{
public:
  typedef std::pair<std::string, std::string> StrStrPair;
  typedef std::list<StrStrPair> StrStrList;
  StrStrList keyValuePairs;

  BasicFileParser();
  BasicFileParser(char split, char comment);

  void parseFile(const std::string& filename);
  void clearAll();
  void printPairs() const;

private:
  char _split;
  char _comment;
};

#endif
