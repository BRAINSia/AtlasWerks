
/**
  * My Exception handling class.
  * Stores info about location and a message.
  */

#include <stdio.h>
#include <errno.h>
#include <fstream>
#include <math.h>

#include "BasicException.h"
#include "Debugging.h"

using std::endl;

BasicException::BasicException() throw(){
  whatStr = "";
  lastError = errno;
  line = -1;
  file = "Unknown location";
  message = "Unknown problem";
  if(Debugging::debugging()){
    Debugging::out() << what() << endl;
  }
}

BasicException::BasicException(const string tLocation,const string tMessage) throw(){
  whatStr = "";
  lastError = errno;
  file = tLocation;
  message = tMessage;
  line = -1;
  if(Debugging::debugging()){
    Debugging::out() << what() << endl;
  }
}

BasicException::BasicException(const string tFile,
                       const string tMessage,const int tLine) throw(){
  whatStr = "";
  file = tFile;
  line = tLine;
  message = tMessage;
  lastError = errno;
  if(Debugging::debugging()){
    Debugging::out() << what() << endl;
  }
}

BasicException::BasicException(const BasicException &me) throw(){
  exception::operator=(me);
  line = me.line;
  file = me.file;
  message = me.message;
  lastError = me.lastError;
}

BasicException::~BasicException() throw(){
}

BasicException &BasicException::operator=(const BasicException &me) throw(){
  exception::operator=(me);
  line = me.line;
  file = me.file;
  message = me.message;
  lastError = me.lastError;
  return *this;
}

const char *BasicException::what() throw(){
  whatStr = "";
  whatStr.append("BasicException: ");
  whatStr.append(" [");
  whatStr.append(message);
  whatStr.append("] ");
  whatStr.append("::");
  whatStr.append(file);
  whatStr.append("(");
  char lineStr[100];
  sprintf(lineStr,"%d",line);
  whatStr.append(lineStr);
  whatStr.append(")");
  return whatStr.c_str();
}

const char *BasicException::getFile(){
  return file.c_str();
}

int BasicException::getLine(){
  return line;
}

const char *BasicException::getMessage(){
  return message.c_str();
}

int BasicException::getLastError(){
  return lastError;
}

