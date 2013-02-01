
/**
 * This is an extension of the exception class to provide and
 * store more information about errors.
 */

#ifndef BASICEXCEPTION_H
#define BASICEXCEPTION_H

#include <exception>
#include <string>
#include <stdio.h>

using std::string;

//This macro prints the file and line number along with
//the exception text.
#define bException(details) \
        BasicException(details,__FILE__,__LINE__)

class BasicException : public std::exception{
  public:
    BasicException() throw();
    BasicException(const string,const string) throw();
    BasicException(const string,const string,const int) throw();
    BasicException(const BasicException &) throw();
    virtual ~BasicException() throw();
    BasicException &operator=(const BasicException &) throw();
 
    //"What" is the information about the exception
    virtual const char *what() throw();
    //The filename of the occurence.
    virtual const char *getFile();
    //The line number of the occurence.
    virtual int getLine();
    //The specified message.
    virtual const char *getMessage();
    //The error number at the time of creation.
    int getLastError();

  protected:
    string file;
    int line;
    string message;
    int lastError;
    string whatStr;
};

#endif

