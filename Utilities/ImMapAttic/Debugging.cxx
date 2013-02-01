
/* Debugging class
   This class turns debugging on and off and 
   can be checked by other debug classes to see if it on.
   Some macro type functions defined in the .h
   It also provides error number decoding.
*/


#include "Debugging.h"

#include <string.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <map>
#include <errno.h>
#include <string.h>

using std::string;
using std::map;
using std::endl;


bool Debugging::debuggingIsOn = true;
map<string,int> Debugging::counters;
std::ostream* Debugging::outstream = &std::cerr;

//turn debugging on
void Debugging::debugOn(){
    debuggingIsOn = true;
}

//turn debugging off
void Debugging::debugOff(){
    debuggingIsOn = false;
}

//Are we debugging or not?
bool Debugging::debugging(){
    return debuggingIsOn;
}

//Gets the error string of the current error number.
string Debugging::decodeError(){
    return decodeError(errno);
}

//Gets the error string of a specific error number.
string Debugging::decodeError(int error){
    string errorStr = "[System Error]";
    errorStr.append(strerror(error));
    return errorStr;
}

//Keep a count of the number of times dcount(countString) has
//been called.
int Debugging::dcount(string countString){
	std::map<string,int>::iterator cit = counters.find(countString);
    if(cit == counters.end()){
        counters[countString] = 0;
    }
    int returnVal = counters[countString];
    counters[countString]++;
    return returnVal;
}

//Keep a count of the number of times dcount(countInt) has
//been called.
int Debugging::dcount(int countInt){
    char countStr[100];
    sprintf(countStr,"%d",countInt);
    return dcount(string(countStr));
}

//Reset the dcount counter for a string.
void Debugging::dresetCounter(string countString){
    counters[countString] = 0;
}

//Reset the dcount counter for an int.
void Debugging::dresetCounter(int countInt){
    char countStr[100];
    sprintf(countStr,"%d",countInt);
    dresetCounter(string(countStr));
}

std::ostream &Debugging::out(){
    return *outstream;
}

void Debugging::setOutputStream(std::ostream *newOutstream){
    outstream = newOutstream;
}

void Debugging::setOutputStream(std::ostream &newOutstream){
    outstream = &newOutstream;
}

