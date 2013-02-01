
/**
 * This class provides some nice debugging functionality.
 */

#ifndef DEBUGGING_H
#define DEBUGGING_H

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include <fstream>
#include <string>
#include <map>


//This macro prints out the file and line number.
#define dbinfo(param) \
    do{ \
        if(Debugging::debugging()){ \
            Debugging::out() << "[Debug][" << __FILE__; \
            Debugging::out() << "(" << __LINE__ << "): " << param << "]" << std::endl;\
        } \
    } while(false)

//This macro prints the file and line number and keeps track
//of the number of times it has been called.
#define dbcount(param) \
    do{ \
        if(Debugging::debugging()){ \
            Debugging::out() << "[Debug][" << __FILE__; \
            Debugging::out() << "(" << __LINE__ << "): " << param << "(";\
            Debugging::out() << Debugging::dcount(param) << ")]" << std::endl;\
        }\
    } while(false)

    //This macro resets the counter of how many times dbcount(x) has
    //been called.
#define dbresetCounter(param) \
    do{ \
        Debugging::dresetCounter(param);\
    }while(false)

    // Prints out parameter name and value unconditionally, to cout.
#define dbv(param) \
    do{ \
            std::cout << #param << " = " << param << std::endl;\
    }while(false)

    //Prints out filename, line number, parameter name, and value
#define dbvar(param) \
    do{ \
        if(Debugging::debugging()){ \
            Debugging::out() << "[Debug][" << __FILE__; \
            Debugging::out() << "(" << __LINE__ << "): " << #param << "(";\
            Debugging::out() << param << ")]" << std::endl;\
        }\
    }while(false)

    //Prints out filename, line number, no params
#define dbmark \
    do{ \
        if(Debugging::debugging()){ \
            Debugging::out() << "[Debug][" << __FILE__; \
            Debugging::out() << "(" << __LINE__ << ")]" << std::endl;\
        } \
    }while(false) 

class Debugging{
    public:
        //Turn the printing of debugging messages on and off.
        static void debugOn();
        static void debugOff();

        //Is debugging on or off?
        static bool debugging();
        //Print the decoding of the current error number.
        static std::string decodeError();
        //Print the decoding of a specific error number.
        static std::string decodeError(int);

        //Increment the count of the number of times a string
        //has been called.
        static int dcount(std::string);
        //Increment the count of the number of times an int
        //has been called.
        static int dcount(int);
        //Reset the above counters.
        static void dresetCounter(std::string);
        static void dresetCounter(int);

        //Get the output stream
        static std::ostream &out();

        //Set the new output stream
        static void setOutputStream(std::ostream *);
        static void setOutputStream(std::ostream &);


    private:
        static bool debuggingIsOn;
        static std::map<std::string,int> counters;
        static std::ostream *outstream;

};

#endif
