/* ================================================================
 *
 * AtlasWerks Project
 *
 * Copyright (c) Sarang C. Joshi. All rights reserved.  See
 * Copyright.txt or for details.
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notice for more information.
 *
 * ================================================================ */

#include "AtlasWerksDebug.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool ask_question(std::string question, bool def)
{

  std::string s;
  char cstr[256];
  while (true)
    {
    // beep, ask the question, and show the default value
    std::cout << "\a" << question << (def ? " [Yn] " : " [yN] ") << std::flush;
    
    fgets(cstr,256,stdin);
    s = cstr;

    if (s == "\n" || s[0] == 'y' || s[0] == 'Y')
      return true;
    else if ((s.length() == 0 && !def) || s[0] == 'n' || s[0] == 'N')
      return false;

    std::cout << "Unknown input, try again." << std::endl;
    }
}


#ifndef NDEBUG

#include <signal.h>
#include <sys/types.h>
#include <stdio.h>

void printPIDandStop(int sig)
{ // This gets called whenever we get an error signal
  // 
  // It just prints the current process's PID and stops the process,
  // so we can inspect the process live with GDB
  std::cerr << strsignal(sig) << ".  Current PID=" << getpid() << std::endl;
  
  if (ask_question("Would you like to attach gdb to inspect the running process?", true))
    {
      char buf[256];
      snprintf(buf,255,"gdb --pid=%d",getpid());
      system(buf);
    }

  // Stop the process so we can attach to it with gdb if we want
  //kill(getpid(), SIGSTOP);

  // reset the handler to default
  signal(sig, SIG_DFL);
}

void setupSignalHandlers()
{ // handle a few fatal signals gracefully
  signal(SIGSEGV,&printPIDandStop);
  signal(SIGBUS,&printPIDandStop);
  signal(SIGABRT,&printPIDandStop);
}
#endif
