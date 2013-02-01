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

#ifndef Timer_h
#define Timer_h

#include <iostream>
#include <sys/timeb.h>
#include <string>
#include <sstream>

//
// simple timing class that acts like a stop watch
//
// bcd 2003
//

class Timer
{
public:
  Timer();

  // start timer if not already running
  void start();

  // stop timer if running
  void stop();

  // reset all values to 0, stop timer if running
  void reset();

  // sugar for common command: reset and then start
  void restart();

  // get number of days timer has run
  unsigned long getDays() const;

  // get number of hours timer has run
  unsigned long getHours() const;

  // get number of minutes timer has run
  unsigned long getMinutes() const;

  // get number of seconds timer has run
  unsigned long getSeconds() const;

  // get number of milliseconds timer has run
  unsigned long getMilliseconds() const;

  // get time as a string hours:minutes:seconds.milliseconds
  std::string getTime() const;

private:
  bool           _isRunning;
  timeb          _lastStartTime;

  unsigned long  _totalStartToStopSeconds;
  long           _additionalTotalStartToStopMilliseconds;

  void _ftime( timeb& theTime ) const;
};
#endif
