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

#ifdef DARWIN
#include <sys/time.h>
#endif

#include "Timer.h"
#include <iomanip>

Timer
::Timer()
{
  reset();
}

void
Timer
::start()
{
  if (!_isRunning)
    {
      _ftime(_lastStartTime);
      _isRunning  = true;
    }  
}

void
Timer
::stop()
{
  if (_isRunning)
    {
      timeb stopTime;
      _ftime(stopTime);
      
      _totalStartToStopSeconds += 
	static_cast<unsigned long>(stopTime.time - _lastStartTime.time);

      _additionalTotalStartToStopMilliseconds += 
	static_cast<long>(stopTime.millitm) - 
	static_cast<long>(_lastStartTime.millitm);

      _isRunning = false;
    }
}

void
Timer
::reset()
{
  _totalStartToStopSeconds = 0;
  _additionalTotalStartToStopMilliseconds = 0;
  _isRunning = false;
}

void
Timer
::restart()
{
  reset();
  start();
}

unsigned long
Timer
::getDays() const
{
  return this->getSeconds()/(60*60*24);
}

unsigned long
Timer
::getHours() const
{
  return this->getSeconds()/(60*60);
}

unsigned long
Timer
::getMinutes() const
{
  return this->getSeconds()/60;
}

unsigned long
Timer
::getSeconds() const
{
  if (_isRunning)
    {
      timeb now; 
      _ftime(now);
      return _totalStartToStopSeconds + 
	static_cast<unsigned long>(now.time - _lastStartTime.time);
    }
  else
    {
      return _totalStartToStopSeconds;
    }
}

std::string 
Timer
::getTime() const
{
  unsigned long sec;
  unsigned long msec;
  
  if (_isRunning)
    {
      timeb now;
      _ftime(now);
      sec = 
        _totalStartToStopSeconds + 
	static_cast<unsigned long>(now.time - _lastStartTime.time);
      msec =
        _additionalTotalStartToStopMilliseconds
	+ static_cast<long>(now.millitm)
	- static_cast<long>(_lastStartTime.millitm);       
    }
  else
    {
      sec = _totalStartToStopSeconds; 
      msec = _additionalTotalStartToStopMilliseconds;
    }

  unsigned long hours = sec / (60 * 60);
  sec -= hours * 60 * 60;
  
  unsigned long minutes = sec / 60;
  sec -= minutes * 60;

  std::ostringstream oss;
  oss << hours << ":" 
      << std::setw(2) << std::setfill('0') << minutes << ":"
      << std::setw(2) << std::setfill('0') << sec << "." 
      << std::setw(3) << std::setfill('0') << msec;

  return oss.str();
}

unsigned long
Timer
::getMilliseconds() const
{
  if (_isRunning)
    {
      timeb now;
      _ftime(now);
      unsigned long seconds = _totalStartToStopSeconds + 
	static_cast<unsigned long>(now.time - _lastStartTime.time);
      
      return seconds * 1000 + _additionalTotalStartToStopMilliseconds
	+ static_cast<long>(now.millitm)
	- static_cast<long>(_lastStartTime.millitm);       
    }
  else
    {
      return _totalStartToStopSeconds * 1000 
	+ _additionalTotalStartToStopMilliseconds;
    }
}

void
Timer
::_ftime( timeb& theTime ) const
{
#ifndef DARWIN
  ftime( &theTime );
#else
  struct timeval timeVal;
  struct timezone timeZone;
  gettimeofday( &timeVal, &timeZone );
  theTime.time = timeVal.tv_sec;
  theTime.millitm = timeVal.tv_usec / 1000;
  theTime.timezone = timeZone.tz_minuteswest;
  theTime.dstflag = timeZone.tz_dsttime;
#endif
}
