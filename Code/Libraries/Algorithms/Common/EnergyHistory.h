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

#ifndef __ENERGY_HISTORY_H__
#define __ENERGY_HISTORY_H__

#ifndef SWIG

#include <vector>
#include <iostream>

#include "tinyxml.h"
#include "AtlasWerksTypes.h"
#include "AtlasWerksException.h"
#include "Energy.h"

#endif // !SWIG

// ################ HistoryEvent ################ //

class HistoryEvent {
public:
  virtual ~HistoryEvent(){}
  virtual HistoryEvent* Duplicate() const =0;
  virtual bool IsEnergyEvent() const {return false;}
  virtual Energy *GetEnergy(){ return NULL; }
  virtual const Energy *GetEnergy() const { return NULL; };
  virtual void Print(std::ostream& os) const=0;
  virtual TiXmlElement *ToXML(TiXmlElement *parent) const=0;
  friend std::ostream& operator<< (std::ostream& os, const HistoryEvent& e);
};

std::ostream& operator<< (std::ostream& os, const HistoryEvent& e);

// ################ EnergyIncreaseEvent ################ //

class EnergyIncreaseEvent : public HistoryEvent
{
public:
  EnergyIncreaseEvent()
  {}
  virtual ~EnergyIncreaseEvent(){}
  virtual HistoryEvent* Duplicate() const { return new EnergyIncreaseEvent(*this); }
  virtual void Print(std::ostream& os) const;
  virtual TiXmlElement *ToXML(TiXmlElement *parent) const;
};

// ################ StepSizeEvent ################ //

class StepSizeEvent : public HistoryEvent
{
public:
  StepSizeEvent(Real stepSize)
    : mStepSize(stepSize)
  {}
  virtual ~StepSizeEvent(){}
  virtual HistoryEvent* Duplicate() const { return new StepSizeEvent(*this); }
  virtual void Print(std::ostream& os) const;
  virtual TiXmlElement *ToXML(TiXmlElement *parent) const;

protected:
  Real mStepSize;
};

// ################ IterationEvent ################ //

class IterationEvent : public HistoryEvent
{
public:
  IterationEvent(unsigned int scale,
		 unsigned int iter,
		 const Energy &energy);
  IterationEvent(const IterationEvent &other);
  virtual ~IterationEvent();
  virtual IterationEvent& operator=(const IterationEvent &other);
  virtual HistoryEvent* Duplicate() const { return new IterationEvent(*this); }
  virtual bool IsEnergyEvent() const {return true;}
  virtual Energy *GetEnergy(){ return mEnergy; }
  virtual const Energy *GetEnergy() const { return mEnergy; };
  virtual void Print(std::ostream& os) const;
  virtual TiXmlElement *ToXML(TiXmlElement *parent) const;

protected:
  unsigned int mScale;
  unsigned int mIter;
  Energy *mEnergy;
};

// ################ ReparameterizeEvent ################ //

class ReparameterizeEvent : public HistoryEvent
{
public:
  ReparameterizeEvent(const Energy &energy);
  ReparameterizeEvent(const ReparameterizeEvent &other);
  virtual ~ReparameterizeEvent();
  virtual ReparameterizeEvent& operator=(const ReparameterizeEvent &other);
  virtual HistoryEvent* Duplicate() const { return new ReparameterizeEvent(*this); }
  virtual bool IsEnergyEvent() const {return true;}
  virtual Energy *GetEnergy(){ return mEnergy; }
  virtual const Energy *GetEnergy() const { return mEnergy; };
  virtual void Print(std::ostream& os) const;
  virtual TiXmlElement *ToXML(TiXmlElement *parent) const;
  
protected:
  Energy *mEnergy;
};

// ################ EnergyHistory ################ //

class EnergyHistory
{
public:
  
  EnergyHistory() {}
  virtual ~EnergyHistory();

  virtual void AddEvent(const HistoryEvent &ev);

  virtual unsigned int size() const { return mHistory.size(); }
  virtual HistoryEvent* operator[](unsigned int idx);
  virtual const HistoryEvent* operator[](unsigned int idx) const;
  virtual unsigned int NumEnergyEvents() const;
  virtual HistoryEvent* GetEnergyEvent(unsigned int idx);
  virtual const HistoryEvent* GetEnergyEvent(unsigned int idx) const;
  virtual Energy* GetEnergy(unsigned int idx);
  virtual const Energy* GetEnergy(unsigned int idx) const;
  virtual HistoryEvent* LastEnergyEvent();
  virtual const HistoryEvent* LastEnergyEvent() const;
  virtual Energy* LastEnergy();
  virtual const Energy* LastEnergy() const;

  /**
   * Returns the energy difference between the last and second-to-last
   * energy -- if positive, the energy is increasing (generally a bad
   * thing).  Returns zero if no previous energy.
   */
  virtual Real LastEnergyChange();

  virtual void Print(std::ostream& os) const;

  /** Create XML representation of this energy */
  virtual TiXmlElement *ToXML() const;
  virtual void SaveXML(const char *filename) const;

  friend std::ostream& operator<< (std::ostream& os, const EnergyHistory& e);
  
protected:

  std::vector<HistoryEvent*> mHistory;
  std::vector<unsigned int> mEnergyEventMap;

};

std::ostream& operator<< (std::ostream& os, const EnergyHistory& e);

#endif // __ENERGY_HISTORY_H__
