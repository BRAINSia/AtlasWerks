#include "EnergyHistory.h"

std::ostream& operator<< (std::ostream& os, const HistoryEvent& e)
{
  e.Print(os);
  return os;
}

// ################ EnergyIncreaseEvent ################ //

void
EnergyIncreaseEvent::
Print(std::ostream& os) const
{
  os << "Energy Increase Detected!";
}

TiXmlElement*
EnergyIncreaseEvent::
ToXML(TiXmlElement *parent) const
{
  TiXmlElement *eiEl = new TiXmlElement("EnergyIncrease");
  parent->LinkEndChild(eiEl);
  return parent;
}

// ################ StepSizeEvent ################ //

void
StepSizeEvent::
Print(std::ostream& os) const
{
  os << "Step Size changed to " << mStepSize;
}

TiXmlElement*
StepSizeEvent::
ToXML(TiXmlElement *parent) const
{
  static char buff[256];
  TiXmlElement *ssEl = new TiXmlElement("StepSizeChange");
  sprintf(buff, "%f", mStepSize);
  ssEl->SetAttribute("val", buff);
  parent->LinkEndChild(ssEl);
  return parent;
}

// ################ IterationEvent ################ //

IterationEvent::
IterationEvent(unsigned int scale,
	       unsigned int iter,
	       const Energy &energy)
  : mScale(scale),
    mIter(iter)
{
  mEnergy = energy.Duplicate();
}

IterationEvent::
IterationEvent(const IterationEvent &other)
{
  mScale = other.mScale;
  mIter = other.mIter;
  mEnergy = other.mEnergy->Duplicate();
}

IterationEvent::
~IterationEvent()
{
  delete mEnergy;
}

IterationEvent&
IterationEvent::
operator=(const IterationEvent &other){
  if(&other == this) return *this;

  mScale = other.mScale;
  mIter = other.mIter;
  if(mEnergy) delete mEnergy;
  mEnergy = other.mEnergy->Duplicate();
  return *this;
}

TiXmlElement*
IterationEvent::
ToXML(TiXmlElement *parent) const
{
  static char buff[256];
  TiXmlElement *iterEl = new TiXmlElement("Iteration");
  sprintf(buff, "%d", mScale);
  iterEl->SetAttribute("Scale", buff);
  sprintf(buff, "%d", mIter);
  iterEl->SetAttribute("Iter", buff);
  iterEl->LinkEndChild(mEnergy->ToXML());
  parent->LinkEndChild(iterEl);
  return parent;
}

void 
IterationEvent::
Print(std::ostream& os) const
{
  os << "Scale " << mScale 
     << ", Iter " << mIter 
     << ", " << *mEnergy;
}

// ################ ReparameterizeEvent ################ //

ReparameterizeEvent::
ReparameterizeEvent(const Energy &energy)
{
  mEnergy = energy.Duplicate();
}

ReparameterizeEvent::
ReparameterizeEvent(const ReparameterizeEvent &other)
{
  mEnergy = other.mEnergy->Duplicate();
}

ReparameterizeEvent::
~ReparameterizeEvent()
{
  delete mEnergy;
}

ReparameterizeEvent&
ReparameterizeEvent::
operator=(const ReparameterizeEvent &other){
  if(&other == this) return *this;

  if(mEnergy) delete mEnergy;
  mEnergy = other.mEnergy->Duplicate();
  return *this;
}

TiXmlElement*
ReparameterizeEvent::
ToXML(TiXmlElement *parent) const
{
  TiXmlElement *iterEl = new TiXmlElement("Reparameterize");
  iterEl->LinkEndChild(mEnergy->ToXML());
  parent->LinkEndChild(iterEl);
  return parent;
}

void 
ReparameterizeEvent::
Print(std::ostream& os) const
{
  os << "Reparameterization energy: " << *mEnergy;
}

// ################ EnergyHistory ################ //

EnergyHistory::
~EnergyHistory()
{
  for(unsigned int i=0;i<mHistory.size();++i){
    delete mHistory[i];
  }
}

void 
EnergyHistory::
AddEvent(const HistoryEvent &ev)
{
  HistoryEvent *h = ev.Duplicate();
  if(h->IsEnergyEvent()){
    mEnergyEventMap.push_back(mHistory.size());
  }
  mHistory.push_back(h);
}

unsigned int
EnergyHistory::
NumEnergyEvents() const
{
  return mEnergyEventMap.size();
}

HistoryEvent*
EnergyHistory::
GetEnergyEvent(unsigned int idx)
{
  if(idx < 0 || idx >= NumEnergyEvents()) return NULL;
  return mHistory[mEnergyEventMap[idx]];
}

const HistoryEvent*
EnergyHistory::
GetEnergyEvent(unsigned int idx) const
{
  if(idx < 0 || idx >= NumEnergyEvents()) return NULL;
  return mHistory[mEnergyEventMap[idx]];
}

Energy*
EnergyHistory::
GetEnergy(unsigned int idx)
{
  HistoryEvent *e = GetEnergyEvent(idx);
  if(e){
    return e->GetEnergy();
  }
  return NULL;
}

const Energy*
EnergyHistory::
GetEnergy(unsigned int idx) const
{
  const HistoryEvent *e = GetEnergyEvent(idx);
  if(e){
    return e->GetEnergy();
  }
  return NULL;
}

HistoryEvent*
EnergyHistory::
LastEnergyEvent()
{
  return this->GetEnergyEvent(this->NumEnergyEvents()-1);
}

const HistoryEvent*
EnergyHistory::
LastEnergyEvent() const
{
  return this->GetEnergyEvent(this->NumEnergyEvents()-1);
}

Energy*
EnergyHistory::
LastEnergy()
{
  HistoryEvent *last = LastEnergyEvent();
  if(last){
    return last->GetEnergy();
  }
  return NULL;
}

const Energy*
EnergyHistory::
LastEnergy() const
{
  const HistoryEvent *last = LastEnergyEvent();
  if(last){
    return last->GetEnergy();
  }
  return NULL;
}

HistoryEvent* 
EnergyHistory::
operator[](unsigned int idx)
{
  return mHistory[idx];
}

const HistoryEvent* 
EnergyHistory::
operator[](unsigned int idx) const
{
  return mHistory[idx];
}

Real 
EnergyHistory::
LastEnergyChange()
{
  if(this->NumEnergyEvents() > 1){
    Real last = this->GetEnergy(this->NumEnergyEvents()-1)->GetEnergy();
    Real prev = this->GetEnergy(this->NumEnergyEvents()-2)->GetEnergy();
    return last-prev;
  }
  return 0.f;
}

void 
EnergyHistory::
Print(std::ostream& os) const
{
  for(unsigned int i=0;i<mHistory.size();++i){
    mHistory[i]->Print(os);
    os << std::endl;
  }
}

TiXmlElement*
EnergyHistory::
ToXML() const
{
  TiXmlElement *histEl = new TiXmlElement("EnergyHistory");
  TiXmlElement *curParent = histEl;
  for(unsigned int i=0; i<mHistory.size();++i){
    curParent = mHistory[i]->ToXML(curParent);
  }
  return histEl;
}

void 
EnergyHistory::
SaveXML(const char *filename) const
{
  TiXmlDocument *doc = new TiXmlDocument();
  TiXmlElement *el = this->ToXML();
  doc->LinkEndChild(el);
  doc->SaveFile(filename);
  delete doc;
}

std::ostream& operator<< (std::ostream& os, const EnergyHistory& e)
{
  e.Print(os);
  return os;
}
