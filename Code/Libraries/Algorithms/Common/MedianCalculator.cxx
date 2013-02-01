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

#include <algorithm>
#include <sstream>
#include "Array3DUtils.h"
#include "MedianCalculator.h"
#include "log.h"

Real 
MedianCalculator::
Select(Real *buff, 
       unsigned int buffSz, 
       unsigned int k)
{
  Real rtn;
  Select(buff, buffSz, &k, 1, &rtn);
  return rtn;
}

void 
MedianCalculator::
Select(Real *buff, 
       unsigned int buffSz, 
       unsigned int *k,
       unsigned int kSz,
       Real *rtn)
{
  int rank = 0;

#ifdef MPI_ENABLED
  int err;

  LOGNODETHREAD(logDEBUG3) << "Getting rank:";
  err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(err != MPI_SUCCESS){
    throw AtlasWerksException(__FILE__,__LINE__,"Error returned from MPI_Comm_rank");
  }
#endif

  std::vector<Real> sortvec = Sort(buff, buffSz);

  if(rank == 0){
    for(unsigned int i=0; i<kSz; ++i){
      rtn[i] = sortvec[k[i]];
      LOGNODETHREAD(logDEBUG3) << "k is: " << k[i] << ", val is: " << rtn[i];
    }
  }

#ifdef MPI_ENABLED
  // broadcast result to all nodes
  LOGNODETHREAD(logDEBUG3) << "Broadcasting...";
  err = MPI_Bcast(rtn, kSz, MPI_FLOAT, 0, MPI_COMM_WORLD);
  LOGNODETHREAD(logDEBUG3) << "Done Broadcasting.";
  if(err != MPI_SUCCESS){
    throw AtlasWerksException(__FILE__,__LINE__,"Error returned from MPI_Gather");
  }
#endif
  
}

std::vector<Real>
MedianCalculator::
Sort(Real *buff, 
     unsigned int buffSz)
{
  Real *total_buff = NULL;
  unsigned int nTotalSz = 0;
  int rank = 0;
  int gsize = 0;
  std::ostringstream debugmsg;

#ifdef MPI_ENABLED
  int *size_buff = NULL;
  int *off_buff = NULL;
  int err;

  LOGNODETHREAD(logDEBUG3) << "Getting rank:";
  err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(err != MPI_SUCCESS){
    throw AtlasWerksException(__FILE__,__LINE__,"Error returned from MPI_Comm_rank");
  }
  err = MPI_Comm_size(MPI_COMM_WORLD, &gsize);
  if(err != MPI_SUCCESS){
    throw AtlasWerksException(__FILE__,__LINE__,"Error returned from MPI_Comm_size");
  }
  if(rank == 0){
    size_buff = new int[gsize];
    off_buff = new int[gsize];
  }
  if(gsize > 1){
    MPI_Gather(&buffSz, 1, MPI_INT, 
	       size_buff, 1, MPI_INT, 
	       0, MPI_COMM_WORLD);
    if(rank == 0){
      for(int i=0;i<gsize;++i){
	off_buff[i] = nTotalSz;
	nTotalSz += size_buff[i];
      }
    }
  }else{
    nTotalSz = buffSz;
  }
#else
  nTotalSz = buffSz;
#endif

  LOGNODETHREAD(logDEBUG3) << "Rank is: " << rank;
  LOGNODETHREAD(logDEBUG3) << "group size is: " << gsize;

  if(rank == 0){
    LOGNODETHREAD(logDEBUG3) << "Rank zero allocating total_buff of size " << nTotalSz;
    total_buff = new Real[nTotalSz];
    LOGNODETHREAD(logDEBUG3) << "Rank zero total_buff is " << total_buff;
  }

  // send all data to central node
  if(gsize > 1){
#ifdef MPI_ENABLED
    LOGNODETHREAD(logDEBUG3) << "Gathering...";
    err =
      MPI_Gatherv(buff, buffSz, MPI_FLOAT, 
		  total_buff, size_buff, 
		  off_buff, MPI_FLOAT,
		  0, MPI_COMM_WORLD);
    LOGNODETHREAD(logDEBUG3) << "Done Gathering.";
    if(err != MPI_SUCCESS){
      throw AtlasWerksException(__FILE__,__LINE__,"Error returned from MPI_Gather");
    }
#else
    throw AtlasWerksException(__FILE__,__LINE__,"Error, group size > 1, but MPI not enabled?");
#endif
  }else{
    memcpy(total_buff, buff, buffSz*sizeof(Real));
  }

  std::vector<Real> sortvec;
  // have central node sort
  if(rank == 0){
    LOGNODETHREAD(logDEBUG3) << "Rank zero running sort";
    debugmsg.str("");
    debugmsg << "Gathered buff: ";
    for(unsigned int i=0;i<nTotalSz;++i){
      debugmsg << total_buff[i] << " ";
      sortvec.push_back(total_buff[i]);
    }
    LOGNODETHREAD(logDEBUG3) << debugmsg.str();
    std::sort(sortvec.begin(), sortvec.end());
    debugmsg.str("");
    debugmsg << "Sorted Buff: ";
    for(unsigned int i=0;i<nTotalSz;++i){
      debugmsg << sortvec[i] << " ";
    }
    LOGNODETHREAD(logDEBUG3) << debugmsg.str();
    delete [] total_buff;
  }

  return sortvec;
}

void 
MedianCalculator::
MinMax(std::vector<RealImage*> input,
       RealImage &min,
       RealImage &max)
{
  unsigned int nImages = input.size();
  SizeType size = input[0]->getSize();
  unsigned int nVox = size.productOfElements();
  OriginType origin = input[0]->getOrigin();
  OriginType spacing = input[0]->getSpacing();
  min = *input[0];
  max = *input[0];
  Real *minData = min.getDataPointer();
  Real *maxData = max.getDataPointer();
  Real *curData = NULL;
  // compute min/max of local images
  for(unsigned int imIdx=1;imIdx < nImages; ++imIdx){
    curData = input[imIdx]->getDataPointer();
    for(unsigned int i=0; i<nVox; ++i){
      minData[i] = curData[i] < minData[i] ? curData[i] : minData[i];
      maxData[i] = curData[i] > maxData[i] ? curData[i] : maxData[i];
    }
  }
#ifdef MPI_ENABLED
  RealImage tmp(size, origin, spacing);
  MPI_Allreduce(min.getDataPointer(), tmp.getDataPointer(), nVox, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  min = tmp;
  MPI_Allreduce(max.getDataPointer(), tmp.getDataPointer(), nVox, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  max = tmp;
#endif
}

void 
MedianCalculator::
GetIndex(std::vector<RealImage*> input,
	 const RealImage &pim,
	 const RealImage &gmin,
	 const RealImage &gmax,
	 unsigned int *idx,
	 Real *min,
	 Real *max)
{
  unsigned int nImages = input.size();
  SizeType size = input[0]->getSize();
  unsigned int nVox = size.productOfElements();

  const Real *gminData = gmin.getDataPointer();
  const Real *gmaxData = gmax.getDataPointer();
  const Real *pData = pim.getDataPointer();
  for(unsigned int i=0;i<nVox;++i){
    min[i] = gminData[i];
    max[i] = gmaxData[i];
    idx[i] = 0;
  }

  for(unsigned int imIdx=0;imIdx<nImages;++imIdx){

    Real *curData = input[imIdx]->getDataPointer();
    for(unsigned int i=0;i<nVox;++i){
      Real v = curData[i];
      Real p = pData[i];
      if(v < p){
	idx[i]++;
	if(v > min[i]){
	  min[i] = v;
	}
      }else{
	if(v < max[i]){
	  max[i] = v;
	}
      }
		 
    } // end loop over voxels
  } // end loop over images

#ifdef MPI_ENABLED
  unsigned int *tmp_ui = new unsigned int[nVox];
  Real *tmp_f = new Real[nVox];

  MPI_Allreduce(idx, tmp_ui, nVox, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  for(unsigned int i=0;i<nVox;++i){ idx[i] = tmp_ui[i]; }
  MPI_Allreduce(min, tmp_f, nVox, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  for(unsigned int i=0;i<nVox;++i){ min[i] = tmp_f[i]; }
  MPI_Allreduce(max, tmp_f, nVox, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  for(unsigned int i=0;i<nVox;++i){ max[i] = tmp_f[i]; }

  delete [] tmp_ui;
  delete [] tmp_f;
#endif
  
}

void 
MedianCalculator::
Select(std::vector<RealImage*> input,
       unsigned int k,
       RealImage &rtn)
{
  unsigned int nImages = input.size();
  SizeType size = input[0]->getSize();
  unsigned int nVox = size.productOfElements();
  OriginType origin = input[0]->getOrigin();
  OriginType spacing = input[0]->getSpacing();
  rtn.resize(size);
  rtn.setOrigin(origin);
  rtn.setSpacing(spacing);
  Real *rtnData = rtn.getDataPointer();
  unsigned int maxIters = nImages;

#ifdef MPI_ENABLED  
  int gsize;
  MPI_Comm_size(MPI_COMM_WORLD, &gsize);
  maxIters *= gsize;
#endif

  Real *curMin = new Real[nVox];
  Real *curMax = new Real[nVox];
  unsigned int *idx = new unsigned int[nVox];
  Real *vp = new Real[nVox];
  Real *vn = new Real[nVox];
  bool *finished = new bool[nVox];
  unsigned int finishedCount = 0;

  RealImage pim(size, origin, spacing);
  RealImage gmin(size, origin, spacing);
  RealImage gmax(size, origin, spacing);

  // get global min/max
  MinMax(input, gmin, gmax);
  Real totalMin, totalMax, foo;
  Array3DUtils::getMinMax(gmin, totalMin, foo);
  Array3DUtils::getMinMax(gmax, foo, totalMax);
  Real eps = (totalMax-totalMin)/10000.f;
  
  Real *pData = pim.getDataPointer();
  Real *gmaxData = gmax.getDataPointer();
  Real *gminData = gmin.getDataPointer();

  for(unsigned int i=0;i<nVox;++i){
    finished[i] = false;
    curMin[i] = gminData[i];
    curMax[i] = gmaxData[i];
  }
  
  unsigned int iter = 0;
  while(true){

    LOGNODETHREAD(logDEBUG3) << "iter: " << iter << ", finishedCount: " << finishedCount;

    for(unsigned int i=0;i<nVox;++i){

      if(!finished[i]){
	// if max == min
	if(eps > fabs(curMax[i] - curMin[i])){
	  rtnData[i] = curMin[i];
	  finished[i] = true;
	  finishedCount++;
	  continue;
	}
	pData[i] = (curMin[i]+curMax[i])/2.0;
	
      }
    }
    
    GetIndex(input, pim, gmin, gmax, idx, vp, vn);

    for(unsigned int i=0;i<nVox;++i){

      if(!finished[i]){
        if(idx[i] == k){
	  rtnData[i] = vn[i];
	  finished[i] = true;
	  finishedCount++;
	  continue;
	}else if(idx[i]-1 == k){
	  rtnData[i] = vp[i];
	  finished[i] = true;
	  finishedCount++;
	  continue;
	}else if(idx[i] > k){
	  curMax[i] = vp[i];
	}else if(idx[i] < k){
	  curMin[i] = vn[i];
	}
	
      }
    }

    if(iter > maxIters){
      for(unsigned int i=0;i<nVox;++i){
	if(!finished[i]){ 
	  LOGNODETHREAD(logWARNING) << "iter " << iter << ", voxel " 
				    << i << " not finished.";
	  LOGNODETHREAD(logWARNING) << "idx " << idx[i];
	  LOGNODETHREAD(logWARNING) << "p " << pData[i];
	  LOGNODETHREAD(logWARNING) << "min " << curMin[i];
	  LOGNODETHREAD(logWARNING) << "max " << curMax[i];
	  LOGNODETHREAD(logWARNING) << "max-min " << curMax[i]-curMin[i];
	  LOGNODETHREAD(logWARNING) << "vp " << vp[i];
	  LOGNODETHREAD(logWARNING) << "vn " << vn[i];
	  LOGNODETHREAD(logWARNING) << "vp-vn " << vp[i]-vn[i];
	  LOGNODETHREAD(logWARNING) << "gmin " << gmin(i);
	  LOGNODETHREAD(logWARNING) << "gmax" << gmax(i);
	  LOGNODETHREAD(logWARNING) << "k " << k;
	  for(unsigned int imIdx=0; imIdx<input.size(); ++imIdx){
	    LOGNODETHREAD(logWARNING) << "pixel val " << (*input[imIdx])(i);
	  }
	}
      }
      throw AtlasWerksException("Error, Select() not converging (see prev. warnings)");
    }
    iter++;
    if(finishedCount == nVox) break;
  }

  delete [] curMin;
  delete [] curMax;
  delete [] idx;
  delete [] vp;
  delete [] vn;
  delete [] finished;

}

