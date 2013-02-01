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


//#include "LDMMWarpAd.h"

/** ################ LDMMWarpAd ################ **/

template<class LDMMAdShootingIteratorT>
LDMMWarpAd<LDMMAdShootingIteratorT>::
LDMMWarpAd(const RealImage *I0,
	   const RealImage *I1,
	   const LDMMWarpParam &param,
	   std::string warpName)
  : LDMMWarp<LDMMAdShootingIteratorT>(I0, I1, param, warpName)
{  
}

template<class LDMMAdShootingIteratorT>
LDMMWarpAd<LDMMAdShootingIteratorT>::
~LDMMWarpAd()
{
}

template<class LDMMAdShootingIteratorT>
void 
LDMMWarpAd<LDMMAdShootingIteratorT>::
FinishWarp()
{
  LDMMWarp<LDMMAdShootingIteratorT>::mIterator->finalUpdatePhi0T(LDMMWarp<LDMMAdShootingIteratorT>::mDeformationData);
  LDMMWarp<LDMMAdShootingIteratorT>::mIterator->finalUpdatePhiT0(LDMMWarp<LDMMAdShootingIteratorT>::mDeformationData);
}

