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


// this file is #include'd in "GreedyWarp.h"

/** ################ GreedyWarp ################ **/

template<class GreedyIteratorT>
GreedyWarp<GreedyIteratorT>::
GreedyWarp(const RealImage *I0,
	   const RealImage *I1,
	   const GreedyWarpParam &param,
	   Affine3D *aff,
	   std::string warpName)
  : WarpInterface(I0, I1),
    mNScaleLevels(param.GetNumberOfScaleLevels()),
    mDeformationData(I0, I1, param),
    mParam(param),
    mIterator(NULL)
{
  mScaleManager = new MultiscaleManager(mImSize, mImSpacing, mImOrigin, mParam);
  mScaleManager->SetScaleVectorFields(true);
  mDeformationData.ComputeInverseHField(mParam.WriteInvDefField());
  if(aff){
    mDeformationData.SetInitialAffine(*aff);
  }
  // use image name as identifier
  mDeformationData.SetName(warpName);
  mIterator = new GreedyIteratorT(mImSize, mImOrigin, mImSpacing, true);
}

template<class GreedyIteratorT>
GreedyWarp<GreedyIteratorT>::
~GreedyWarp()
{
}

template<class GreedyIteratorT>
void
GreedyWarp<GreedyIteratorT>::
SetScaleLevel(const MultiscaleManager &scaleManager)
{

  // get scale level information
  mCurSize = scaleManager.CurScaleSize();
  mCurSpacing = scaleManager.CurScaleSpacing();
  
  // scale per-thread data
  int scale = scaleManager.CurScaleLevel();
  mCurScaleParam = &mParam.GetScaleLevel(scale).Iterator();
  mIterator->SetScaleLevel(scaleManager, *mCurScaleParam);
  
  // scale deformation data
  mDeformationData.SetScaleLevel(scaleManager);

}

template<class GreedyIteratorT>
void
GreedyWarp<GreedyIteratorT>::
RunWarp()
{

  mDeformationData.InitializeWarp();

  for(unsigned int scaleLevel = 0; scaleLevel < mNScaleLevels; scaleLevel++)
    {
      mScaleManager->SetScaleLevel(scaleLevel);
      this->SetScaleLevel(*mScaleManager);
      
      this->GenerateCurOutput(StringUtils::strPrintf("ScaleLevel%02dInitial",scaleLevel), 
			      mParam.WriteInitialScaleLevelDefImage(), 
			      mParam.WriteInitialScaleLevelInvDefImage(), 
			      mParam.WriteInitialScaleLevelDefField(), 
			      mParam.WriteInitialScaleLevelInvDefField()); 

      if(mParam.WriteScaleLevelImages()){
	this->WriteScaleLevelImages(scaleLevel);
      }

      unsigned int nIters = mParam.GetScaleLevel(scaleLevel).NIterations();
      
      // iterate
      for(unsigned int iter=0;iter<nIters;iter++){

	mDeformationData.SetCurIter(iter);

	if(iter == 0) mIterator->UpdateStepSizeNextIteration();
	
	mIterator->Iterate(mDeformationData);

	if(iter == 0){
	  LOGNODE(logINFO) << "Step Size is " << mDeformationData.StepSize();
	}
	
	if(mDeformationData.HasEnergy()){
	  LOGNODE(logINFO) << *mDeformationData.GetEnergyHistory().LastEnergyEvent();
	}

      } // end iteration

      this->GenerateCurOutput(StringUtils::strPrintf("ScaleLevel%02dFinal",scaleLevel), 
			      mParam.WriteFinalScaleLevelDefImage(), 
			      mParam.WriteFinalScaleLevelInvDefImage(), 
			      mParam.WriteFinalScaleLevelDefField(), 
			      mParam.WriteFinalScaleLevelInvDefField()); 

    } // end iterate over scale levels

  mDeformationData.FinalizeWarp();

  this->GenerateOutput();

}

template<class GreedyIteratorT>
void
GreedyWarp<GreedyIteratorT>::
WriteScaleLevelImages(int scaleLevel)
{
  char fname[1024];

  LOGNODE(logINFO) << "Writing scale level I0";
  sprintf(fname, "%sScaleLevel%dI0.%s", 
	  mParam.OutputPrefix().c_str(), 
	  scaleLevel,
	  mParam.OutputSuffix().c_str());
  ApplicationUtils::SaveImageITK(fname, mDeformationData.I0());

  LOGNODE(logINFO) << "Writing scale level I1";
  sprintf(fname, "%sScaleLevel%dI1.%s", 
	  mParam.OutputPrefix().c_str(), 
	  scaleLevel,
	  mParam.OutputSuffix().c_str());
  ApplicationUtils::SaveImageITK(fname, mDeformationData.I1());

}

template<class GreedyIteratorT>
void
GreedyWarp<GreedyIteratorT>::
WriteDefImage(std::string title)
{
  std::cout << "Writing deformed image" << std::endl;
  RealImage defImage(mImSize, mImOrigin, mImSpacing);
  std::string fname = 
    StringUtils::strPrintf("%s%s%sDefImage.%s", 
			   mParam.OutputPrefix().c_str(), 
			   mDeformationData.GetName().c_str(), 
			   title.c_str(), 
			   mParam.OutputSuffix().c_str());
  mDeformationData.GetI0At1(defImage);
  ApplicationUtils::SaveImageITK(fname.c_str(), defImage);
}

template<class GreedyIteratorT>
void
GreedyWarp<GreedyIteratorT>::
WriteInvDefImage(std::string title)
{
  std::cout << "Writing reverse deformed image" << std::endl;
  RealImage defImage(mImSize, mImOrigin, mImSpacing);
  std::string fname = 
    StringUtils::strPrintf("%s%s%sInvDefImage.%s", 
			   mParam.OutputPrefix().c_str(), 
			   mDeformationData.GetName().c_str(), 
			   title.c_str(), 
			   mParam.OutputSuffix().c_str());
  mDeformationData.GetI1At0(defImage);
  ApplicationUtils::SaveImageITK(fname.c_str(), defImage);
}

template<class GreedyIteratorT>
void
GreedyWarp<GreedyIteratorT>::
WriteDefField1To0(std::string title)
{
  std::cout << "Writing deformation field" << std::endl;
  VectorField h(mImSize);
  std::string fname = 
    StringUtils::strPrintf("%s%s%sDefField1To0.%s", 
			   mParam.OutputPrefix().c_str(), 
			   mDeformationData.GetName().c_str(), 
			   title.c_str(), 
			   mParam.OutputSuffix().c_str());
  mDeformationData.GetDef1To0(h);
  ApplicationUtils::SaveHFieldITK(fname.c_str(), h, mImOrigin, mImSpacing);
}

template<class GreedyIteratorT>
void
GreedyWarp<GreedyIteratorT>::
WriteDefField0To1(std::string title)
{
  std::cout << "Writing inverse deformation field:" << std::endl;
  VectorField h(mImSize);
  std::string fname = 
    StringUtils::strPrintf("%s%s%sDefField0To1.%s", 
			   mParam.OutputPrefix().c_str(), 
			   mDeformationData.GetName().c_str(), 
			   title.c_str(), 
			   mParam.OutputSuffix().c_str());
  mDeformationData.GetDef0To1(h);
  ApplicationUtils::SaveHFieldITK(fname.c_str(), h, mImOrigin, mImSpacing);
}

template<class GreedyIteratorT>
void
GreedyWarp<GreedyIteratorT>::
WriteEnergy(std::string title)
{
  RealImage im;
  std::string fname = 
    StringUtils::strPrintf("%s%s%sEnergy.xml", 
			   mParam.OutputPrefix().c_str(), 
			   mDeformationData.GetName().c_str(), 
			   title.c_str());
  mDeformationData.GetEnergyHistory().SaveXML(fname.c_str());
}

template<class GreedyIteratorT>
void
GreedyWarp<GreedyIteratorT>::
GenerateCurOutput(const std::string &prefix, 
		  bool writeDefImage, bool writeInvDefImage, 
		  bool writeDefField, bool writeInvDefField)
{

  if(writeDefImage){
    this->WriteDefImage(prefix);
  }
  
  if(writeInvDefImage){
    this->WriteInvDefImage(prefix);
  }
  
  if(writeDefField){
    this->WriteDefField1To0(prefix);
  }

  if(writeInvDefField){
    this->WriteDefField0To1(prefix);
  }

}

template<class GreedyIteratorT>
void 
GreedyWarp<GreedyIteratorT>::
GenerateOutput()
{
  this->GenerateCurOutput(std::string(""), 
			  mParam.WriteDefImage(), 
			  mParam.WriteInvDefImage(), 
			  mParam.WriteDefField(), 
			  mParam.WriteInvDefField()); 

  if(mParam.WriteEnergy()){
    this->WriteEnergy();
  }

}
