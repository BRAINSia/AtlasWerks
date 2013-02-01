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

//#include "LDMMWarp.h"

/* ################ LDMMWarp ################ */

template<class LDMMIteratorT>
LDMMWarp<LDMMIteratorT>::
LDMMWarp(const RealImage *I0,
	 const RealImage *I1,
	 const LDMMWarpParam &param,
	 std::string warpName)
  : WarpInterface(I0, I1),
    mNTimeSteps(param.NTimeSteps()),
    mNScaleLevels(param.GetNumberOfScaleLevels()),
    mDeformationData(I0, I1, param),
    mParam(param),
    mIterator(NULL),
    mCurScaleParam(NULL)
{
    
  mScaleManager = new MultiscaleManager(mImSize, mImSpacing, mImOrigin, mParam);
  mScaleManager->SetScaleVectorFields(false);
  std::cerr << "Create multi scale manager done" << std::endl;
  //cutilCheckMsg("Multi scale manager");
  
  mIterator = new LDMMIteratorT(mImSize, mImOrigin, mImSpacing, mNTimeSteps);
  
  // use image name as identifier
  mDeformationData.SetName(warpName);

}

template<class LDMMIteratorT>
LDMMWarp<LDMMIteratorT>::
~LDMMWarp()
{
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  // get scale level information
  mCurSize = scaleManager.CurScaleSize();
  mCurSpacing = scaleManager.CurScaleSpacing();
  
  int scale = scaleManager.CurScaleLevel();
  mCurScaleParam = &mParam.GetScaleLevel(scale);
  mIterator->SetScaleLevel(scaleManager, mCurScaleParam->Iterator());
  
  // scale deformation data
  mDeformationData.SetScaleLevel(scaleManager);
  mDeformationData.StepSize(mCurScaleParam->Iterator().StepSize());
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
RunWarp()
{

  mDeformationData.InitializeWarp();

  this->BeginWarp();
  
  for(unsigned int scaleLevel = 0; scaleLevel < mNScaleLevels; scaleLevel++)
    {

      mScaleManager->SetScaleLevel(scaleLevel);
      this->SetScaleLevel(*mScaleManager);

      LOGNODETHREAD(logDEBUG) 
	<< StringUtils::strPrintf("Scale %d Image size %d %d %d", 
				  scaleLevel, mCurSize.x, mCurSize.y, mCurSize.z);

      if(mParam.WriteInitialScaleLevelDefImage()){
	std::string title = StringUtils::strPrintf("ScaleLevel%dInitial",scaleLevel);
	this->WriteDefImage(title);
      }

      unsigned int nIters = mParam.GetScaleLevel(scaleLevel).NIterations();

      // iterate
      for(unsigned int iter=0;iter<nIters;iter++){
	
	if(iter == 0 && mCurScaleParam->Iterator().UseAdaptiveStepSize()){
	  LOGNODETHREAD(logINFO) << "Calculating step size";
	  mIterator->UpdateStepSizeNextIteration();
	  mIterator->Iterate(mDeformationData);
	  LOGNODETHREAD(logINFO) << "Done calculating step size";
	}

	mDeformationData.SetCurIter(iter);

	this->BeginIteration(scaleLevel, iter);
	
	mIterator->Iterate(mDeformationData);

	if(mCurScaleParam->ReparameterizeEveryNIterations() > 0 &&
	   (iter+1)%mCurScaleParam->ReparameterizeEveryNIterations() == 0){
	  mIterator->ReParameterize(mDeformationData);
	}
	
	this->FinishIteration(scaleLevel, iter);

	if(mDeformationData.HasEnergy()){
	  LOGNODE(logINFO) << *mDeformationData.GetEnergyHistory().LastEnergyEvent();
	}
      }

      if(mParam.WriteFinalScaleLevelDefImage()){
	std::string title = StringUtils::strPrintf("ScaleLevel%dFinal",scaleLevel);
	this->WriteDefImage(title);
      }

      
    } // end iterate over scale levels
  
  this->FinishWarp();
  
  mDeformationData.FinalizeWarp();
  
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
GetDefImage(RealImage &defIm)
{
  mDeformationData.GetI0At1(defIm);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
GetInvDefImage(RealImage &defIm)
{
  mDeformationData.GetI1At0(defIm);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
GetDefImage(RealImage &defIm, unsigned int tIdx)
{
  mDeformationData.GetI0AtT(defIm, tIdx);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
GetInvDefImage(RealImage &defIm, unsigned int tIdx)
{
  mDeformationData.GetI1AtT(defIm, tIdx);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
GetDefField(VectorField& hField)
{
  mDeformationData.GetDef1To0(hField);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
GetInvDefField(VectorField& hField)
{
  mDeformationData.GetDef0To1(hField);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
GetAlpha0(RealImage &alpha)
{
  alpha =  mDeformationData.Alpha(0); 
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
GetAlpha(RealImage &alpha, int tIdx)
{
  alpha = mDeformationData.Alpha(tIdx); 
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
GetVField(VectorField &v, int tIdx)
{ 
  mDeformationData.GetVField(v, tIdx); 
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
WriteDefImage(std::string title)
{
  LOGNODE(logDEBUG) << "Writing deformed image.";
  std::string fname = 
    StringUtils::strPrintf("%s%s%sDefImage.%s", 
			   mParam.OutputPrefix().c_str(), 
			   mDeformationData.GetName().c_str(), 
			   title.c_str(), 
			   mParam.OutputSuffix().c_str());
  RealImage defImage;
  this->GetDefImage(defImage);
  ApplicationUtils::SaveImageITK(fname.c_str(), defImage);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
WriteInvDefImage(std::string title)
{
  LOGNODE(logDEBUG) << "Writing inverse deformed image.";
  std::string fname = 
    StringUtils::strPrintf("%s%s%sInvDefImage.%s", 
			   mParam.OutputPrefix().c_str(), 
			   mDeformationData.GetName().c_str(), 
			   title.c_str(), 
			   mParam.OutputSuffix().c_str());
  RealImage defImage;
  this->GetInvDefImage(defImage);
  ApplicationUtils::SaveImageITK(fname.c_str(), defImage);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
WriteDefField(std::string title)
{
  LOGNODE(logDEBUG) << "Writing deformation field.";
  std::string fname = 
    StringUtils::strPrintf("%s%s%sDefField.%s", 
			   mParam.OutputPrefix().c_str(), 
			   mDeformationData.GetName().c_str(), 
			   title.c_str(), 
			   mParam.OutputSuffix().c_str());
  VectorField h;
  this->GetDefField(h);
  ApplicationUtils::SaveHFieldITK(fname.c_str(), h, mImOrigin, mCurSpacing);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
WriteInvDefField(std::string title)
{
  LOGNODE(logDEBUG) << "Writing inverse deformation field.";
  std::string fname = 
    StringUtils::strPrintf("%s%s%sInvDefField.%s", 
			   mParam.OutputPrefix().c_str(), 
			   mDeformationData.GetName().c_str(), 
			   title.c_str(), 
			   mParam.OutputSuffix().c_str());
  VectorField h;
  this->GetInvDefField(h);
  ApplicationUtils::SaveHFieldITK(fname.c_str(), h, mImOrigin, mCurSpacing);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
WriteAlpha0(std::string title)
{
  LOGNODE(logDEBUG) << "Writing alpha0.";
  std::string fname = 
    StringUtils::strPrintf("%s%s%sAlpha0.%s", 
			   mParam.OutputPrefix().c_str(), 
			   mDeformationData.GetName().c_str(), 
			   title.c_str(), 
			   mParam.OutputSuffix().c_str());
  RealImage alpha;
  this->GetAlpha(alpha, 0);
  ApplicationUtils::SaveImageITK(fname.c_str(), alpha);
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
WriteAlphas(std::string title)
{
  std::cout << "Writing alphas" << std::endl;
  RealImage alpha;
  for(unsigned int t=0;t<mNTimeSteps;t++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sAlpha%02d.%s", 
			     mParam.OutputPrefix().c_str(), 
			     mDeformationData.GetName().c_str(), 
			     title.c_str(), 
			     t,
			     mParam.OutputSuffix().c_str());
    this->GetAlpha(alpha, t);
    ApplicationUtils::SaveImageITK(fname.c_str(), alpha);
  }
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
WriteVelocityFields(std::string title)
{
  VectorField v;
  for(unsigned int t=0;t<mNTimeSteps;t++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sVelField%02d.%s", 
			     mParam.OutputPrefix().c_str(), 
			     mDeformationData.GetName().c_str(), 
			     title.c_str(), 
			     t,
			     mParam.OutputSuffix().c_str());
    this->GetVField(v, t);
    ApplicationUtils::SaveHFieldITK(fname.c_str(), v, mImOrigin, mImSpacing);
  }
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
WriteIntermediateImages(std::string title)
{
  RealImage im;
  for(unsigned int t=0;t<=mNTimeSteps;t++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sDefImage%02d.%s", 
			     mParam.OutputPrefix().c_str(), 
			     mDeformationData.GetName().c_str(), 
			     title.c_str(), 
			     t,
			     mParam.OutputSuffix().c_str());
    this->GetDefImage(im, t);
    ApplicationUtils::SaveImageITK(fname.c_str(), im);
  }
}

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
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

template<class LDMMIteratorT>
void
LDMMWarp<LDMMIteratorT>::
GenerateOutput()
{
  // save images/deformations
  
  if(mParam.WriteDefImage()){
    this->WriteDefImage();
  }
  
  if(mParam.WriteInvDefImage()){
    this->WriteInvDefImage();
  }
  
  if(mParam.WriteDefField()){
    this->WriteDefField();
  }
  
  if(mParam.WriteInvDefField()){
    this->WriteInvDefField();
  }
  
  if(mParam.WriteAlpha0()){
    this->WriteAlpha0();
  }

  if(mParam.WriteAlphas()){
    this->WriteAlphas();
  }

  if(mParam.WriteVelocityFields()){
    this->WriteVelocityFields();
  }

  if(mParam.WriteIntermediateImages()){
    this->WriteIntermediateImages();
  }

  if(mParam.WriteEnergy()){
    this->WriteEnergy();
  }

}
