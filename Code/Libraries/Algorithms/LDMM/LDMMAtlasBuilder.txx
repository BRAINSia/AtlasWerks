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


// #include "LDMMAtlasBuilder.h"

#include "log.h"
#include <cmath>

#include "MedianCalculator.h"

template<class AtlasManagerType>
LDMMAtlasBuilder<AtlasManagerType>::
LDMMAtlasBuilder(const WeightedImageSet &imageSet,
		    ParamType &param,
		    unsigned int nThreads,
		    unsigned int nodeId,
		    unsigned int nNodes,
		    unsigned int nTotalImages)
  : AtlasBuilderInterface(imageSet, nThreads, nodeId, nNodes, nTotalImages),
    mParam(param),
    mNTimeSteps(mParam.NTimeSteps()),
    mMeanImage(NULL),
    mJacDetSum(NULL),
    mMPIBuff(NULL),
    mCurIterEnergy(mNTimeSteps, 0.f)
{
  mScaleManager = new MultiscaleManagerType(mImSize, mImSpacing, mImOrigin, mParam);
  mScaleManager->SetScaleVectorFields(false);
  mNScaleLevels = mScaleManager->NumberOfScaleLevels();
  mScaleManager->SetInitialScaleLevel(mParam.StartScaleLevel());
  
  // allocate the mean image
  mMeanImage = new RealImage(mImSize);
  mMeanImage->fill(0.f);
  mMeanImage->setOrigin(mImOrigin);
  mMeanImage->setSpacing(mImSpacing);
  mJacDetSum = new RealImage();
  mJacDetSum->setOrigin(mImOrigin);
  mJacDetSum->setSpacing(mImSpacing);

  if(mNNodes > 1){
    mMPIBuff = new RealImage();
  }

  if(mParam.DoPCAStep() && mParam.UsePerVoxelMedian()){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, cannot DoPCAStep AND UsePerVoxelMedian");
  }

  if(mParam.DoPCAStep()){
    // allocate PCA matrix
    mImagePCA = new ImagePCA(mNImages);
    mImagePCA->SetNComponents(mParam.NPCAComponents());
    mImagePCA->SetNPowerIters(mParam.NPowerPCAIters());
  }

  // set up the deformation data
  for(unsigned int i=0; i<mNImages; i++){

    RealImage *mean;
    if(mParam.DoPCAStep()){
      mIndividualMeans.push_back(new RealImage(*mMeanImage));
      mean = mIndividualMeans.back();
    }else{
      mean = mMeanImage;
    }

    DeformationDataType *curDefData = 
      new DeformationDataType(mImageSet.GetImage(i), 
			      mean, 
			      mImageSet.GetWeight(i), 
			      mParam);
    
    if(mParam.DoPCAStep()){
      curDefData->SaveDefToMean(true, mean);
    }
    if(mParam.UsePerVoxelMedian()){
      curDefData->SaveDefToMean(true);
      mIndividualMeans.push_back(curDefData->GetDefToMean());
    }

    // use image name as identifier
    std::string path, nameBase, nameExt;
    ApplicationUtils::SplitName(mImageSet.GetImageName(i).c_str(), path, nameBase, nameExt);
    curDefData->SetName(nameBase);

    mDeformationData.push_back(curDefData);
  }

  if(mParam.WriteInputImages()){
    this->WriteInputImages();
  }

  mNThreads = AtlasManagerType::CheckNumThreads(nThreads,mNImages);

  if(pthread_barrier_init(&mBarrier, NULL, mNThreads)){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, unable to initialize pthreads barrier");
  }

  //
  // Divide up images among threads
  //
  int bid = 0;
  int nLocalImgs = 0;
  // set up the threads
  for(unsigned int threadIdx=0; threadIdx<mNThreads; threadIdx++)
    {
      ApplicationUtils::Distribute(mNImages, mNThreads, threadIdx, bid, nLocalImgs);
      LOGNODE(logINFO) << "thread: " << threadIdx 
		       << ", bid: " << bid 
		       << ", nLocalImgs = " << nLocalImgs 
		       << std::endl;
      std::vector<DeformationDataType*> localDefData(mDeformationData.begin() + bid, 
						     mDeformationData.begin() + bid + nLocalImgs);
      LOGNODE(logDEBUG) << "localDefData.size(): " << localDefData.size() << std::endl;
      mAtlasThread.push_back(new AtlasThreadType(localDefData,
						 mParam,
						 *this,
						 mMeanImage, 
						 mNodeId, mNNodes, 
						 threadIdx, mNThreads, 
						 mNTotalImages));
    }
  
}

template<class AtlasManagerType>
LDMMAtlasBuilder<AtlasManagerType>::
~LDMMAtlasBuilder()
{
  for(unsigned int i=0; i<mNThreads; i++){
    delete mAtlasThread[i];
  }

  for(unsigned int i=0; i<mNImages; i++){
    delete mDeformationData[i];
  }

  delete mScaleManager;
}

template<class AtlasManagerType>
void 
LDMMAtlasBuilder<AtlasManagerType>::
BuildAtlas()
{

  // just start the threads...
  std::vector<pthread_t> threads(mNThreads);
  for(unsigned int threadIdx=0;threadIdx<mNThreads;threadIdx++){
    int rv = pthread_create(&threads[threadIdx], NULL,
			    ((PTHREAD_STARTFUNC)&AtlasThreadType::StartThread), 
			    mAtlasThread[threadIdx]);
    if (rv != 0)
      {
	throw AtlasWerksException(__FILE__,__LINE__,"Error creating thread.");
      }
  }
  
  // and wait for threads to complete
  for(unsigned int threadIdx=0;threadIdx<mNThreads;threadIdx++){
    int rv = pthread_join(threads[threadIdx], NULL);
    if (rv != 0)
      {
	throw AtlasWerksException(__FILE__,__LINE__,"Error joining thread.");
      }
  }
  
}

template<class AtlasManagerType>
typename LDMMAtlasBuilder<AtlasManagerType>::DeformationDataType &
LDMMAtlasBuilder<AtlasManagerType>::
GetDeformationData(unsigned int idx)
{
  if(idx >= mDeformationData.size()){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, request for deformation data outside range: " + idx);
  }
  return *mDeformationData[idx];
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
ComputePCA()
{
  // deformed images already stored in mIndividualMeans
  if(mParam.DoMeanSubtraction()){
    for(unsigned int imIdx=0; imIdx<mNImages; imIdx++){
      RealImage &defIm = *mIndividualMeans[imIdx];
      defIm.pointwiseSubtract(*mMeanImage);
    }
  }

  // compute using power method
  mImagePCA->ComputePCAPower(mIndividualMeans);

  if(mParam.DoMeanSubtraction()){
    // add back mean
    for(unsigned int imIdx=0; imIdx<mNImages; imIdx++){
      RealImage &defIm = *mIndividualMeans[imIdx];
      defIm.pointwiseAdd(*mMeanImage);
    }
  }
}

template<class AtlasManagerType>
void 
LDMMAtlasBuilder<AtlasManagerType>::
ComputeMedian()
{
  // deformed images already stored in mIndividualMeans
  unsigned int medianIdx = (mNTotalImages % 2) == 0 ? mNTotalImages/2-1 : mNTotalImages/2;
  LOGNODE(logDEBUG2) << "median index: " << medianIdx << std::endl;

  // TEST
  //mMeanImage->fill(0.f);
  // END TEST
  
  MedianCalculator::Select(mIndividualMeans,
			   medianIdx,
			   *mMeanImage);
}

template<class AtlasManagerType>
void 
LDMMAtlasBuilder<AtlasManagerType>::
SumAcrossNodes()
{
#ifdef MPI_ENABLED

  if(mNNodes > 1){
    
    // compute the total sum of all image on the cluster
    // use the size of the average as the standard
    int nVox = mMeanImage->getNumElements();
    MPI_Allreduce(mMeanImage->getDataPointer(), mMPIBuff->getDataPointer(),
		  nVox, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    *mMeanImage = *mMPIBuff;
    
    // compute total JacDetSum
    MPI_Allreduce(mJacDetSum->getDataPointer(), mMPIBuff->getDataPointer(),
		  nVox, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    *mJacDetSum = *mMPIBuff;
    
    // sum energy across nodes
    unsigned int buffSz = mCurIterEnergy.GetSerialSize();

    Real sendBuff[buffSz];
    Real rtnBuff[buffSz];

    mCurIterEnergy.Serialize(sendBuff);
    MPI_Allreduce(sendBuff, rtnBuff, buffSz, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    mCurIterEnergy.Unserialize(rtnBuff);

  }
#endif
}

// called from individual threads, performs thread synchronization
template<class AtlasManagerType>
void 
LDMMAtlasBuilder<AtlasManagerType>::
ComputeMean()
{
  // Thread synchronization, allow only one thread to do the processing
  int barrier_val = pthread_barrier_wait(&mBarrier);
  if(barrier_val == 0){
    // have all but one thread wait
    pthread_barrier_wait(&mBarrier);
  }else if(barrier_val == PTHREAD_BARRIER_SERIAL_THREAD){
    // single thread computes mean,
    // images already scaled so just sum up node means

    // TEST
    LOGNODETHREAD(logDEBUG2) << "Begin compute mean";
    // END TEST
    
    // create the weighted average for all images on this node
    *mMeanImage = *mAtlasThread[0]->GetMean();
    *mJacDetSum = *mAtlasThread[0]->GetJacDetSum();
    mCurIterEnergy = mAtlasThread[0]->GetEnergy();
    // TEST
    // float curMeanNorm = ImageUtils::l2NormSqr(*mMeanImage);
    // LOGNODETHREAD(logDEBUG2) << "local mean norm is " << curMeanNorm;
    // float curJacDetSumNorm = ImageUtils::l2NormSqr(*mJacDetSum);
    // LOGNODETHREAD(logDEBUG2) << "local jac det sum norm is " << curJacDetSumNorm;
    // END TEST

    for(unsigned int threadIdx=1; threadIdx<mNThreads; threadIdx++){

      mMeanImage->pointwiseAdd(*mAtlasThread[threadIdx]->GetMean());
      mJacDetSum->pointwiseAdd(*mAtlasThread[threadIdx]->GetJacDetSum());
      mCurIterEnergy += mAtlasThread[threadIdx]->GetEnergy();

      // TEST
      // curMeanNorm = ImageUtils::l2NormSqr(*mAtlasThread[threadIdx]->GetMean());
      // LOGNODETHREAD(logDEBUG2) << "local mean norm is " << curMeanNorm;
      // curJacDetSumNorm = ImageUtils::l2NormSqr(*mAtlasThread[threadIdx]->GetJacDetSum());
      // LOGNODETHREAD(logDEBUG2) << "local jac det sum norm is " << curJacDetSumNorm;
      // END TEST
    }

    // TEST
    // curMeanNorm = ImageUtils::l2NormSqr(*mMeanImage);
    // LOGNODETHREAD(logDEBUG2) << "single node mean norm is " << curMeanNorm;
    // curJacDetSumNorm = ImageUtils::l2NormSqr(*mJacDetSum);
    // LOGNODETHREAD(logDEBUG2) << "single node jac det sum norm is " << curJacDetSumNorm;
    // END TEST

    // if built with MPI support, this will sum the average images
    // across all nodes
    SumAcrossNodes();

    // TEST
    // curMeanNorm = ImageUtils::l2NormSqr(*mMeanImage);
    // LOGNODETHREAD(logDEBUG2) << "cross-node mean norm is " << curMeanNorm;
    // curJacDetSumNorm = ImageUtils::l2NormSqr(*mJacDetSum);
    // LOGNODETHREAD(logDEBUG2) << "cross-node jac det sum norm is " << curJacDetSumNorm;
    // END TEST

    if(mParam.JacobianScale()){
      mMeanImage->pointwiseDivideBy(*mJacDetSum);
    }

    if(mParam.UsePerVoxelMedian()){
      this->ComputeMedian();
    }

    if(mParam.DoPCAStep()){
      LOGNODETHREAD(logINFO) << "Computing PCA";
      ComputePCA();
      LOGNODETHREAD(logDEBUG) << "Done computing PCA";
    }

    // allow all threads to proceed
    pthread_barrier_wait(&mBarrier);

    // TEST
    // LOGNODETHREAD(logDEBUG2) << "End compute mean, mean image spacing is " << mMeanImage->getSpacing();
    // END TEST

  }else{
    throw AtlasWerksException(__FILE__, __LINE__, "Error, unknown value returned from pthread_barrier_wait");
  }
}

template<class AtlasManagerType>
void 
LDMMAtlasBuilder<AtlasManagerType>::
SetScaleLevel(int scaleLevel)
{
  // Thread synchronization, allow only one thread to do the processing
  int barrier_val = pthread_barrier_wait(&mBarrier);
  if(barrier_val == 0){
    // have all but one thread wait
    pthread_barrier_wait(&mBarrier);
  }else if(barrier_val == PTHREAD_BARRIER_SERIAL_THREAD){
    // single thread updates scale level
    
    mScaleManager->SetScaleLevel(scaleLevel);
    
    SizeType curSize = mScaleManager->CurScaleSize();
    SpacingType curSpacing = mScaleManager->CurScaleSpacing();

    mMeanImage->resize(curSize);
    mMeanImage->setSpacing(curSpacing);
    if(mNNodes > 1){
      mMPIBuff->resize(curSize);
      mMPIBuff->setSpacing(curSpacing);
    }

    // TEST

    // this is after deformation data has been initialized, but before
    // SetScaleLevel has been called on it

    // RealImage testIm;
    // LOGNODE(logDEBUG2) << "Writing test images (before setscalelevel)";
    // for(int imIdx=0;imIdx<mNImages;imIdx++){
    //   mDeformationData[imIdx]->GetI1(testIm);
    //   std::string fname = 
    // 	StringUtils::strPrintf("%sMedianTestOrigFromDeviceBeforeSetScaleLevel.%s", 
    // 			       mDeformationData[imIdx]->GetName().c_str(), 
    // 			       mParam.OutputSuffix().c_str());
    //   ApplicationUtils::SaveImageITK(fname.c_str(), testIm);
    //   fname = 
    // 	StringUtils::strPrintf("%sMedianTestOrigFromHostBeforeSetScaleLevel.%s", 
    // 			       mDeformationData[imIdx]->GetName().c_str(), 
    // 			       mParam.OutputSuffix().c_str());
    //   ApplicationUtils::SaveImageITK(fname.c_str(), mDeformationData[imIdx]->I1());
    // }
    // END TEST

    if(mParam.DoPCAStep()){
      mImagePCA->SetImageSize(curSize);
      // start with normalized random vectors for power method
      mImagePCA->RandomizeComponents();

      
      for(unsigned int imIdx=0;imIdx < mNImages; ++imIdx){
	mIndividualMeans[imIdx]->resize(curSize);
	mIndividualMeans[imIdx]->setSpacing(curSpacing);
      }
    }

    // allow all threads to proceed
    pthread_barrier_wait(&mBarrier);
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, "Error, unknown value returned from pthread_barrier_wait");
  }
}

template<class AtlasManagerType>
void 
LDMMAtlasBuilder<AtlasManagerType>::
ComputeWeights()
{
  // Thread synchronization, allow only one thread to do the processing
  int barrier_val = pthread_barrier_wait(&mBarrier);
  if(barrier_val == 0){
    // have all but one thread wait
    pthread_barrier_wait(&mBarrier);
  }else if(barrier_val == PTHREAD_BARRIER_SERIAL_THREAD){

    LOGNODETHREAD(logDEBUG2) << "Builder computing weights";

    static Real *distances = NULL;
    if(!distances) distances = new Real[mNImages];
    unsigned int curImIdx = 0;
    // collect distances from all threads
    for(unsigned int threadIdx=0; threadIdx<mNThreads; threadIdx++){
      unsigned int curNImages = mAtlasThread[threadIdx]->GetNImages();
      memcpy(&distances[curImIdx], 
	     mAtlasThread[threadIdx]->GetDistances(),
	     curNImages*sizeof(Real));
      curImIdx += curNImages;
    }
    unsigned int k = mParam.TrimmedMeanSize();
    LOGNODETHREAD(logDEBUG3) << "Running Select() for " << mNImages 
			     << " local and " << mNTotalImages 
			     << " total images";
    Real kVal = MedianCalculator::Select(distances, 
					 mNImages, 
					 k);
    LOGNODETHREAD(logDEBUG3) << "Done running Select()";
    // compute set of images to be used (weight = 1.0)
    Real wSum = 0.f;
    unsigned int selectCount = 0;
    unsigned int imNodeIdx = 0;
    for(unsigned int threadIdx=0; threadIdx<mNThreads; threadIdx++){
      Real *curWeights = mAtlasThread[threadIdx]->GetTMWeights();
      unsigned int curNImages = mAtlasThread[threadIdx]->GetNImages();
      for(unsigned int imThreadIdx = 0; imThreadIdx < curNImages; ++imThreadIdx){
	if(distances[imNodeIdx] == 0.f || // first iteration, use all images
	   (distances[imNodeIdx] <= kVal && selectCount < k))
	  {
	    Real w = mDeformationData[imNodeIdx]->Weight();
	    curWeights[imThreadIdx] = w;
	    wSum += w;
	    selectCount++;
	    mDeformationData[imNodeIdx]->SetUsedInMeanCalc(true);
	  }
	else
	  {
	    curWeights[imThreadIdx] = 0.0;
	    mDeformationData[imNodeIdx]->SetUsedInMeanCalc(false);
	  }
	imNodeIdx++;
      }
    }

#ifdef MPI_ENABLED
    if(mNNodes > 1){
      LOGNODE(logDEBUG3) << "Summing weights from non-trimmed images";
      Real wSumAll=0.f;
      MPI_Allreduce(&wSum, &wSumAll, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      wSum = wSumAll;
    }
#endif
    
    for(unsigned int threadIdx=0; threadIdx<mNThreads; threadIdx++){
      Real *curWeights = mAtlasThread[threadIdx]->GetTMWeights();
      unsigned int curNImages = mAtlasThread[threadIdx]->GetNImages();
      for(unsigned int imThreadIdx = 0; 
	  imThreadIdx < curNImages; 
	  ++imThreadIdx)
	{
	  curWeights[imThreadIdx] /= wSum;
	}
    }

    LOGNODE(logDEBUG3) << "Completing ComputeWeights";
    // allow all threads to proceed
    pthread_barrier_wait(&mBarrier);
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, "Error, unknown value returned from pthread_barrier_wait");
  }
}

template<class AtlasManagerType>
void 
LDMMAtlasBuilder<AtlasManagerType>::
FinishIteration(int iteration)
{
  // add iteration event to history
  unsigned int scale = mScaleManager->CurScaleLevel();
  IterationEvent iterEvent(scale, iteration, mCurIterEnergy);
  mEnergyHistory.AddEvent(iterEvent);

  if(mNodeId == 0){
    // print out energy
    LOGNODE(logINFO) 
      << "Scale " << scale+1 << "/" << mNScaleLevels 
      << " Iter " << iteration << " energy: " << mCurIterEnergy;
  }
}

// template<class AtlasManagerType>
// int
// LDMMAtlasBuilder<AtlasManagerType>::
// LocalToGlobalMapping(int localIdx)
// {
//   int nodeBId,nNodeImages;
//   ApplicationUtils::Distribute(mNTotalImages, mNNodes, mNodeId, nodeBId, nNodeImages);
  
//   assert(nNodeImages == (int)mNImages);
  
//   return nodeBId + localIdx;
// }

template<class AtlasManagerType>
void 
LDMMAtlasBuilder<AtlasManagerType>::
BeginScaleLevel(int scaleLevel)
{
  std::string title = StringUtils::strPrintf("ScaleLevel%dInitial", scaleLevel);
  if(mParam.WriteInitialScaleLevelMean()){
    LOGNODE(logDEBUG1) << "Writing initial mean image";
    this->WriteMeanImage(title);
  }
}

template<class AtlasManagerType>
void 
LDMMAtlasBuilder<AtlasManagerType>::
FinishScaleLevel(int scaleLevel)
{
  std::string title = StringUtils::strPrintf("ScaleLevel%dFinal", scaleLevel);
  if(mParam.WriteFinalScaleLevelMean()){
    this->WriteMeanImage(title);
  }
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteMeanImage(std::string title)
{
  char fname[1024];

  if(mNodeId == 0){
    std::cout << "Writing mean image." << std::endl;
    sprintf(fname, "%s%sMeanImage.%s", mParam.OutputPrefix().c_str(), title.c_str(), mParam.OutputSuffix().c_str());
    ApplicationUtils::SaveImageITK(fname, *mMeanImage);
  }
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteInputImages(std::string title)
{
  std::cout << "Writing " << mNImages << " input images:" << std::endl;
  for(unsigned int i=0;i < mNImages;i++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sOrig.%s", 
			     mParam.OutputPrefix().c_str(), 
			     GetDeformationData(i).GetName().c_str(), 
			     title.c_str(), 
			     mParam.OutputSuffix().c_str());
    ApplicationUtils::SaveImageITK(fname.c_str(), *mImageSet.GetImage(i));
    std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteDefImages(std::string title)
{
  std::cout << "Writing " << mNImages << " deformed images:" << std::endl;
  RealImage defImage(mImSize, mImOrigin, mImSpacing);
  for(unsigned int i=0;i < mNImages;i++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sDefToMean.%s", 
			     mParam.OutputPrefix().c_str(), 
			     GetDeformationData(i).GetName().c_str(), 
			     title.c_str(), 
			     mParam.OutputSuffix().c_str());
    GetDeformationData(i).GetI1At0(defImage);
    ApplicationUtils::SaveImageITK(fname.c_str(), defImage);
    std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteDefMean(std::string title)
{
  std::cout << "Writing mean deformed to " << mNImages << " images:" << std::endl;
  RealImage defImage(mImSize, mImOrigin, mImSpacing);
  for(unsigned int i=0;i < mNImages;i++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sDefMean.%s", 
			     mParam.OutputPrefix().c_str(), 
			     GetDeformationData(i).GetName().c_str(), 
			     title.c_str(), 
			     mParam.OutputSuffix().c_str());
    GetDeformationData(i).GetI0At1(defImage);
    ApplicationUtils::SaveImageITK(fname.c_str(), defImage);
    std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteDefFieldsImToMean(std::string title)
{
  std::cout << "Writing " << mNImages << " deformation fields:" << std::endl;
  VectorField h(mImSize);
  for(unsigned int i=0;i<mNImages;i++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sDefFieldImToMean.%s", 
			     mParam.OutputPrefix().c_str(), 
			     GetDeformationData(i).GetName().c_str(), 
			     title.c_str(), 
			     mParam.OutputSuffix().c_str());
    GetDeformationData(i).GetDef1To0(h);
    ApplicationUtils::SaveHFieldITK(fname.c_str(), h, mImOrigin, mImSpacing);
    std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteDefFieldsMeanToIm(std::string title)
{
  std::cout << "Writing " << mNImages << " deformation fields:" << std::endl;
  VectorField h(mImSize);
  for(unsigned int i=0;i<mNImages;i++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sDefFieldMeanToIm.%s", 
			     mParam.OutputPrefix().c_str(), 
			     GetDeformationData(i).GetName().c_str(), 
			     title.c_str(), 
			     mParam.OutputSuffix().c_str());
    GetDeformationData(i).GetDef0To1(h);
    ApplicationUtils::SaveHFieldITK(fname.c_str(), h, mImOrigin, mImSpacing);
    std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteEnergy(std::string title)
{
  RealImage im;
  std::string fname = 
    StringUtils::strPrintf("%s%sEnergy.xml", 
			   mParam.OutputPrefix().c_str(), 
			   title.c_str());
  mEnergyHistory.SaveXML(fname.c_str());
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteIndividualEnergies(std::string title)
{
  for(unsigned int i=0;i<mNImages;i++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sEnergy.xml", 
			     mParam.OutputPrefix().c_str(), 
			     GetDeformationData(i).GetName().c_str(), 
			     title.c_str());
    GetDeformationData(i).GetEnergyHistory().SaveXML(fname.c_str());
  }
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteAlpha0s(std::string title)
{
  std::cout << "Writing " << mNImages << " alpha0s:" << std::endl;
  for(unsigned int i=0;i<mNImages;i++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sAlpha0.%s", 
			     mParam.OutputPrefix().c_str(), 
			     GetDeformationData(i).GetName().c_str(), 
			     title.c_str(), 
			     mParam.OutputSuffix().c_str());
    ApplicationUtils::SaveImageITK(fname.c_str(), GetDeformationData(i).Alpha0());
    std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteIntermediateImages(std::string title)
{
  std::cout << "Writing intermediate images:" << std::endl;
  RealImage defImage(mImSize, mImOrigin, mImSpacing);
  for(unsigned int i=0;i<mNImages;i++){
    for(unsigned int t=0;t<mNTimeSteps;t++){
      std::string fname = 
	StringUtils::strPrintf("%s%s%sDefToMeanTime%02d.%s", 
			       mParam.OutputPrefix().c_str(), 
			       GetDeformationData(i).GetName().c_str(), 
			       title.c_str(), 
			       t,
			       mParam.OutputSuffix().c_str());
      GetDeformationData(i).GetI1AtT(defImage, t);
      ApplicationUtils::SaveImageITK(fname.c_str(), defImage);
    }
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteVelFields(std::string title)
{
  std::cout << "Writing velocity fields:" << std::endl;
  VectorField vField(mImSize);
  for(unsigned int i=0;i<mNImages;i++){
    for(unsigned int t=0;t<mNTimeSteps;t++){
      this->GetDeformationData(i).GetVField(vField, t);
      std::string fname = 
	StringUtils::strPrintf("%s%s%sVelFieldTime%02d.%s",
			       mParam.OutputPrefix().c_str(), 
			       GetDeformationData(i).GetName().c_str(), 
			       title.c_str(), 
			       t,
			       mParam.OutputSuffix().c_str());
      ApplicationUtils::SaveHFieldITK(fname.c_str(), vField, mImOrigin, mImSpacing);
    }
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WritePCAData(std::string title)
{
  std::cout << "Writing PCA data" << std::endl;
  unsigned int nComponents = mParam.NPCAComponents();
  for(unsigned int i=0;i<nComponents;i++){
    std::string fname = 
      StringUtils::strPrintf("%s%sPC%d.%s", 
			     mParam.OutputPrefix().c_str(), 
			     title.c_str(), 
			     i, // # of components
			     mParam.OutputSuffix().c_str());
    ApplicationUtils::SaveImageITK(fname.c_str(), mImagePCA->GetComponent(i));
  }
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WritePCAProjections(std::string title)
{
  std::cout << "Writing PCA projections" << std::endl;
  for(unsigned int i=0;i<mNImages;i++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sProjection.%s", 
			     mParam.OutputPrefix().c_str(), 
			     GetDeformationData(i).GetName().c_str(), 
			     title.c_str(), 
			     mParam.OutputSuffix().c_str());
    ApplicationUtils::SaveImageITK(fname.c_str(), *mIndividualMeans[i]);
  }
}

template<class AtlasManagerType>
void
LDMMAtlasBuilder<AtlasManagerType>::
WriteTrimmedStatus(std::string title)
{
  // eventually we need to write this out as a file, but because the
  // information is distributed across nodes that will have to wait.
  // std::string fname = 
  //   StringUtils::strPrintf("%s%sTrimmedStatus.txt", 
  // 			   mParam.OutputPrefix().c_str(), 
  // 			   title.c_str());
  // std::ofstream statusFile(fname.c_str());
  for(unsigned int i=0;i<mNImages;i++){
    LOGNODE(logINFO) << "Trimmed status: "
		     << GetDeformationData(i).GetName().c_str()
		     << "\t" << GetDeformationData(i).GetUsedInMeanCalc();
  }
  // statusFile.close();
}

template<class AtlasManagerType>
void 
LDMMAtlasBuilder<AtlasManagerType>::
GenerateOutput()
{
  if(mParam.WriteMeanImage()){
    this->WriteMeanImage();
  }
  
  if(mParam.WriteDefImages()){
    this->WriteDefImages();
  }

  if(mParam.WriteDefMean()){
    this->WriteDefMean();
  }

  if(mParam.WriteDefFieldsMeanToIm()){
    this->WriteDefFieldsMeanToIm();
  }

  if(mParam.WriteDefFieldsImToMean()){
    this->WriteDefFieldsImToMean();
  }

  if(mParam.WriteEnergy()){
    this->WriteEnergy();
  }

  if(mParam.WriteIndividualEnergies()){
    this->WriteIndividualEnergies();
  }

  if(mParam.WriteAlpha0s()){
    this->WriteAlpha0s();
  }
  
  if(mParam.WriteIntermediateImages()){
    this->WriteIntermediateImages();
  }
  
  if(mParam.WriteVelocityFields()){
    this->WriteVelFields();
  }

  if(mParam.DoPCAStep()){
    this->WritePCAData();
    this->WritePCAProjections();
  }

  if(mParam.TrimmedMeanSize() > 0){
    this->WriteTrimmedStatus();
  }
}
  

