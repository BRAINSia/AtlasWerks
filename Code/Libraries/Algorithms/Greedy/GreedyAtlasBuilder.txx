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


// #include "GreedyAtlasBuilder.h"

#include "log.h"
#include <cmath>

template<class AtlasManagerType>
GreedyAtlasBuilder<AtlasManagerType>::
GreedyAtlasBuilder(const WeightedImageSet &imageSet,
		    GreedyAtlasParam &param,
		    unsigned int nThreads,
		    unsigned int nodeId,
		    unsigned int nNodes,
		    unsigned int nTotalImages)
  : AtlasBuilderInterface(imageSet, nThreads, nodeId, nNodes, nTotalImages),
    mParam(param),
    mMeanImage(NULL),
    mMPIBuff(NULL)
{
  mScaleManager = new MultiscaleManagerType(mImSize, mImSpacing, mImOrigin, mParam);
  mScaleManager->SetScaleVectorFields(true);
  mNScaleLevels = mScaleManager->NumberOfScaleLevels();
  
  // allocate the mean image
  mMeanImage = new RealImage(mImSize);
  mMeanImage->fill(0.f);
  mMeanImage->setOrigin(mImOrigin);
  mMeanImage->setSpacing(mImSpacing);

  if(mNNodes > 1){
    mMPIBuff = new RealImage();
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
      mean = new RealImage(*mMeanImage);
      mIndividualMeans.push_back(mean);
    }else{
      mean = mMeanImage;
    }
    
    DeformationDataType *curDefData = 
      new DeformationDataType(mImageSet.GetImage(i), mean, mImageSet.GetWeight(i), mParam);


    if(mParam.DoPCAStep()){
      curDefData->SaveDefToMean(true, mean);
    }

    // use image name as identifier
    std::string path, nameBase, nameExt;
    ApplicationUtils::SplitName(mImageSet.GetImageName(i).c_str(), path, nameBase, nameExt);
    curDefData->SetName(nameBase);

    // set whether to compute inverse field
    if(mParam.WriteDefMean() || mParam.WriteDefFieldsImToMean()){
      curDefData->ComputeInverseHField(true);
    }

    // set transform if necessary
    if(mImageSet.HasTransforms())
      {
	const RealAffineTransform *transform = 	
	  mImageSet.GetTransform(i);
	curDefData->SetInitialAffine(*transform);
      }

    // add this deformation data to the list
    mDeformationData.push_back(curDefData);
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
GreedyAtlasBuilder<AtlasManagerType>::
~GreedyAtlasBuilder()
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
GreedyAtlasBuilder<AtlasManagerType>::
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
typename GreedyAtlasBuilder<AtlasManagerType>::DeformationDataType &
GreedyAtlasBuilder<AtlasManagerType>::
GetDeformationData(unsigned int idx)
{
  if(idx >= mDeformationData.size()){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, request for deformation data outside range: " + idx);
  }
  return *mDeformationData[idx];
}

template<class AtlasManagerType>
void 
GreedyAtlasBuilder<AtlasManagerType>::
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
    
    unsigned int buffSz = mCurIterEnergy.GetSerialSize();
    Real sendBuff[buffSz];
    Real rtnBuff[buffSz];
    
    mCurIterEnergy.Serialize(sendBuff);
    MPI_Allreduce(sendBuff, rtnBuff, buffSz, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    mCurIterEnergy.Unserialize(rtnBuff);
  }
#endif
}

template<class AtlasManagerType>
void
GreedyAtlasBuilder<AtlasManagerType>::
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
  //mImagePCA->ComputePCA(mIndividualMeans);

  // add back mean
  if(mParam.DoMeanSubtraction()){
    for(unsigned int imIdx=0; imIdx<mNImages; imIdx++){
      RealImage &defIm = *mIndividualMeans[imIdx];
      defIm.pointwiseAdd(*mMeanImage);
    }
  }
}

// called from individual threads, performs thread synchronization
template<class AtlasManagerType>
void 
GreedyAtlasBuilder<AtlasManagerType>::
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
    LOGNODETHREAD(logDEBUG2) << "Begin compute mean, mean image spacing is " << mMeanImage->getSpacing();
    // END TEST
    
    // create the weighted average for all images on this node
    *mMeanImage = *mAtlasThread[0]->GetMean();
    mCurIterEnergy = mAtlasThread[0]->GetEnergy();
    // TEST
    float curMeanNorm = ImageUtils::l2NormSqr(*mMeanImage);
    LOGNODETHREAD(logDEBUG2) << "local mean norm is " << curMeanNorm;
    // END TEST
    for(unsigned int threadIdx=1; threadIdx<mNThreads; threadIdx++){
      mMeanImage->pointwiseAdd(*mAtlasThread[threadIdx]->GetMean());
      mCurIterEnergy += mAtlasThread[threadIdx]->GetEnergy();
      // TEST
      curMeanNorm = ImageUtils::l2NormSqr(*mAtlasThread[threadIdx]->GetMean());
      LOGNODETHREAD(logDEBUG2) << "local mean norm is " << curMeanNorm;
      // END TEST
    }
    
    // TEST
    curMeanNorm = ImageUtils::l2NormSqr(*mMeanImage);
    LOGNODETHREAD(logDEBUG2) << "single node mean norm is " << curMeanNorm;
    // END TEST

    // if built with MPI support, this will sum the average images
    // across all nodes
    SumAcrossNodes();

    // TEST
    curMeanNorm = ImageUtils::l2NormSqr(*mMeanImage);
    LOGNODETHREAD(logDEBUG2) << "cross-node mean norm is " << curMeanNorm;
    // END TEST

    if(mParam.DoPCAStep()){
      LOGNODETHREAD(logINFO) << "Computing PCA";
      ComputePCA();
      LOGNODETHREAD(logDEBUG) << "Done computing PCA";
    }

    // allow all threads to proceed
    pthread_barrier_wait(&mBarrier);

    // TEST
    LOGNODETHREAD(logDEBUG2) << "End compute mean, mean image spacing is " << mMeanImage->getSpacing();
    // END TEST

  }else{
    throw AtlasWerksException(__FILE__, __LINE__, "Error, unknown value returned from pthread_barrier_wait");
  }
}

template<class AtlasManagerType>
void 
GreedyAtlasBuilder<AtlasManagerType>::
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
    if(mParam.DoPCAStep()){
      mImagePCA->SetImageSize(curSize);
      // start with normalized random vectors for power method
      mImagePCA->RandomizeComponents();
    }
    // allow all threads to proceed
    pthread_barrier_wait(&mBarrier);
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, "Error, unknown value returned from pthread_barrier_wait");
  }
}

template<class AtlasManagerType>
void 
GreedyAtlasBuilder<AtlasManagerType>::
FinishIteration(int iteration)
{
  // add iteration event to history
  unsigned int scale = mScaleManager->CurScaleLevel();
  IterationEvent iterEvent(scale, iteration, mCurIterEnergy);
  mEnergyHistory.AddEvent(iterEvent);

  // print out energy
  if(mNodeId == 0){
    LOGNODE(logINFO) 
      << "Scale " << scale+1 << "/" << mScaleManager->NumberOfScaleLevels()
      << " Iter " << iteration << " energy: " << mCurIterEnergy;
  }
}

template<class AtlasManagerType>
int
GreedyAtlasBuilder<AtlasManagerType>::
LocalToGlobalMapping(int localIdx)
{
  int nodeBId,nNodeImages;
  ApplicationUtils::Distribute(mNTotalImages, mNNodes, mNodeId, nodeBId, nNodeImages);
  
  assert(nNodeImages == (int)mNImages);
  
  return nodeBId + localIdx;
}

template<class AtlasManagerType>
void 
GreedyAtlasBuilder<AtlasManagerType>::
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
GreedyAtlasBuilder<AtlasManagerType>::
FinishScaleLevel(int scaleLevel)
{
  std::string title = StringUtils::strPrintf("ScaleLevel%dFinal", scaleLevel);
  if(mParam.WriteFinalScaleLevelMean()){
    this->WriteMeanImage(title);
  }
}

template<class AtlasManagerType>
void
GreedyAtlasBuilder<AtlasManagerType>::
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
GreedyAtlasBuilder<AtlasManagerType>::
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
    GetDeformationData(i).GetI0At1(defImage);
    ApplicationUtils::SaveImageITK(fname.c_str(), defImage);
    std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
GreedyAtlasBuilder<AtlasManagerType>::
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
    GetDeformationData(i).GetI1At0(defImage);
    ApplicationUtils::SaveImageITK(fname.c_str(), defImage);
    std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
GreedyAtlasBuilder<AtlasManagerType>::
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
    GetDeformationData(i).GetDef1To0(h);
    ApplicationUtils::SaveHFieldITK(fname.c_str(), h, mImOrigin, mImSpacing);
    std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
GreedyAtlasBuilder<AtlasManagerType>::
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
    GetDeformationData(i).GetDef0To1(h);
    ApplicationUtils::SaveHFieldITK(fname.c_str(), h, mImOrigin, mImSpacing);
    std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

template<class AtlasManagerType>
void
GreedyAtlasBuilder<AtlasManagerType>::
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
GreedyAtlasBuilder<AtlasManagerType>::
WriteIndividualEnergies(std::string title)
{
  for(unsigned int i=0;i<mNImages;i++){
    std::string fname = 
      StringUtils::strPrintf("%s%s%sEnergy.xml", 
			     mParam.OutputPrefix().c_str(), 
			     GetDeformationData(i).GetName().c_str(), 
			     title.c_str());
    mDeformationData.GetEnergyHistory().SaveXML(fname.c_str());
  }
}

template<class AtlasManagerType>
void
GreedyAtlasBuilder<AtlasManagerType>::
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
GreedyAtlasBuilder<AtlasManagerType>::
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
GreedyAtlasBuilder<AtlasManagerType>::
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

  if(mParam.WriteEnergy()){
    this->WriteEnergy();
  }

  if(mParam.WriteIndividualEnergies()){
    this->WriteEnergy();
  }

  if(mParam.WriteDefFieldsImToMean()){
    this->WriteDefFieldsImToMean();
  }

  if(mParam.DoPCAStep()){
    this->WritePCAData();
    this->WritePCAProjections();
  }

}

