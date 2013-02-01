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


#include "GreedyAtlasBuilderCPU.h"

#ifndef appout
#define appout std::cerr
#endif

/**
 * Initialization for MPI version (multiple nodes)
 */
GreedyAtlasBuilderCPU::
GreedyAtlasBuilderCPU(unsigned int nodeId, unsigned int nNodes, 
		      unsigned int nTotalImgs,
		      WeightedImageSet &imageSet,
		      const MultiParam<GreedyAtlasScaleLevelParam> &param,
		      unsigned int nThreads)
  :  m_nodeId(nodeId),
     m_nNodes(nNodes),
     m_nTotalImgs(nTotalImgs),
     m_nThreads(nThreads),
     m_imageSet(imageSet),
     m_scaleLevelParams(param),
     m_iHat(NULL),
     m_scaledImages(NULL),
     m_h(NULL),
     m_hinv(NULL)
{
  m_nImages = m_imageSet.NumImages();
  
  m_imSize = m_imageSet.GetImage(0)->getSize();
  m_imOrigin = m_imageSet.GetImage(0)->getOrigin();
  m_imSpacing = m_imageSet.GetImage(0)->getSpacing();

}

GreedyAtlasBuilderCPU::
~GreedyAtlasBuilderCPU()
{
  //
  // clean up memory
  //
  for (int i = 0; i < (int) m_nImages;++i){
      if(m_h){
	delete m_h[i];
      }
      
      if (m_hinv){
	  delete m_hinv[i];
      }
  }

  if(m_scaledImages){
    delete [] m_scaledImages;
  }

  if(m_h){
    delete [] m_h;
  }

  if (m_hinv){
      delete [] m_hinv;
  }
  
}

void 
GreedyAtlasBuilderCPU::
BuildAtlas()
{
  //
  // build atlas at each scale level
  //

  m_scaledImages  = new RealImage*[m_nImages];
  m_h = new VectorField*[m_nImages];
  m_hinv = NULL;
  
  if (m_computeInverseHFields)
    {
      m_hinv = new VectorField*[m_nImages];  
    }
  Vector3D<float> curSpacing;
  int nScaleLevels =  (int) m_scaleLevelParams.size();
  for (int scale = 0; scale < nScaleLevels; ++scale)
    {
      int f = (int) m_scaleLevelParams[scale].ScaleLevel().DownsampleFactor();
      appout << "Scale: " << f << std::endl;
      Vector3D<unsigned int> curSize = m_imSize / f;
      curSpacing = m_imSpacing * f;
      
      appout << "size = " << curSize 
	     << ", spacing = " << curSpacing
	     << std::endl;
      
      
      //
      // create images for this scale level
      //
      appout << "Downsampling Images...";
      if (f == 1) {
	appout << "Using Actual Images..." << std::endl;
      }
      else {
	appout << "factor=" << f << "...";
      }

      for (int i = 0; i < (int) m_nImages; ++i)
	{
	  if (f == 1) {
	    m_scaledImages[i] = m_imageSet.GetImage(i);
	  }
	  else {
	    m_scaledImages[i] = new RealImage;
	    ImageUtils::gaussianDownsample(*m_imageSet.GetImage(i),
					   *m_scaledImages[i],
					   curSize);
	  }
	}
      //
      // create h fields for this scale level
      //
      appout << "Creating h-fields...";
      if (scale == 0)
	{
	  // start with identity
	  for (int i = 0; i < (int) m_nImages; ++i)
	    {
	      m_h[i] = new VectorField(curSize);
	      if (m_hinv)
		{
		  m_hinv[i] = new VectorField(curSize);
		}
	      
	      // initalize deformation fields to that specified by the
	      // affine transforms
	      if(m_imageSet.HasTransforms())
		{
		  
		  //itk::AffineTransform<float, 3>::Pointer transform = 
		  RealAffineTransform *transform = 	
		    m_imageSet.GetTransform(i);
		  
		  if(m_hinv)
		    initializeHField(m_h[i], m_hinv[i], transform, curSpacing, m_imOrigin);
		  else
		    initializeHField(m_h[i], NULL, transform, curSpacing, m_imOrigin);
		  
		}
	      // Set Transforms to identity
	      else
		{
		  HField3DUtils::setToIdentity(*m_h[i]);
		  if (m_hinv)
		    {
		      HField3DUtils::setToIdentity(*m_hinv[i]);            
		    }
		}
	    }
	}
      else
	{
	  appout << "Upsampling...";
	  // upsample old xforms
	  VectorField tmph(m_h[0]->getSize());
	  for (int i = 0; i < (int) m_nImages; ++i)
	    {
	      tmph = *m_h[i];
	      HField3DUtils::resampleNew(tmph, *m_h[i], curSize, 
					 HField3DUtils::BACKGROUND_STRATEGY_CLAMP);

	      if (m_hinv)
		{
		  tmph = *m_hinv[i];
		  HField3DUtils::resampleNew(tmph, *m_hinv[i], curSize, 
					     HField3DUtils::BACKGROUND_STRATEGY_CLAMP);
		}
	    }
	}
      appout << "DONE" << std::endl;

      //
      // create atlas at this scale level
      //
      appout << "Computing atlas at this scale level..." << std::endl;

      if(m_iHat) delete m_iHat;
      m_iHat = new RealImage(curSize, m_imOrigin, curSpacing);
      
      AtlasBuilder atlasBuilder;
      
      ArithmeticMeanComputationStrategy<AtlasBuilder::VoxelType> meanCalculator;
      meanCalculator.SetNumberOfElements(m_nImages);
      for (unsigned int i = 0; i < m_nImages; ++i)
	{
	  meanCalculator.SetNthWeight(i, m_imageSet.GetWeight(i));
	}
      
      atlasBuilder.SetNumberOfInputImages(m_nImages);
      atlasBuilder.
	SetMeanComputationStrategy((AtlasBuilder::
				    MeanComputationStrategyType*) 
				   &meanCalculator);
      atlasBuilder.SetParams(m_scaleLevelParams[scale]);
      atlasBuilder.SetNumberOfThreads(m_nThreads);
      atlasBuilder.SetLogOutputStream(std::cerr);
      if (m_computeInverseHFields)
	atlasBuilder.SetComputeInverseDeformationsOn();
      else
	atlasBuilder.SetComputeInverseDeformationsOff();
      atlasBuilder.SetAverageImage(m_iHat);
      atlasBuilder.SetScaleLevel(scale);

      // set AtlasBuilder's images and hfields
      for (unsigned int imageIndex = 0; imageIndex < m_nImages; ++imageIndex)
	{
	  atlasBuilder.SetNthInputImage(imageIndex, m_scaledImages[imageIndex]);
	  atlasBuilder.SetNthDeformationField(imageIndex, m_h[imageIndex]);
	  if (m_hinv)
	    {
	      atlasBuilder.SetNthDeformationFieldInverse(imageIndex, m_hinv[imageIndex]);
	    }
	  // sanity check that hField size is the same as image size
	  std::cout << m_h[imageIndex]->getSize() << " = ";
	  std::cout << atlasBuilder.GetNthDeformationField(imageIndex)->getSize();
	  std::cout << std::endl;
	}

      // run atlas building
      atlasBuilder.GenerateAverage();

      //
      // If this is the final scale level, get the deformed images.
      // Note that this overwrites the images in imageSet.
      //
      if(scale == nScaleLevels-1){
 	for (unsigned int imageIndex = 0; imageIndex < m_nImages; ++imageIndex)
 	  {
 	    *m_scaledImages[imageIndex] = *atlasBuilder.GetNthDeformedImage(imageIndex);
 	    m_scaledImages[imageIndex]->setOrigin(m_imOrigin);
 	    m_scaledImages[imageIndex]->setSpacing(curSpacing);
 	  }
      }
      
      if(f != 1){
	//
	// delete scaled images from this scale level, unless
	// downsample factor is 1, in which case these are the images
	// from m_imageSet
	//
	for (int i = 0; i < (int) m_nImages; ++i)
	  {
	    delete m_scaledImages[i];
	  }
      }
    } // end scale iteration
} // end CPU atlas building  

void 
GreedyAtlasBuilderCPU::
invert(RealAffineTransform *invtransform)
{
  std::cout<<"matrix"<<invtransform->matrix<<std::endl;
  std::cout<<"vector"<<invtransform->vector<<std::endl;
  
  if(invtransform->matrix.invert())
    {
      invtransform->vector *= (-1);
      invtransform->vector = invtransform->matrix * invtransform->vector;
      //        std::cout<<"inverse"<<*invtransform<<std::endl;
    }
  
}

/**
 * initialize h and hinv to a vector field representation of affine
 * and inv(affine), respectively
 */
void 
GreedyAtlasBuilderCPU::
initializeHField(VectorField* h,
		 VectorField* hinv,
		 //itk::AffineTransform<float, 3>::Pointer affine,
		 RealAffineTransform *affine,
		 Vector3D<float> spacing,
		 Vector3D<float> origin)
{
  Vector3D<unsigned int> size = h->getSize();
  //  itk::AffineTransform<float, 3>::Pointer invtransform = itk::AffineTransform<float, 3>::New();
  //  invtransform->SetCenter(affine->GetCenter());
  RealAffineTransform *invtransform = new RealAffineTransform();
  *invtransform=*affine;
  //  affine->GetInverse(invtransform);
  invert(invtransform);
  /*
    std::cout<<"matrix"<<invtransform->matrix<<std::endl;
    std::cout<<"vector"<<invtransform->vector<<std::endl;
    
    if(invtransform->matrix.invert())
    {
    invtransform->vector *= (-1);
    invtransform->vector = invtransform->matrix * invtransform->vector;
    std::cout<<"inverse"<<*invtransform<<std::endl;
    }
  */
  std::cout<<"Transform is"<<*affine<<"\n";
  std::cout<<"Inverse transform is"<<*invtransform<<"\n";
  
  for (unsigned int z = 0; z < size.z; ++z) 
    {
      for (unsigned int y = 0; y < size.y; ++y) 
	{
	  for (unsigned int x = 0; x < size.x; ++x) 
	    {
	      //itk::Point<float, 3> p, tp, tinvp;
	      Vector3D<float> p,tp, tinvp;
	      p[0] = x * spacing[0] + origin[0];
	      p[1] = y * spacing[1] + origin[1];
	      p[2] = z * spacing[2] + origin[2];
	      //tp = affine->TransformPoint(p);
	      //tinvp = invtransform->TransformPoint(p);
	      affine->transformVector(p,tp);
	      invtransform->transformVector(p,tinvp);
	      
	      (*h)(x,y,z).set((tp[0] - origin[0]) /spacing[0],
			      (tp[1] - origin[1]) /spacing[1],
			      (tp[2] - origin[2]) /spacing[2]);
	      
	      if(hinv != NULL)
		{
		  (*hinv)(x,y,z).set((tinvp[0] - origin[0]) /spacing[0],
				     (tinvp[1] - origin[1]) /spacing[1],
				     (tinvp[2] - origin[2]) /spacing[2]);
		}
	    }
	}
    }
  
}

void 
GreedyAtlasBuilderCPU::
GetDeformedImage(int imIdx, RealImage &im)
{
  if(imIdx >= 0 && imIdx < (int)m_nImages){
    im = *m_scaledImages[imIdx];
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "Error, requesting invalid deformed image");
  }
}

void 
GreedyAtlasBuilderCPU::
GetHField(int imIdx, VectorField &vf)
{
  if(imIdx >= 0 && imIdx < (int)m_nImages){
    vf = *m_h[imIdx];
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "Error, requesting invalid deformation");
  }
}

void 
GreedyAtlasBuilderCPU::
GetInvHField(int imIdx, VectorField &vf)
{
  if(imIdx >= 0 && imIdx < (int)m_nImages){
    vf = *m_hinv[imIdx];
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "Error, requesting invalid inverse deformation");
  }
}

void 
GreedyAtlasBuilderCPU::
GetMeanImage(RealImage &mean)
{
  mean = *m_iHat;
}
  
