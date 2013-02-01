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


#include "AffineAtlasBuilderCPU.h"

#ifndef appout
#define appout std::cerr
#endif

/**
 * Initialization for MPI version (multiple nodes)
 */
AffineAtlasBuilderCPU::
AffineAtlasBuilderCPU(unsigned int nodeId, unsigned int nNodes, 
		      unsigned int nTotalImgs,
		      WeightedImageSet &imageSet,
		      unsigned int nThreads, 
		      std::string registrationtype, 
		      unsigned int iterations, bool WriteFinalImages)
  :  m_nodeId(nodeId),
     m_nNodes(nNodes),
     m_nTotalImgs(nTotalImgs),
     m_nThreads(nThreads),
     m_imageSet(imageSet),
     m_iavg(NULL),
     m_Images(NULL)
{
  m_nImages = m_imageSet.NumImages();
  regtype = registrationtype;
  nIterations = iterations;
  WriteTransformedImages = WriteFinalImages;

  m_imSize = m_imageSet.GetImage(0)->getSize();
  m_imOrigin = m_imageSet.GetImage(0)->getOrigin();
  m_imSpacing = m_imageSet.GetImage(0)->getSpacing();

}

AffineAtlasBuilderCPU::
~AffineAtlasBuilderCPU()
{
	delete [] m_Images;
}

void 
AffineAtlasBuilderCPU::
BuildAtlas()
{
  //
  // build atlas at each scale level
  //

	m_Images  = new AffineAtlas::ImageType*[m_nImages];
      	for (int i = 0; i < (int) m_nImages; ++i)
	{
	    m_Images[i] = m_imageSet.GetImage(i);
	}
	appout<<"Node: "<<m_nodeId<<"  Num of Images: "<<m_nImages<<std::endl;

	 m_iavg= new AffineAtlas::ImageType(*m_Images[0]);

       //SumAcrossNodes();
        Array3DUtils::arithmeticSum(m_nImages,(const Array3D<AffineAtlas::VoxelType>** const) m_Images,*m_iavg);
	//m_iavg->scale(m_nImages);
	//Array3DUtils::arithmeticMean(4,m_Images,*m_iavg);
        SumAcrossNodes();
	if(m_nodeId == 0)
	{
	        ImageUtils::writeMETA(*m_iavg,"average");
		
	}
	AffineTransform3D<double> finalTransform[m_nImages];
        for(unsigned int j=0;j<m_nImages;j++)
        {
        	finalTransform[j].eye();
	}

        for(int i=0;i<nIterations;i++)
        {

//              std::cout<<"Iteration : "<<i+1<<std::endl;

                double squarederror=0.0,det[m_nImages],detsum=0,weights[m_nImages];


                for(unsigned int j=0;j<m_nImages;j++)
                {
			AffineTransform3D<double> transform;

                        AffineAtlas abc((Image<AffineAtlas::VoxelType>*)m_Images[j],(Image<AffineAtlas::VoxelType>*) m_iavg,"Affine");

                        transform=abc.registrationTransform;
                        if(!(transform.invert()))
                        {
				std::cout<<"Error: unable to compute inverse... exiting"<<std::endl;
				exit(-1);
			}
			finalTransform[j].applyTransform(transform);

                        std::string imageNameString = "image->overlay affine";
                        //      images[0]=abc._applyAffineTransform(transform,iavg,images[0],imageNameString);

                        squarederror += ImageUtils::squaredError(*m_Images[j],*abc.finalImage);
                        abc._applyAffineTransform(transform,m_iavg,m_Images[j],imageNameString,m_Images[j]);
                        if(i==(nIterations-1))
                        {
                        //writing transformations
                                char transFileName[5];
                                sprintf(transFileName,"%d-%d",m_nodeId+1,j+1);
                                char *fileName = transFileName;
                                try
                                {
                                        finalTransform[j].writePLUNCStyle(fileName);
                                }
                                catch (...)
                                {
                                        std::cout<<"Failed to save matrix"<<std::endl;
                                       // return 0;
					exit(-1);
                                }
				if(WriteTransformedImages)
				{
					ImageUtils::writeMETA(*m_Images[j],transFileName);
				}
                        }
                        det[j] =  transform.determinant();
                        detsum = detsum + det[j];
                        weights[j] = det[j] / m_nTotalImgs;

                        delete abc.finalImage;
                }

                Array3DUtils::weightedArithmeticMean(m_nImages,(const Array3D<AffineAtlas::VoxelType>** const) m_Images, weights,*m_iavg);
                SumAcrossNodesWithJacobian(detsum);
		//Array3DUtils::arithmeticMean(m_nImages,(const Array3D<AffineAtlas::VoxelType>** const) m_Images, *m_iavg);
		//SumAcrossNodes();
		ImageUtils::writeMETA(*m_iavg,"final");
		if(m_nodeId == 0)
			appout<<"Iteration : "<<i+1<<"   SquaredError: "<<squarederror<<std::endl;

        }
	if(m_nodeId == 0)
	{
		ImageUtils::writeMETA(*m_iavg,"final");
	}
} // end CPU atlas building  

void 
AffineAtlasBuilderCPU::
SumAcrossNodesWithJacobian(double detsum)
{
#ifdef MPI_ENABLED
  // compute the total sum of all image on the cluster
  // use the size of the average as the standard
//  mMPIMeanImage = new AffineAtlas::ImageType(*iavg);
	//m_iavg->scale((1/m_nNodes));
  double mpidetsum[1] ;
  MPI_Reduce(&detsum, &mpidetsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  mPIMeanImage = new AffineAtlas::ImageType(*m_Images[0]);
  int nVox = m_iavg->getNumElements();
  MPI_Allreduce(m_iavg->getDataPointer(), mPIMeanImage->getDataPointer(),
                nVox, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  //mPIMeanImage->scale((m_nTotalImgs/(mpidetsum[0])));
  *m_iavg = *mPIMeanImage;
  //m_iavg->scale((((double)1 / (double)m_nTotalImgs)));
  delete mPIMeanImage;
  //delete mMPIMeanImage;
#endif
}

void
AffineAtlasBuilderCPU::
SumAcrossNodes()
{
#ifdef MPI_ENABLED
  // compute the total sum of all image on the cluster
  // use the size of the average as the standard
//  mMPIMeanImage = new AffineAtlas::ImageType(*iavg);
 if(m_nNodes > 1)
 {
       // m_iavg->scale((1/m_nNodes))
//	m_iavg->scale((((double)1 / (double)m_nNodes)));
 }
  mPIMeanImage = new AffineAtlas::ImageType(*m_Images[0]);
  int nVox = m_iavg->getNumElements();
  MPI_Allreduce(m_iavg->getDataPointer(), mPIMeanImage->getDataPointer(),
                nVox, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  *m_iavg = *mPIMeanImage;
  m_iavg->scale((((double)1 / (double)m_nTotalImgs)));
  delete mPIMeanImage;
  //delete mMPIMeanImage;
#endif
}


void 
AffineAtlasBuilderCPU::
GetMeanImage(RealImage &mean)
{
  mean = *m_iavg;
}
  
