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

/*
This program generates registration transforms for each Image which will be useful for atlasing.
Usage :
AffineAtlas -f <parameters.xml>

To generate a sample xml file use
AffineAtlas -g <parameters.xml>

Give the names of input Images in the xml, and change the number of Iterations if necessary
*/

#include<stdio.h>
#include<AffineAtlas.h>
#include "CmdLineParser.h"
#include<string.h>
#include "ImagePreprocessor.h"
#include<Array3DUtils.h>
#include "WeightedImageSet.h"

typedef float T;
//  void arithmeticMean(unsigned int , const Array3D* const* , Array3D& );


class AffineAtlasParamFile : public CompoundParam {
public:
	AffineAtlasParamFile()
	: CompoundParam("ParameterFile", "top-level node", PARAM_REQUIRED)
	{
                this->AddChild(WeightedImageSetParam("WeightedImageSet"));
                this->AddChild(ImagePreprocessorParam("ImagePreprocessor"));
		this->AddChild(ValueParam<std::string>("RegistrationType", "Resistration Type: Affine, Translation, Rigid",PARAM_REQUIRED, "Affine"));
		this->AddChild(ValueParam<unsigned int>("nIterations", "number of Iterations", PARAM_REQUIRED, 50));
		this->AddChild(ValueParam<bool>("WriteTransformedImages", "If the value is true the final transformed images are written", PARAM_COMMON, false));
        }

        ParamAccessorMacro(WeightedImageSetParam, WeightedImageSet)
        ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor)
        ValueParamAccessorMacro(std::string, RegistrationType)
	ValueParamAccessorMacro(unsigned int, nIterations)
	ValueParamAccessorMacro(bool, WriteTransformedImages)
        CopyFunctionMacro(AffineAtlasParamFile)

};

int main(int argc, char ** argv)
{



        AffineAtlasParamFile pf;

        CmdLineParser parser(pf);

        try{
                parser.Parse(argc,argv);
        }catch(ParamException e){
                std::cerr << "Error parsing arguments:" << std::endl;
                std::cerr << "   " << e.what() << std::endl;
                return EXIT_FAILURE;
        }
//
// load images
//
        WeightedImageSet imageSet(pf.WeightedImageSet());
//        InputImagePreprocessor preprocessor(pf.WeightedImageSet());
// verbose load
        // preprocess
        ImagePreprocessor preprocessor(pf.ImagePreprocessor());
        {
	        std::vector<RealImage*> imVec = imageSet.GetImageVec();
        	std::vector<std::string> imNames = imageSet.GetImageNameVec();
	        preprocessor.Process(imVec, imNames);
        }

  //      preprocessor.Load(true);
	imageSet.Load(true);

// copy out loaded, preprocessed images
        unsigned int numImages = imageSet.NumImages();
        AffineAtlas::ImageType** images = new AffineAtlas::ImageType*[numImages];
//      std::string paths[numImages];
//      double *imageWeights = new double[numImages];
        for(unsigned int i=0;i<numImages;i++){
//              paths[i] = preprocessor.
        //        images[i] = preprocessor.GetImage(i);
		images[i] = imageSet.GetImage(i);
        //      imageWeights[i] = preprocessor.GetWeight(i);
        }

//      parser.Parse(argc,argv);

        std::string regtype = pf.RegistrationType();
	int  nIterations = pf.nIterations();


        parser.GenerateFile("AffineParsedOutput.xml");



	AffineAtlas::ImageType* iavg= new AffineAtlas::ImageType(*images[0]);
	Array3DUtils::arithmeticMean(numImages,(const Array3D<AffineAtlas::VoxelType>** const) images,*iavg);
	ImageUtils::writeMETA(*iavg,"average");

        AffineTransform3D<double> finalTransform[numImages];
        for(unsigned int j=0;j<numImages;j++)
        {
	        finalTransform[j].eye();
        }

	for(int i=0;i<nIterations;i++)
	{

//		std::cout<<"Iteration : "<<i+1<<std::endl;
	
		double squarederror=0.0;
		double det[numImages],detsum=0,weights[numImages];
		for(unsigned int j=0;j<numImages;j++)
		{

			AffineAtlas abc((Image<AffineAtlas::VoxelType>*)images[j],(Image<AffineAtlas::VoxelType>*) iavg, regtype);
			AffineTransform3D<double> transform;
			transform=abc.registrationTransform;
			if(transform.invert())
			{
				//std::cout<<"Inverse computed ..."<<std::endl;
				finalTransform[j].applyTransform(transform);
			}
			else
			{
				std::cout<<"Unable to find the matrix inverse"<<std::endl;
				exit(-1);
			}
		        std::string imageNameString = "image->overlay affine";	
			//	images[0]=abc._applyAffineTransform(transform,iavg,images[0],imageNameString);

			squarederror += ImageUtils::squaredError(*images[j],*abc.finalImage);
			//	images[j] = abc._applyAffineTransform(transform,iavg,images[j],imageNameString);
			abc._applyAffineTransform(transform,iavg,images[j],imageNameString,images[j]);

			//	iavg = abc._applyAffineTransform(abc.registrationTransform,images[j],iavg,imageNameString);
			if(i==(nIterations-1))
			{
			//writing transformations
				char transFileName[5];
	        	        sprintf(transFileName,"%d",j+1);
				char *fileName = transFileName;
				try
				{
					finalTransform[j].writePLUNCStyle(fileName);
				}
				catch (...)
				{
					std::cout<<"Failed to save matrix"<<std::endl;
					return 0;
				}
                                if(pf.WriteTransformedImages())
                                {
                                        ImageUtils::writeMETA(*images[j],transFileName);
                                }

				
			}
                        det[j] =  transform.determinant();
			detsum = detsum + det[j];
			weights[j] = det[j] / numImages;
			delete abc.finalImage;
			//abc.AffineAtlas::~AffineAtlas();
			//std::cout<<"J: "<<j<<"  det: "<<det[j]<<"  detsum: "<<detsum<<std::endl;
		}
        	std::cout<<"Iteration : "<<i+1<<" SquaredError"<<squarederror<<std::endl;
	        //Array3DUtils::arithmeticMean(numImages,(const Array3D<AffineAtlas::VoxelType>** const) images,*iavg);
		Array3DUtils::weightedArithmeticMean(numImages,(const Array3D<AffineAtlas::VoxelType>** const) images, weights,*iavg);

		iavg->scale((numImages/detsum));
		detsum = 0;
	        ImageUtils::writeMETA(*iavg,"final");

	}

        ImageUtils::writeMETA(*iavg,"final");
return 1;
}
