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


#include "DiffOper.h"
#include "ImagePreprocessor.h"
#include "CmdLineParser.h"
#include "KernelParam.h"
#include "KernelFactory.h"
#include "ApplicationUtils.h"

#ifdef CUDA_ENABLED
#include "cudaInterface.h"
#include "DiffOperGPU.h"
#include "CUDAUtilities.h"
#endif

/**
 * \page DiffOper
 *
 * Simple frontend for applying differential operator to a vector field
 */

class DiffOperParamFile : public CompoundParam 
{
public:
  DiffOperParamFile()
    : CompoundParam("ParameterFile", "top-level node", PARAM_REQUIRED)
  {
    this->AddChild(ValueParam<std::string>("InputVectorField", "vector field to which the differential operator is to be applied", PARAM_REQUIRED, ""));
    this->AddChild(ValueParam<std::string>("OutputFileName", "Filname of result vector field", PARAM_REQUIRED, ""));
    this->AddChild(KernelParam("Kernel"));
    this->AddChild(ValueParam<bool>("UseGPU", "use GPU version of DiffOper?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("ApplyInverse", "Apply K, not L?", PARAM_COMMON, true));
  }

  ValueParamAccessorMacro(std::string, InputVectorField)
  ValueParamAccessorMacro(std::string, OutputFileName)
  ParamAccessorMacro(KernelParam, Kernel)
  ValueParamAccessorMacro(bool, UseGPU)
  ValueParamAccessorMacro(bool, ApplyInverse)

  CopyFunctionMacro(DiffOperParamFile)
};


int main(int argc, char ** argv)
{

  DiffOperParamFile pf;

  try{
    CmdLineParser parser(pf);
    parser.Parse(argc,argv);
  }catch(ParamException &pe){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << pe.what() << std::endl;
    std::exit(-1);
  }

  VectorField v;
  SizeType size;
  OriginType origin;
  SpacingType spacing;
  ApplicationUtils::LoadHFieldITK(pf.InputVectorField().c_str(), origin, spacing, v);
  size = v.getSize();
  
  if(pf.UseGPU()){
#ifdef CUDA_ENABLED
    cplVector3DArray dV;
    allocateDeviceVector3DArray(dV, size.productOfElements());
    CUDAUtilities::CopyVectorFieldToDevice(v, dV, true); 
    LOGNODE(logDEBUG) <<  "Using GPU Kernel";
    KernelInterfaceGPU *mdKernel = 
      KernelFactory::NewGPUKernel(pf.Kernel());
    mdKernel->SetSize(size, spacing, pf.Kernel());
    if(pf.ApplyInverse()){
      LOGNODE(logDEBUG) <<  "Applying Inverse Kernel";
      mdKernel->ApplyInverseOperator(dV);
    }else{
      LOGNODE(logDEBUG) <<  "Applying Forward Kernel";
      mdKernel->ApplyOperator(dV);
    }
    CUDAUtilities::CopyVectorFieldFromDevice(dV, v, true); 
    delete mdKernel;
#else // !CUDA_ENABLED
    throw AtlasWerksException(__FILE__, __LINE__, "Error, GPU code not built.  Select USE_CUDA in CMake settings.");
#endif // CUDA_ENABLED
  }else{
    LOGNODE(logDEBUG) <<  "Using CPU Kernel";
    KernelInterface *mKernel = 
      KernelFactory::NewKernel(pf.Kernel(), size, spacing);
    mKernel->CopyIn(v);
    if(pf.ApplyInverse()){
      LOGNODE(logDEBUG) <<  "Applying Inverse Kernel";
      mKernel->ApplyInverseOperator();
    }else{
      LOGNODE(logDEBUG) <<  "Applying Forward Kernel";
      mKernel->ApplyOperator();
    }
    mKernel->CopyOut(v);
    delete mKernel;
  }
 
  ApplicationUtils::SaveHFieldITK(pf.OutputFileName().c_str(), 
				  v, origin, spacing);
  
  LOGNODE(logDEBUG) <<  "Done writing results";
  
}

