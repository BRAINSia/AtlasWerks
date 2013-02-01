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


#include "LDMM.h"
#include "CmdLineParser.h"
#include "ImagePreprocessor.h"
#include "StringUtils.h"
#include "ApplicationUtils.h"
#include "HField3DUtils.h"
#include "ForwardSolve.h"

class FileFormatParam : public CompoundParam {
public:
  FileFormatParam(const std::string& name = "FileFormat", 
		  const std::string& desc = "Specify a set of input files", 
		  ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(ValueParam<std::string>("FormatString", "filename format, expects single integer format (%d or %0d)", PARAM_REQUIRED, ""));
    this->AddChild(ValueParam<unsigned int>("MinIndex", "Minimum file index", PARAM_COMMON, 0));
    this->AddChild(ValueParam<unsigned int>("MaxIndex", "Maximum file index", PARAM_REQUIRED, 0));
  }
  
  ValueParamAccessorMacro(std::string, FormatString)
  ValueParamAccessorMacro(unsigned int, MinIndex)
  ValueParamAccessorMacro(unsigned int, MaxIndex)
  
  CopyFunctionMacro(FileFormatParam)
};

class VelocityUtilsParamFile : public CompoundParam 
{
public:
  VelocityUtilsParamFile()
    : CompoundParam("ParameterFile", "top-level node", PARAM_REQUIRED)
  {
    // input image and preprocessing, if there is one
    this->AddChild(ValueParam<std::string>("ImageToDeform", "image to deform by the final deformation field [DeformedImage = ImageToDeform(defField)]", PARAM_COMMON, ""));
    this->AddChild(ImagePreprocessorParam("ImagePreprocessor", "Preprocessing to perform on ImageToDeform"));

    XORParam inputField("InputField", "One of several different ways to specify an input field", PARAM_COMMON);
    inputField.AddChild(FileFormatParam("VFieldFormat", "Format for reading in a set of vFields"));
    inputField.AddChild(ValueParam<std::string>("VField", "vField filename", PARAM_COMMON, ""));
    inputField.AddChild(FileFormatParam("HFieldFormat", "Format for reading in a set of hFields"));
    inputField.AddChild(ValueParam<std::string>("HField", "hField filename", PARAM_COMMON, ""));
    this->AddChild(MultiParam<XORParam>(inputField));

    // misc. options
    this->AddChild(ValueParam<bool>("TestHV", "Test each H/VField to make sure it is the type specified", PARAM_COMMON, true)); 
    
    // composition options
    this->AddChild(ValueParam<bool>("Reverse", "Compute the deformation field pointing in the reverse direction of velocities.  Remember, the forward deformation pulls the final image back to the initial -- the reverse deformation pulls the initial image to the final.", PARAM_COMMON, false)); 
    this->AddChild(ValueParam<bool>("ConserveMass", "Should we scale by the jacobian of the deformation in order to conserve mass?", PARAM_RARE, false));
    this->AddChild(ValueParam<bool>("PrecomposeDeformation", "Compose deformation fields before deforming image (versus deforming the image by each individual deformation in succession)", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("Splat", "Use forward-apply ('splatting') instead of deformation", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("Iterative", "If a forward (Reverse=false) deformation (ConserveMass=false, Splat=false) is specified, and this flag is true, an iterative solution for the deformatin will be found.", PARAM_RARE, false));
    this->AddChild(ForwardSolveParam("ForwardSolve"));
    this->AddChild(ValueParam<unsigned int>("NIterations", "If 'Iterative' is specified, controls the number of iterations", PARAM_COMMON, 20));
    
    // output file options
    this->AddChild(ValueParam<std::string>("JacDetName", "output name for the final determinant of jacobian of deformation -- only computed if ConserveMass is true", PARAM_COMMON, ""));
    this->AddChild(ValueParam<std::string>("DefImageName", "output name for the final deformed image", PARAM_COMMON, ""));
    this->AddChild(ValueParam<std::string>("DefFieldName", "output name for the final deformation field", PARAM_COMMON, ""));
    this->AddChild(ValueParam<std::string>("IntermediateImageFormat", "printf-style format for intermediate images, intermediate images will be saved if this field is non-null", PARAM_COMMON, ""));
  }

  ValueParamAccessorMacro(std::string, ImageToDeform);
  ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor);

  ParamAccessorMacro(MultiParam<XORParam>, InputField);

  ValueParamAccessorMacro(bool, TestHV);

  ValueParamAccessorMacro(bool, Reverse);
  ValueParamAccessorMacro(bool, ConserveMass);
  ValueParamAccessorMacro(bool, PrecomposeDeformation);
  ValueParamAccessorMacro(bool, Splat);
  ValueParamAccessorMacro(bool, Iterative);
  ParamAccessorMacro(ForwardSolveParam, ForwardSolve);
  ValueParamAccessorMacro(unsigned int, NIterations);
  
  ValueParamAccessorMacro(std::string, JacDetName);
  ValueParamAccessorMacro(std::string, DefImageName);
  ValueParamAccessorMacro(std::string, DefFieldName);
  ValueParamAccessorMacro(std::string, IntermediateImageFormat);

  CopyFunctionMacro(VelocityUtilsParamFile)
};

enum VecFieldType {HFIELD, VFIELD};

//
// Load the given vector field as a vField (convert if it's an hField)
//
void loadAsVField(std::string name, VecFieldType type, VectorField &vf, 
		  bool testType = true, const Vector3D<Real> &spacing = Vector3D<Real>(1.0, 1.0, 1.0))
{

  std::cout << "loading ";
  if(type == VFIELD){
    std::cout << "vField ";
  }else{
    std::cout << "hField ";
  }
  std::cout << name << std::endl;

  ApplicationUtils::LoadHFieldITK(name.c_str(), vf);
  if(testType){
    std::stringstream ss;
    if(type == HFIELD && !HField3DUtils::IsHField(vf)){
      ss << "Error, hField looks more like a vField: " << name;
      throw AtlasWerksException(__FILE__, __LINE__, ss.str());
    }else if(type == VFIELD && HField3DUtils::IsHField(vf)){
      ss << "Error, vField looks more like a hField: " << name;
      throw AtlasWerksException(__FILE__, __LINE__, ss.str());
    }
  }

  if(type == HFIELD){
    HField3DUtils::hToVelocity(vf, spacing);
  }
}

/**
 * Update the jacobian determinant by the specified hField
 */
void updateJacDet(RealImage &jacDet, const VectorField &v, const Vector3D<Real> &spacing)
{
  RealImage scratchI(v.getSize());
  VectorField scratchV(v.getSize());
  // deform last timestep's jacobian
  HField3DUtils::applyU(jacDet, v, scratchI, spacing);
  jacDet = scratchI;
  // compute x+v in world space
  HField3DUtils::setToIdentity(scratchV);
  scratchV.scale(spacing);
  scratchV.pointwiseAdd(v);
  // compute jacobian determinant of x+v
  HField3DUtils::jacobian(scratchV,scratchI,spacing);
  // scale deformed jacobian by this update
  jacDet.pointwiseMultiplyBy(scratchI);
}

/**
 * Composes fields and returns an HField
 */
void composeFields(std::vector<std::string> vecNames, 
		   std::vector<VecFieldType> filetype,
		   VectorField &h, 
		   const Vector3D<Real> spacing,
		   bool reverse=false, bool invert=true, bool testHV=true,
		   RealImage *jacDet=NULL)
{
  int start = 0;
  int end = vecNames.size()-1;
  int step = 1;
  if(reverse){
    start = vecNames.size()-1;
    end = 0;
    step = -1;
  }
  
  std::cout << "Running composeFields with reverse = " << reverse << ", invert = " << invert << std::endl;

  HField3DUtils::setToIdentity(h);
  if(jacDet){ 
    jacDet->fill(1.0);
  }
  VectorField vf(h.getSize());;
  VectorField scratchV(h.getSize());;
  for(int vfIdx=start;fabs(vfIdx-start)<=fabs(end-start);vfIdx+=step){
    std::cout << "   Composing vField " << vfIdx << std::endl;
    loadAsVField(vecNames[vfIdx], filetype[vfIdx], vf, testHV, spacing);
    if(invert) vf.scale(-1.0);
    HField3DUtils::composeHV(h, vf, scratchV, spacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    h = scratchV;
    if(jacDet){
      updateJacDet(*jacDet, vf, spacing);
    }
  }
}

// void createImage(std::vector<std::string> filename,
// 		 std::vector<VecFieldType> filetype,
// 		 bool splat, bool conserveMass, bool iterative, 
// 		 bool reverseCompose, bool invertField, bool testHV,
// 		 VectorField &h, RealImage &I)
// {
//   // compose field
//   if(!splat && conserveMass()){
//     composeFields(filename, filetype, hField, spacing, reverseCompose, invertField, testHV, &jacDet);
//   }else{
//     composeFields(filename, filetype, hField, spacing, reverseCompose, invertField, testHV);
//   }
  
//   // create image
//   if(pf.Splat()){
//     HField3DUtils::forwardApply(*image, hField, scratchI, (Real)0.0, !pf.ConserveMass());
//   }else{
//     if(pf.Iterative()){
//       HField3DUtils::hToVelocity(scratchV, spacing);
//       forwardSolver->Solve(*image, scratchV, scratchI);
//     }else{
//       HField3DUtils::apply(*image, hField, scratchI);
//       if(pf.ConserveMass()){
// 	scratchI.pointwiseMultiplyBy(jacDet);
//       }
//     }
//   }
// }

/**
 * \page LDMMVelocityUtils
 *
 * Just some simple utilities for computing a deformation and/or
 * deformed image from a set of velocities or hFields
 */
int main(int argc, char ** argv)
{

  VelocityUtilsParamFile pf;

  try{
    CmdLineParser parser(pf);
    parser.Parse(argc,argv);
  }catch(ParamException &pe){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << pe.what() << std::endl;
    std::exit(-1);
  }

  std::vector<std::string> filename;
  std::vector<VecFieldType> filetype;

  //
  // Get all the v/hField names in the correct order.  Just the names
  // for now, we don't want to load them all into memory.
  //
  std::string fname;
  for(unsigned int argIdx=0;argIdx<pf.InputField().size();argIdx++){
    ParamBase *base = pf.InputField()[argIdx].GetSetParam();
    if(base){
      if(base->MatchesName("VFieldFormat")){
	FileFormatParam *formatParam = dynamic_cast<FileFormatParam*>(base);
	if(!formatParam) throw(AtlasWerksException(__FILE__, __LINE__, "Error casting XORParam to FileFormatParam"));
	for(unsigned int i=formatParam->MinIndex(); i <= formatParam->MaxIndex(); i++){
	  fname = StringUtils::strPrintf(formatParam->FormatString().c_str(), i);
	  filename.push_back(fname);
	  filetype.push_back(VFIELD);
	}
      }else if(base->MatchesName("HFieldFormat")){
	FileFormatParam *formatParam = dynamic_cast<FileFormatParam*>(base);
	if(!formatParam) throw(AtlasWerksException(__FILE__, __LINE__, "Error casting XORParam to FileFormatParam"));
	for(unsigned int i=formatParam->MinIndex(); i <= formatParam->MaxIndex(); i++){
	  fname = StringUtils::strPrintf(formatParam->FormatString().c_str(), i);
	  filename.push_back(fname);
	  filetype.push_back(HFIELD);
	}
      }else if(base->MatchesName("VField")){
	ValueParam<std::string> *fileParam = dynamic_cast<ValueParam<std::string>*>(base);
	if(!fileParam) throw(AtlasWerksException(__FILE__, __LINE__, "Error casting XORParam to string ValueParam"));
	filename.push_back(fileParam->Value());
	filetype.push_back(VFIELD);
      }else if(base->MatchesName("HField")){
	ValueParam<std::string> *fileParam = dynamic_cast<ValueParam<std::string>*>(base);
	if(!fileParam) throw(AtlasWerksException(__FILE__, __LINE__, "Error casting XORParam to string ValueParam"));
	filename.push_back(fileParam->Value());
	filetype.push_back(HFIELD);
      }else{
	std::stringstream ss;
	ss << "Error, unrecognized InputField type: " << base->GetName();
	throw AtlasWerksException(__FILE__, __LINE__, ss.str());
      }
    }
  }

  // Test that we got at least one vector field
  unsigned int nFields = filename.size();
  if(nFields == 0){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, no vField/hField files specified");
  }
  
  // load the first field to get the size
  VectorField vf;
  ApplicationUtils::LoadHFieldITK(filename[0].c_str(), vf);
  Vector3D<unsigned int> size = vf.getSize();

  // default is unit spacing
  Vector3D<Real> origin(1.0, 1.0, 1.0);
  Vector3D<Real> spacing(1.0, 1.0, 1.0);

  // if there is an image to deform, load it and use its spacing
  RealImage *image = NULL;
  if(pf.ImageToDeform().length() > 0){
    image = new RealImage();
    ApplicationUtils::LoadImageITK(pf.ImageToDeform().c_str(), *image);
    spacing = image->getSpacing();
    origin = image->getOrigin();
  }

  // TODO: have a way to set the spacing

  //
  // Test that we have a legal set of parameters
  // 
  if(!pf.PrecomposeDeformation() && !image){
    std::cerr << "Error, stepwise deformation requested but no input image specified" << std::endl;
    std::exit(-1);
  }

  if(pf.DefFieldName().length() > 0){
    if(!pf.PrecomposeDeformation()){
      std::cerr << "Error, cannot save deformation, stepwise deformation "
		<< "requested (PrecomposeDeformation = false) "
		<< std::endl;
      std::exit(-1);
    }
  }

  ForwardSolve *forwardSolver = NULL;
  if(pf.Iterative()){
    if(pf.Reverse() || pf.ConserveMass() || pf.Splat()){
      std::cerr << "Error, illegal set of parameters with Iterative=true "
		<< "(Reverse, ConserveMass, and Splat must be false)" 
		<< std::endl;
      std::exit(-1);
    }else{
      forwardSolver = new ForwardSolve(size, origin, spacing, pf.ForwardSolve());
    }
  }

  if(pf.JacDetName().length() > 0){
    if(!pf.ConserveMass() || pf.Splat() || !pf.PrecomposeDeformation()){
      std::cerr << "Error, cannot save jacobian determinant, it will not "
		<< "be calculated with current parameters."
		<< std::endl;
      std::exit(-1);
    }
  }

  bool writeSteps = false;
  if(pf.IntermediateImageFormat().length() > 0){
    writeSteps = true;
  }

  VectorField hField(size);
  HField3DUtils::setToIdentity(hField);
  RealImage jacDet(size, origin, spacing);
  VectorField scratchV(size);
  RealImage scratchI(size, origin, spacing);

  bool reverseCompose = false;
  bool invertField = true;
  if(pf.PrecomposeDeformation()){
    //
    // Precomposed deformation
    //
    if(pf.Splat()){
      if(pf.Reverse()){
	//
	// Reverse Precomposed Splatting
	//
	reverseCompose = false;
	invertField = true;
      }else{
	//
	// Forward Precomposed Splatting
	//
	reverseCompose = true;
	invertField = false;
      }
    }else{
      if(pf.Reverse()){
	//
	// Reverse Precomposed Deformation
	//
	reverseCompose = true;
	invertField = false;
      }else{
	//
	// Forward Precomposed Deformation
	//
	if(pf.Iterative()){
	  reverseCompose = true;
	  invertField = false;
	}else{
	  reverseCompose = false;
	  invertField = true;
	}
      }
    }
    
    if(writeSteps){
      std::vector<std::string> partFilename;
      std::vector<VecFieldType> partFiletype;
      int cnt = 0;

      if(reverseCompose){
	// iterate forward, taking fields from beginning to current position
	std::vector<std::string>::iterator fnit = filename.begin();
	std::vector<VecFieldType>::iterator ftit = filetype.begin();
	while(true){

	  if(fnit == filename.end()) break;

	  ++fnit;
	  ++ftit;
	  partFilename.assign(filename.begin(), fnit);
	  partFiletype.assign(filetype.begin(), ftit);
	  
	  // create the intermediate image
	  
	  // compose field
	  if(!pf.Splat() && pf.ConserveMass()){
	    composeFields(partFilename, partFiletype, hField, spacing, reverseCompose, invertField, pf.TestHV(), &jacDet);
	  }else{
	    composeFields(partFilename, partFiletype, hField, spacing, reverseCompose, invertField, pf.TestHV());
	  }
	  
	  // create image
	  if(pf.Splat()){
	    HField3DUtils::forwardApply(*image, hField, scratchI, (Real)0.0, !pf.ConserveMass());
	  }else{
	    if(pf.Iterative()){
	      scratchV = hField;
	      HField3DUtils::hToVelocity(scratchV, spacing);
	      forwardSolver->Solve(*image, scratchV, scratchI);
	    }else{
	      HField3DUtils::apply(*image, hField, scratchI);
	      if(pf.ConserveMass()){
		scratchI.pointwiseMultiplyBy(jacDet);
	      }
	    }
	  }
	  
	  ++cnt;
	  ApplicationUtils::
	    SaveImageITK(StringUtils::
			 strPrintf(pf.IntermediateImageFormat().c_str(), cnt).c_str(), 
			 scratchI);
	  
	} // end iterate
	
      }else{ // not reverseCompose
	// iterate backwards, taking fields from current position to end
	// iterate forward, taking fields from beginning to current position
	std::vector<std::string>::reverse_iterator fnit = filename.rbegin();
	std::vector<VecFieldType>::reverse_iterator ftit = filetype.rbegin();
	while(true){

	  if(fnit == filename.rend()) break;

	  ++fnit; 
	  ++ftit;

	  partFilename.assign(fnit.base(), filename.end());
	  partFiletype.assign(ftit.base(), filetype.end());

	  // create the intermediate image

	  // compose field
	  if(!pf.Splat() && pf.ConserveMass()){
	    composeFields(partFilename, partFiletype, hField, spacing, reverseCompose, invertField, pf.TestHV(), &jacDet);
	  }else{
	    composeFields(partFilename, partFiletype, hField, spacing, reverseCompose, invertField, pf.TestHV());
	  }
    
	  // create image
	  if(pf.Splat()){
	    HField3DUtils::forwardApply(*image, hField, scratchI, (Real)0.0, !pf.ConserveMass());
	  }else{
	    if(pf.Iterative()){
	      HField3DUtils::hToVelocity(scratchV, spacing);
	      forwardSolver->Solve(*image, scratchV, scratchI);
	    }else{
	      HField3DUtils::apply(*image, hField, scratchI);
	      if(pf.ConserveMass()){
		scratchI.pointwiseMultiplyBy(jacDet);
	      }
	    }
	  }
	  
	  ++cnt;
	  ApplicationUtils::
	    SaveImageITK(StringUtils::
			 strPrintf(pf.IntermediateImageFormat().c_str(), cnt).c_str(), 
			 scratchI);

	} // end iterate

      } // end if/else reverseCompose

    } // end if writeSteps

    // compose field
    if(!pf.Splat() && pf.ConserveMass()){
      composeFields(filename, filetype, hField, spacing, reverseCompose, invertField, pf.TestHV(), &jacDet);
    }else{
      composeFields(filename, filetype, hField, spacing, reverseCompose, invertField, pf.TestHV());
    }
    
    // create image
    if(image){
      if(pf.Splat()){
	HField3DUtils::forwardApply(*image, hField, scratchI, (Real)0.0, !pf.ConserveMass());
      }else{
	if(pf.Iterative()){
	  HField3DUtils::hToVelocity(scratchV, spacing);
	  forwardSolver->Solve(*image, scratchV, scratchI);
	}else{
	  HField3DUtils::apply(*image, hField, scratchI);
	  if(pf.ConserveMass()){
	    scratchI.pointwiseMultiplyBy(jacDet);
	  }
	}
      }
      
      *image = scratchI;
    }

  }else{

    //
    // Stepwise deformation
    //

    if(pf.Splat()){
      if(pf.Reverse()){
	//
	// Reverse Stepwise Splatting
	//
	reverseCompose = true;
	invertField = true;
      }else{
	//
	// Forward Stepwise Splatting
	//
	reverseCompose = false;
	invertField = false;
      }
    }else{
      if(pf.Reverse()){
	//
	// Reverse Stepwise Deformation
	//
	reverseCompose = true;
	invertField = false;
      }else{
	//
	// Forward Stepwise Deformation
	//
	if(pf.Iterative()){
	  reverseCompose = false;
	  invertField = false;
	}else{
	  reverseCompose = false;
	  invertField = true;
	}
      }
    }

    int start = 0;
    int end = filename.size()-1;
    int step = 1;
    if(reverseCompose){
      start = filename.size()-1;
      end = 0;
      step = -1;
    }
    
    HField3DUtils::setToIdentity(hField);
    for(int vfIdx=start;fabs(vfIdx-start)<=fabs(end-start);vfIdx+=step){
      loadAsVField(filename[vfIdx], filetype[vfIdx], vf, pf.TestHV(), spacing);
      if(invertField) vf.scale(-1.0);
      
      if(pf.Splat()){
	HField3DUtils::velocityToH(vf, spacing);
	HField3DUtils::forwardApply(*image, vf, scratchI, (Real)0.0, !pf.ConserveMass());
	*image = scratchI;
      }else{
	if(pf.Iterative()){
	  forwardSolver->Solve(*image, vf, scratchI);
	  *image = scratchI;
	}else{
	  HField3DUtils::applyU(*image, vf, scratchI, spacing);
	  *image = scratchI;
	  if(pf.ConserveMass()){
	    jacDet.fill(1.0);
	    updateJacDet(jacDet, vf, spacing);	
	    image->pointwiseMultiplyBy(jacDet);
	  }
	} // end Iterative
      } // end Splat
      
      if(writeSteps){
	int stepnum = 1 + (int)(fabs(vfIdx-start)+0.5);
	ApplicationUtils::
	  SaveImageITK(StringUtils::
		       strPrintf(pf.IntermediateImageFormat().c_str(), stepnum).c_str(), 
		       *image);
      }
    } // end iterate over fields
  } // end PrecomposeDeformation

  // save the final deformation
  if(pf.PrecomposeDeformation() && pf.DefFieldName().length() > 0){
    std::cout << "Saving deformed field " << pf.DefFieldName() << std::endl;
    ApplicationUtils::SaveHFieldITK(pf.DefFieldName().c_str(), hField);
  }
  
  // save the final image
  if(image && pf.DefImageName().length() > 0){
    std::cout << "Saving deformed image " << pf.DefImageName() << std::endl;
    ApplicationUtils::SaveImageITK(pf.DefImageName().c_str(), *image);
  }
  
  // save the jacobian determinant
  if(pf.JacDetName().size() > 0 && pf.ConserveMass()){
    std::cout << "Saving jacobian determinant " << pf.JacDetName() << std::endl;
    ApplicationUtils::SaveImageITK(pf.JacDetName().c_str(), jacDet);
  }
  
} // end main

