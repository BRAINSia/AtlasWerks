#ifndef IMAGE_IO_CXX
#define IMAGE_IO_CXX

#include "ImageIO.h"

#include <fstream>
#include <ios>
#include <cmath>

#include <Array3DUtils.h>
#include <BasicException.h>
#include <BinaryIO.h>

#ifdef WIN32

#include <io.h>
#include <vector>

#endif

using std::ifstream;
using std::istream;
using std::ofstream;
using std::cout;
using std::cerr;
using std::vector;

using std::endl;

//   ImageIO allow to load images of the following format :
//         -> Analyze (divided in two files the header (*.hdr) with 
// the information about the image and the data (*.img)) 
//         -> Meta (*.[mha|mhd])     
//         -> Gipl (*.gipl)
//         -> PlanIM (no specific name, most of the time you have plan_im 
// in the filename, can be *.pim)
//         -> Dicom (one file per slice, ne specific name) 
//
//   Byte swapping functions have written for analyze  and gipl images
// We assume (it has been tested ! ) that PlanIm, Meta and Dicom are already
// handling byte order.



/*****************************************************************************/
/*                               REMINDER                                    */
/*****************************************************************************/


//   Whatever is the format of the data we are reading, we write it
// in the same structure (image3D) with always the same datatype (float)
//   Array3D<float> image3D;
//
// LOAD  
//  That's why each time we load an image we have a switch to make the 
// "conversion" from the original datatype (the one of the file) to float.
// We do that in 2 times, first we create an Array3D of the same type as the
// image we are loading then from this temporary Array3D we fill image3D.
//
// SAVE
// When we save we are doing exactly the opposite.
// We create an Array3D of the same type as the data we load (we keep track
// of the data type in header.data_type), we fill it from image3D.
// That's why we'll have "switch (header.data_type)" in the save functions.


/*****************************************************************************/





// Guess the format of an image depending 
//    1. on its extension
//           if ( *.img or *.hdr) return analyze
//           if ( *.gipl ) return gipl
//           if ( *.[mha|mhd] ) return meta
//           if ( *.dcm ) return dicom
//           if ( *.pim ) return planIM
//   2. if none of the above match, check if there is "plan_im" in 
//      the filename, if so return planIM
//   3. Failing that, if there is "sum" in the name, then it is a dose
//      grid.
//   4. if there is a * in the filename the image is considered as a Dicom


ImageIO::ImageType ImageIO::GuessImageFormat(string filename){
  
  string extension=filename;
  int pos = extension.find_last_of(".");
  extension.erase(extension.begin(),extension.begin() + pos + 1 );

  if (extension.compare("hdr")==0) return analyze;
  if (extension.compare("dcm")==0) return dicom;
  if (extension.compare("img")==0) return dicom;
  if (extension.compare("pim")==0) return planIM;

  string::size_type posRTOG = filename.find ("aapm",0);
  if (posRTOG != string::npos) return RTOG;

  string::size_type posPlan_im = filename.find ("plan_im",0);
  if (posPlan_im != string::npos) return planIM;

  string::size_type posGrid = filename.find ("sum",0);
  if (posGrid != string::npos) return PLUNC_dose_grid;

  //
  // these are here for backwards compatibility
  // with a small group of misnamed files, 
  // DON'T USE big_im or small_im for any new files
  //
  string::size_type posbig_im = filename.find ("big_im",0);
  if (posbig_im != string::npos) return planIM;

  string::size_type possmall_im = filename.find ("small_im",0);
  if (possmall_im != string::npos) return planIM;

  string::size_type posStar = filename.find ("*",0);
  if (posStar != string::npos) return dicom;

  return unknown;
}

/*ImageIO::ImageType ImageIO::GuessImageFormat(string filename){
  
  char *temp, *extension;
  char *temp_name = new char[strlen(filename.c_str())+1];
  strcpy(temp_name,filename.c_str());
  temp = strtok(temp_name,".");  

  // delete [] temp_name;
  while (temp != NULL)
    {
      extension=temp;
      temp = strtok(NULL,".");
    }

  if (strcmp(extension,"img")==0 ||
      strcmp(extension,"hdr")==0) { 
    return analyze;
  }
  if (strcmp(extension,"mha")==0 || strcmp(extension,"mhd")==0) return meta;  
  if (strcmp(extension,"gipl")==0) return gipl; 
  if (strcmp(extension,"dcm")==0) return dicom;
  if (strcmp(extension,"pim")==0) return planIM;

  string::size_type posPlan_im = filename.find ("plan_im",0);
  if (posPlan_im != string::npos) return planIM;
  
  string::size_type posStar = filename.find ("*",0);
  if (posStar != string::npos) return dicom;

  return unknown;
}*/

// Load an image guessing its format
// With PlanIM, take the default parameter by guessParameters(...))
// With Dicom, we browse the different iamges of the volume (nb_DICOM)
// and we select the one with the maximun nb of slice in Z.
// We assume that the "interesting" part is the one with the max nb of slice
// We throw an exception if the file format is unknown.

void ImageIO::LoadThisImage(string filename,Image<float> &image3D,ImageType extension){

  //ImageType extension;
  if (extension==unknown){
    extension=GuessImageFormat(filename);
  }
  int i=0;
  
  switch(extension){
  case analyze :
    LoadAnalyze(filename,image3D);
    break;
  case PLUNC_dose_grid :
    LoadDoseGrid(filename,image3D);
    break;
  case planIM : // We take the default parameter
    try {
      LoadPlanIM(filename,image3D);
    }
    catch (BasicException e){ // Load_PlanIM irregularly sliced
      //Debugging::debugOn();
      LoadPlanIMIrreg(filename,image3D);
    }
   break;
  case dicom :
    {
      int selectedDicom;
      int maxSlice;
      selectedDicom  =0;
      maxSlice=0;
      
      LoadDicom(filename);
      
      for (i=0; i<nb_DICOM ; i++){
	dicomImage.SelectImage(i);
	if (dicomImage.getsizeZ()>maxSlice){
	  maxSlice=dicomImage.getsizeZ();
	  selectedDicom=i;
	}
      }
      // unused var//bool uniformSpacing = CheckSpacingSelectedDicom(selectedDicom);
      LoadSelectedDicom(image3D);
    }
    break;
  // unhandled options
  case gipl:
  case meta:
  case RTOG :
    // TODO: interface options to load 
    break;
  case unknown :
    throw bException("Load Failed : Unknown file format");
    break;
  }
}

// Save an image guessing its format
// Don't work with PlanIM and Dicom because of the user interface part
// We throw an exception in such cases.

void ImageIO::SaveImage(string filename, Image<float> &image3D){
	//std::string filename(Filename);
  ImageType extension;
  extension=GuessImageFormat(filename);

  switch(extension){
  case analyze :
    SaveAnalyze(filename,image3D);
    break;
  case PLUNC_dose_grid :
    SaveDoseGrid(filename, image3D);
    break;
  case planIM :
    SavePlanIM(filename,image3D,0.0);
    //throw  bException("Save failed : Can't save PlanIm images");
    break;
  case dicom :
    throw  BasicException("ImageIO.cxx",
                          "Save failed: Can't save Dicom images");
    break;
  case gipl:
  case meta:
  case RTOG :
    // TODO: unhandled
    break;
  case unknown :
    throw  BasicException(
      "ImageIO.cxx",
      "Save failed : Unknown file format.\n"
      "Filename must either contain the string \"plan_im\" or \"sum\", \n"
      "or end in .hdr,  or .pim." );
    break;
  }
}



/**********************************************************************/
/**********************************************************************/
/*                                                                    */
/*                            ANALYZE                                 */
/*                                                                    */
/**********************************************************************/
/**********************************************************************/


// Given a filename return :   filename.img (for the data)
//                             filename.hdr (for the header)
void ImageIO::ConstructAnalyzeNames(const char *filename,char *ret_prefix,
                           	      char *ret_hdrname,char *ret_imgname)
{
  int l = strlen(filename);
  sprintf(ret_prefix,filename);

  if((l>4)&&((!strncmp(&(ret_prefix[l-4]),".img",3))||
	     (!strncmp(&(ret_prefix[l-4]),".hdr",3))) )
    ret_prefix[l-4] = 0;

  sprintf(ret_hdrname,"%s.hdr",ret_prefix);
  sprintf(ret_imgname,"%s.img",ret_prefix);
}


// Swap the analyze header using binaryIO

void SwapAnalyzeHeader(void *hdr)
{
  int i;
  struct dsr *h = (struct dsr*)hdr;

  BinaryIO binIO;

  h->hk.sizeof_hdr    = binIO.swabInt(h->hk.sizeof_hdr);
  h->hk.extents       = binIO.swabInt(h->hk.extents);
  h->hk.session_error = binIO.swabShort(h->hk.session_error);

  for(i=0;i<8;i++)
    h->dime.dim[i] = binIO.swabShort(h->dime.dim[i]);
  h->dime.unused1 = binIO.swabShort(h->dime.unused1);
  h->dime.datatype = binIO.swabShort(h->dime.datatype);
  h->dime.bitpix = binIO.swabShort(h->dime.bitpix);
  h->dime.dim_un0 = binIO.swabShort(h->dime.dim_un0);

  for(i=0;i<8;i++)
    h->dime.pixdim[i] = binIO.swabFloat(h->dime.pixdim[i]);
  h->dime.vox_offset = binIO.swabFloat(h->dime.vox_offset);
  h->dime.roi_scale =binIO.swabFloat(h->dime.roi_scale);
  h->dime.funused1 = binIO.swabFloat(h->dime.funused1);
  h->dime.funused2 = binIO.swabFloat(h->dime.funused2);
  h->dime.cal_max = binIO.swabFloat(h->dime.cal_max);
  h->dime.cal_min = binIO.swabFloat(h->dime.cal_min);
  h->dime.compressed = binIO.swabInt(h->dime.compressed);
  h->dime.verified = binIO.swabInt(h->dime.verified);
  h->dime.glmax = binIO.swabInt(h->dime.glmax);
  h->dime.glmin = binIO.swabInt(h->dime.glmin);

  h->hist.views = binIO.swabInt(h->hist.views);
  h->hist.vols_added = binIO.swabInt(h->hist.vols_added);
  h->hist.start_field = binIO.swabInt(h->hist.start_field);
  h->hist.field_skip = binIO.swabInt(h->hist.field_skip);
  h->hist.omax = binIO.swabInt(h->hist.omax);
  h->hist.omin = binIO.swabInt(h->hist.omin);
  h->hist.smax = binIO.swabInt(h->hist.smax);
  h->hist.smin = binIO.swabInt(h->hist.smin);
}


/**********************************************************************/
/*************************   LOAD ANALYZE  ****************************/
/**********************************************************************/


// Swab data from u_short

void SwabData(Image<u_short> &dat)
{
  
   BinaryIO binIO;
   Vector3D<unsigned int> size = dat.getSize();
   for(unsigned int k=0;k<size.x;k++)
     for(unsigned int j=0;j<size.y;j++)
       for(unsigned int i=0;i<size.z;i++)
	dat(k,j,i) = binIO.swabShort(dat(k,j,i));

}

// Swab data from float

void SwabData(Image<float> &dat)
{
  BinaryIO binIO;

  for(unsigned int k=0;k<dat.getSize().x;k++)
    for(unsigned int j=0;j<dat.getSize().y;j++)
      for(unsigned int i=0;i<dat.getSize().z;i++)
	dat(k,j,i) = binIO.swabFloat(dat(k,j,i));
}

// Load data depending on the data type

void ImageIO::LoadData(Image<u_char> &dat, istream &infile)
{
  infile.read((char*)dat.getDataPointer(),dat.getSizeBytes());
#ifndef DARWIN_PROBLEMS
  if(infile.bad()) throw bException("ImageIO::loadData Error reading the file");
#endif
}

void ImageIO::LoadData(Image<u_short> &dat, istream &infile)
{
  infile.read((char*)dat.getDataPointer(),dat.getSizeBytes());
//#ifndef DARWIN_PROBLEMS
  if(infile.bad()) throw bException("ImageIO::loadData Error reading the file");
//#endif
  // If byte swapping is needed, we swap bytes for the data
  if (Need_BS){ 
    SwabData(dat);
  }
}

void ImageIO::LoadData(Image<float> &dat, istream &infile)
{
  infile.read((char*)dat.getDataPointer(),dat.getSizeBytes());
#ifndef DARWIN_PROBLEMS
  if(infile.bad()) throw bException("ImageIO::loadData Error reading the file");
#endif
  // If byte swapping is needed, we swap bytes for the data
  if (Need_BS){
    SwabData(dat);
  }
}

// Read the header given by hdrname and 
// write its information in header (given in parameter)

void  ImageIO::ReadAnalyzeHeader(const char *hdrname,Image<float> &image3D)
{

// Open and read the header
 
  struct dsr hdr;

  ifstream inFile(hdrname);
  
  if(inFile.bad()) {
    throw bException("ImageIO::readAnalyzeHeader Error reading the file");
  }

  inFile.read((char*)&hdr,sizeof(struct dsr));

#ifndef DARWIN_PROBLEMS
  if(inFile.bad())
    throw bException("ImageIO::readAnalyzeHeader Error read failed");
#endif

// We check if we need Byte swapping
if ( (((hdr.dime.bitpix>200) || (hdr.dime.bitpix<0)) && (hdr.hk.sizeof_hdr == 0)) || (((hdr.hk.extents != 16384) && (hdr.hk.sizeof_hdr != 348)) && (hdr.hk.sizeof_hdr != 0)))
 {
    Need_BS= true;
    // We swap bytes for the header
    SwapAnalyzeHeader((void*)&hdr);
  }

// Write the info in header (given in parameter)


  image3D.resize(hdr.dime.dim[1],hdr.dime.dim[2],hdr.dime.dim[3]);
  image3D.setSpacing(hdr.dime.pixdim[1], 
	                 hdr.dime.pixdim[2],
					 hdr.dime.pixdim[3]);
 
  if(((int)(hdr.hist.orient)>0)&&((int)(hdr.hist.orient)<24)){
    image3D.setOrientation ((Image<float>::ImageOrientation)hdr.hist.orient); 
  }
  else{ // Default orientation of Analyze is RPI
    image3D.setOrientation( Image<float>::RPI );
  }

  //image3D.setOffsetZ (hdr.dime.funused1);
  //image3D.setOrigin (hdr.dime.funused1,hdr.dime.funused2,hdr.dime.funused3);



  switch(hdr.dime.datatype) {
  case DT_UNSIGNED_CHAR:
    image3D.setDataType( Image<float>::UnsignedChar );
    break;
  case DT_SIGNED_SHORT:
    image3D.setDataType( Image<float>::UnsignedShort );
    break;
  case DT_FLOAT:
    image3D.setDataType( Image<float>::Float );
    break;
  case 0://DT_UNKNOWN:
    switch(hdr.dime.bitpix) {
    case 8:  image3D.setDataType( Image<float>::UnsignedChar ); break;
    case 16: image3D.setDataType( Image<float>::UnsignedShort ); break;
    case 32: image3D.setDataType( Image<float>::Float ); break;
    default: throw bException("ImageIO::readAnalyzeHeader wrong datatype"); break;
    }
    break;
  default:
    throw bException("ImageIO::readAnalyzeHeader wrong datatype");
    break;
  }

  inFile.close();

}

void  ImageIO::ReadAnalyzeHeader(const char *hdrname,Image<unsigned short> &image3D)
{

// Open and read the header
 
  struct dsr hdr;

  ifstream inFile(hdrname);
  
  if(inFile.bad()) {
    throw bException("ImageIO::readAnalyzeHeader Error reading the file");
  }

  inFile.read((char*)&hdr,sizeof(struct dsr));

#ifndef DARWIN_PROBLEMS
  if(inFile.bad())
    throw bException("ImageIO::readAnalyzeHeader Error read failed");
#endif

// We check if we need Byte swapping
  if ( (((hdr.dime.bitpix>200) || (hdr.dime.bitpix<0)) && (hdr.hk.sizeof_hdr == 0)) || (((hdr.hk.extents != 16384) && (hdr.hk.sizeof_hdr != 348)) && (hdr.hk.sizeof_hdr != 0)))
  {
    Need_BS= true;
    // We swap bytes of the header
    SwapAnalyzeHeader((void*)&hdr);
  }

// Write the info in header (given in parameter)

  image3D.resize(hdr.dime.dim[1],hdr.dime.dim[2],hdr.dime.dim[3]);
  image3D.setSpacing(hdr.dime.pixdim[1], 
	                 hdr.dime.pixdim[2],
					 hdr.dime.pixdim[3]);
 
  if(((int)(hdr.hist.orient)>0)&&((int)(hdr.hist.orient)<24)){
    image3D.setOrientation ((Image<unsigned short>::ImageOrientation)hdr.hist.orient); 
  }
  else{ // Default orientation of Analyze is RPI
    image3D.setOrientation (Image<unsigned short>::RPI);
  }

  //image3D.setOffsetZ (hdr.dime.funused1);
  //image3D.setOrigin (hdr.dime.funused1,hdr.dime.funused2,hdr.dime.funused3);

  switch(hdr.dime.datatype) {
  case DT_UNSIGNED_CHAR:
    image3D.setDataType( Image<unsigned short>::UnsignedChar );
    break;
  case DT_SIGNED_SHORT:
    image3D.setDataType( Image<unsigned short>::UnsignedShort );
    break;
  case DT_FLOAT:
    image3D.setDataType( Image<unsigned short>::Float );
    break;
  case 0://DT_UNKNOWN:
    switch(hdr.dime.bitpix) {
    case 8:  image3D.setDataType( Image<unsigned short>::UnsignedChar ); 
      break;
    case 16: image3D.setDataType( Image<unsigned short>::UnsignedShort ); 
      break;
    case 32: image3D.setDataType( Image<unsigned short>::Float ); 
      break;
    default: throw bException("ImageIO::readAnalyzeHeader wrong datatype");
      break;
    }
    break;
  default:
    throw bException("ImageIO::readAnalyzeHeader wrong datatype");
    break;
  }

}

// Load an analyze image pointed by the filename
//
// 1. The information about this image will fill header
// 2. The data will fill image3D

void ImageIO::LoadAnalyze(string filename,Image<float> &image3D){

  char imgname[MAXPATHLEN];
  char hdrname[MAXPATHLEN];
  char fprefix[MAXPATHLEN];
  

  ConstructAnalyzeNames(filename.c_str(),fprefix,hdrname,imgname);
//  sprintf((char *)filename.c_str(),"%s",fprefix);
 
// Part 1 : we fill the header
     
  ReadAnalyzeHeader(hdrname,image3D);
  
// Part 2 : we fill image3D

#ifdef WIN32
  ifstream inFile(imgname, std::ios::binary);
#else
  ifstream inFile(imgname);
#endif
  

  if(inFile.bad()) {
    throw bException("ImageIO::Load_Analyze inFile = 0");
  }


  Image<u_char> u_char_dat;
  Image<u_short> u_short_dat;
  Image<float> float_dat;

  switch(image3D.getDataType()) {
  case Image<float>::UnsignedChar:
    u_short_dat.resize(0,0,0);
    float_dat.resize(0,0,0);
    u_char_dat.resize(image3D.getSize().x,
			     image3D.getSize().y,
			     image3D.getSize().z);
    if(u_char_dat.isEmpty()) {
      throw bException("ImageIO::Load_Analyze Array3D is empty ");
    }
    else {
      LoadData(u_char_dat,inFile);
      float temp;
      for (unsigned int k=0;k<image3D.getSize().x;k++){
		for (unsigned int j=0;j<image3D.getSize().y;j++){
			for (unsigned int i=0;i<image3D.getSize().z;i++){

	    temp = (u_char) u_char_dat(image3D.getSize().x-1-k,image3D.getSize().y-1-j,image3D.getSize().z-1-i);
	    image3D.set(k, j, i, temp);
	  }
	}
      }
    }
    break;

  case Image<float>::UnsignedShort:
    u_char_dat.resize(0,0,0);
    float_dat.resize(0,0,0);
    u_short_dat.resize(image3D.getSize().x,
			      image3D.getSize().y,
			      image3D.getSize().z);
    if(u_short_dat.isEmpty()) {
      throw bException("ImageIO::Load_Analyze Array3D is empty ");
    }
    else {
      LoadData(u_short_dat,inFile);

      float temp;
     for (unsigned int k=0;k<image3D.getSize().x;k++){
	for (unsigned int j=0;j<image3D.getSize().y;j++){
	  for (unsigned int i=0;i<image3D.getSize().z;i++){
	    temp = (u_short) u_short_dat(k,image3D.getSize().y-1-j,image3D.getSize().z-1-i);
	    image3D.set(k, j, i, temp);
	    
	  }
	}
      }
    }
    break;
    
  case Image<float>::Float:
    u_char_dat.resize(0,0,0);
    u_short_dat.resize(0,0,0);
    float_dat.resize(image3D.getSize().x,
			      image3D.getSize().y,
			      image3D.getSize().z);
    if(float_dat.isEmpty()) {
      throw bException("ImageIO::Load_Analyze Array3D is empty ");
    }
    else {
      LoadData(float_dat,inFile);
      float temp;
      for (unsigned int k=0;k<image3D.getSize().x;k++){
	for (unsigned int j=0;j<image3D.getSize().y;j++){
	  for (unsigned int i=0;i<image3D.getSize().z;i++){
	    temp = (float) float_dat(image3D.getSize().x-1-k,image3D.getSize().y-1-j,image3D.getSize().z-1-i);
	    image3D.set(k, j, i, temp);
	  }
	}
      }
    }
    break;
  default:
    throw bException("ImageIO::Load_Analyze switch default ");
    break;
  }


}


void ImageIO::LoadAnalyze(string filename,Image<unsigned short> &image3D){

  char imgname[MAXPATHLEN];
  char hdrname[MAXPATHLEN];
  char fprefix[MAXPATHLEN];
  
  ConstructAnalyzeNames(filename.c_str(),fprefix,hdrname,imgname);
 // sprintf((char *)filename.c_str(),"%s",fprefix);

// Part 1 : we fill the header
     
  ReadAnalyzeHeader(hdrname,image3D);
  
// Part 2 : we fill image3D

#ifdef WIN32
  ifstream inFile(imgname, std::ios::binary);
#else
  ifstream inFile(imgname);
#endif

  if(inFile.bad()) {
    throw bException("ImageIO::Load_Analyze inFile = 0");
  }


  Image<u_char> u_char_dat;
  Image<u_short> u_short_dat;
  Image<float> float_dat;

  switch(image3D.getDataType()) {
  case Image<unsigned short>::UnsignedChar:
    u_short_dat.resize(0,0,0);
    float_dat.resize(0,0,0);
    u_char_dat.resize(image3D.getSize().x,
			     image3D.getSize().y,
			     image3D.getSize().z);
    if(u_char_dat.isEmpty()) {
      throw bException("ImageIO::Load_Analyze Array3D is empty ");
    }
    else {
      LoadData(u_char_dat,inFile);
      u_char temp;
      for (unsigned int k=0;k<image3D.getSize().x;k++){
	for (unsigned int j=0;j<image3D.getSize().y;j++){
	  for (unsigned int i=0;i<image3D.getSize().z;i++){
	    temp = (u_char) u_char_dat(image3D.getSize().x-1-k,image3D.getSize().y-1-j,image3D.getSize().z-1-i);
	    image3D.set(k, j, i, temp);
	  }
	}
      }
    }
    break;

  case Image<unsigned short>::UnsignedShort:
    u_char_dat.resize(0,0,0);
    float_dat.resize(0,0,0);
    u_short_dat.resize(image3D.getSize().x,
			      image3D.getSize().y,
			      image3D.getSize().z);
    if(u_short_dat.isEmpty()) {
      throw bException("ImageIO::Load_Analyze Array3D is empty ");
    }
    else {
      LoadData(u_short_dat,inFile);

      u_short temp;
     for (unsigned int k=0;k<image3D.getSize().x;k++){
	for (unsigned int j=0;j<image3D.getSize().y;j++){
	  for (unsigned int i=0;i<image3D.getSize().z;i++){
	    temp = (u_short) u_short_dat(image3D.getSize().x-1-k,image3D.getSize().y-1-j,image3D.getSize().z-1-i);

	    image3D.set(k, j, i, temp);
	    
	  }
	}
      }
    }
    break;
    
  case Image<unsigned short>::Float:
    u_char_dat.resize(0,0,0);
    u_short_dat.resize(0,0,0);
    float_dat.resize(image3D.getSize().x,
			      image3D.getSize().y,
			      image3D.getSize().z);
    if(float_dat.isEmpty()) {
      throw bException("ImageIO::Load_Analyze Array3D is empty ");
    }
    else {
      LoadData(float_dat,inFile);
      float temp;
      for (unsigned int k=0;k<image3D.getSize().x;k++){
	for (unsigned int j=0;j<image3D.getSize().y;j++){
	  for (unsigned int i=0;i<image3D.getSize().z;i++){
	    temp = (float) float_dat(image3D.getSize().x-1-k,image3D.getSize().y-1-j,image3D.getSize().z-1-i);
	    image3D.set(k, j, i, (u_short)temp); /* Danger */
	  }
	}
      }
    }
    break;
  default:
    throw bException("ImageIO::Load_Analyze switch default ");
    break;
  }
 Need_BS = false;
}


/**********************************************************************/
/*************************   SAVE ANALYZE  ****************************/
/**********************************************************************/


/***********************************************************/
//                   Save Analyse header
/***********************************************************/

// Write the information conatined in header in the file pointed by hdrname
// by copying the information in hdr () and writing the structure in the file

void ImageIO::SaveAnalyzeHeader(string hdrname, Image<float> &image3D)
{

// hdr is the structure that will contain the header info
   struct    dsr hdr;

   ofstream outFile(hdrname.c_str());
   if(outFile.fail()) {
     throw bException("ImageIO::saveAnalyzeHeader Error");
   }

  memset((void*)&hdr,0,sizeof(struct dsr));

  memset(&hdr,0, sizeof(struct dsr));

// We fill hdr with the info from header

  for(int i=0;i<8;i++)
    hdr.dime.pixdim[i]=0.0;

  hdr.dime.vox_offset = 0.0;
  hdr.dime.roi_scale  = 1.0;
  hdr.dime.funused1   = image3D.getOrigin().z;
  hdr.dime.funused2   = 0.0;
  hdr.dime.cal_max    = 0.0;
  hdr.dime.cal_min    = 0.0;

  hdr.hk.regular    = 'r';
  hdr.hk.sizeof_hdr = sizeof(struct dsr);

  /* all Analyze images are taken as 4 dimensional */

  hdr.dime.dim[0] = 4;  
  hdr.dime.dim[1] = image3D.getSize().x;
  hdr.dime.dim[2] = image3D.getSize().y;
  hdr.dime.dim[3] = image3D.getSize().z;
  hdr.dime.dim[4] = 1;
  hdr.dime.pixdim[1] = image3D.getSpacing().x;
  hdr.dime.pixdim[2] = image3D.getSpacing().y;
  hdr.dime.pixdim[3] = image3D.getSpacing().z;

  hdr.dime.vox_offset = 0.0;
  hdr.hist.orient = (int)image3D.getOrientation(); 
  strcpy(hdr.dime.vox_units," ");
  strcpy(hdr.dime.cal_units," ");
  hdr.dime.cal_max = 0.0;
  hdr.dime.cal_min = 0.0;

// Depending on the datatype we fill hdr.dime

  switch(image3D.getDataType()){
  case Image<float>::UnsignedChar:
    hdr.dime.datatype = DT_UNSIGNED_CHAR;
    hdr.dime.bitpix   = 8;
    hdr.dime.glmax    = 255;
    hdr.dime.glmin    = 0;
    break;
  case Image<float>::UnsignedShort:
    // Analyze has no u_short
    hdr.dime.datatype = DT_SIGNED_SHORT;
    hdr.dime.bitpix   = 16;
    hdr.dime.glmax    = 32767;
    hdr.dime.glmin    = 0;
    break;
  case Image<float>::Float:
    hdr.dime.datatype = DT_FLOAT; 
    hdr.dime.bitpix   = 32;
    hdr.dime.glmax    = INT_MAX;
    hdr.dime.glmin    = INT_MIN;

    break;
  default:
    throw bException("ImageIO::saveAnalyzeHeader Switch default");
  }
  
  //SwapAnalyzeHeader((void*)&hdr);
  
// Last, we write hdr in the file

  outFile.write((const char*)&hdr,sizeof(struct dsr));
  
  if(outFile.fail()){
    throw bException("ImageIO::saveAnalyzeHeader Error");
  }
}

// SaveData write the data, dat, in a file depending on its type

void ImageIO::SaveData(Image<float> image3D, std::ostream &file)
{
  //SwabData(image3D);

  Array3D<u_char> u_char_dat;
  Array3D<u_short> u_short_dat;
  Array3D<float> float_dat;
  

  unsigned int k=0;

  unsigned int j =0;

  unsigned int i=0;
  switch(image3D.getDataType()){
  case Image<float>::UnsignedChar:
    u_char_dat.resize(image3D.getSize().x,image3D.getSize().y,image3D.getSize().z);
    for( k=0;k<image3D.getSize().x;k++)
      for(j=0;j<image3D.getSize().y;j++)
	for( i=0;i<image3D.getSize().z;i++)
	  u_char_dat(k,image3D.getSize().y-1-j,image3D.getSize().z-1-i)=(u_char)image3D(k,j,i);
    
    file.write((char*)u_char_dat.getDataPointer(),u_char_dat.getSizeBytes());
    
    break;
  case Image<float>::UnsignedShort:
    u_short_dat.resize(image3D.getSize().x,image3D.getSize().y,image3D.getSize().z);
    for(k=0;k<image3D.getSize().x;k++)
      for( j=0;j<image3D.getSize().y;j++)
	for(i=0;i<image3D.getSize().z;i++)
	  u_short_dat(k,image3D.getSize().y-1-j,image3D.getSize().z-1-i)=(u_short)image3D(k,j,i);

    file.write((char*)u_short_dat.getDataPointer(),u_short_dat.getSizeBytes());
    
    break;
  case Image<float>::Float:
    float_dat.resize(image3D.getSize().x,image3D.getSize().y,image3D.getSize().z);
    for(k=0;k<image3D.getSize().x;k++)
      for(j=0;j<image3D.getSize().y;j++)
	for(i=0;i<image3D.getSize().z;i++)
	  float_dat(k,image3D.getSize().y-1-j,image3D.getSize().z-1-i)=(float)image3D(k,j,i);
    
    file.write((char*)float_dat.getDataPointer(),float_dat.getSizeBytes());
    
    break;
  default:
    std::cerr << "Uh, what up?" << std::endl;
    break;
    SwabData(image3D);
    if(!file) throw bException("ImageIO::saveData file is empty ");
  }
}

// SaveRaw open the file where we'll save the data and call SaveData

void ImageIO::SaveRaw(string fname,Image<float> &image3D)
{
   //ofstream outFile(fname.c_str());
#ifdef WIN32
  ofstream outFile(fname.c_str(), std::ios::binary);
#else
  ofstream outFile(fname.c_str());
#endif
  if(outFile.fail()) {
    throw bException("ImageIO::saveRaw Error opening the file");
  }

    SaveData(image3D,outFile);
  
}

// Save analyse is the main function to save analyse images (3 parts)
//    1. Construct the name that will be use to save the image
//    2. Save the header
//    3. Save the data

void ImageIO::SaveAnalyze(string filename,Image<float> &image3D){
  
  char      imgname[MAXPATHLEN];
  char      hdrname[MAXPATHLEN];
  char      fprefix[MAXPATHLEN];

  ConstructAnalyzeNames(filename.c_str(),fprefix,hdrname,imgname);
  
//   cout<<"fname : "<<filename<<endl;
//   cout<<"fprefix : "<<fprefix<<endl;
//   cout<<"hdrname : "<<hdrname<<endl;
//   cout<<"imgname : "<<imgname<<endl;

  SaveAnalyzeHeader(hdrname,image3D);
  
#ifdef WIN32
  ofstream outFile(imgname, std::ios::binary);
#else
  ofstream outFile(imgname);
#endif

  if(outFile.fail()) {
    throw bException("ImageIO::SaveAnalyze Error");
  }

  SaveRaw(imgname,image3D);

}

/**********************************************************************/
/**********************************************************************/
/*                                                                    */
/*                         PLUNC dose grid                            */
/*                                                                    */
/**********************************************************************/
/**********************************************************************/


// Similar 3 steps to meta:
//    1. Load a grid object
//    2. Fill our header
//    3. Copy our data

void ImageIO::LoadDoseGrid(string filename,Image<float> &image3D)
{

  GRID g;

  // Open the file and load the data.
#ifdef WIN32 
  int fdes = open(filename.c_str(), O_RDONLY|O_BINARY, 0);
#else 
  int fdes = open(filename.c_str(), O_RDONLY, 0);
#endif
  if (fdes < 0) {
    throw bException("PlanIO::read: cannot open file for reading.");
  }
  if (read_grid(fdes, &g, false) == -1) {
    throw bException("PlanIO::read: error reading grid.");
  }
  close(fdes);

  // Fill header
  image3D.resize( g.x_count, g.y_count, g.z_count );
  image3D.setSpacing( g.inc.x, g.inc.y, g.inc.z );
  image3D.setOrigin( g.start.x, g.start.y, g.start.z );

  // By default we set the orientation to RPS.
  image3D.setOrientation( Image<float>::RPS );

  // Then we fill image3D.
  image3D.copyData( g.matrix );

}

void ImageIO::SaveDoseGrid(string filename, Image<float> &image3D)
{

  GRID g;

  g.x_count = image3D.getSizeX();
  g.y_count = image3D.getSizeY();
  g.z_count = image3D.getSizeZ();

  g.inc.x = image3D.getSpacingX();
  g.inc.y = image3D.getSpacingY();
  g.inc.z = image3D.getSpacingZ();

  g.start.x = image3D.getOriginX();
  g.start.y = image3D.getOriginY();
  g.start.z = image3D.getOriginZ();

  memset(&g.grid_to_pat_T[0][0], 0, 16 * sizeof(float));
  memset(&g.pat_to_grid_T[0][0], 0, 16 * sizeof(float));
  for (unsigned int i = 0; i < 4; ++i) {
    g.grid_to_pat_T[i][i] = g.pat_to_grid_T[i][i] = 1.0f;
  }

  Array3DUtils::getMinMax(image3D, g.min, g.max);

  g.matrix = image3D.getDataPointer();

  // Open the file and load the data.
#ifdef WIN32 
  int fdes = open(filename.c_str(), O_WRONLY|O_CREAT|O_TRUNC|O_BINARY);
#else 
  int fdes = open(filename.c_str(), O_WRONLY|O_CREAT|O_TRUNC,
                  S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
#endif
  if (fdes < 0) {
    throw bException("PlanIO::read: cannot open file for writing.");
  }
  if (write_grid(fdes, &g, false) == -1) {
    throw bException("PlanIO::read: error writing grid.");
  }
  close(fdes);

}

#endif

/**********************************************************************/
/**********************************************************************/
/*                                                                    */
/*                             PLAN IM                                */
/*                                                                    */
/**********************************************************************/
/**********************************************************************/


/**********************************************************************/
/*************************   LOAD PLANIM  *****************************/
/**********************************************************************/

// If the image loaded is a PlanIM, this is called and check if slices are
// regularly spaced. 
// If so it calls the function 
// loadPlanIM(vox_data, image3D, input_header, scans);
// If not, the function PlanIMaskParameters(...) is called and set the 
// different parameters needed to reslice the data. Then the corresponding
// loadPlanIM() function is called.

void ImageIO::LoadPlanIM(string filename,Image<float> &image3D){  

  try {
    IM_input_header = PlanIO::readHeader(filename.c_str());
  }
  catch (PlanIMIOException e) {
    cerr << e.getMessage() << endl;
    return;
  }

  // compute slice spacing
  IM_z_positions = getSlicePositions(IM_input_header);
  IM_z_diffs = getSliceDiffs(IM_input_header);

  // decide if slices are regularly spaced
  double epsilon = 0.001;
  bool regular_sliced = true;
  for (int i = 1; i < IM_input_header.slice_count - 1; i++) {
    if (fabs(IM_z_diffs[i] - IM_z_diffs[i-1]) > epsilon) {
      cout << IM_z_diffs[i] << " != "
           << IM_z_diffs[i - 1] << "[epsilon == " << epsilon << "]" << endl;
      regular_sliced = false;
      break;
    }
  }
  // debug output
  if (regular_sliced == true) {
    // float slice_spacing = IM_z_diffs[0];
    // cout << "plan_im has regular slice spacing: " << slice_spacing
    //      << "mm [to within " << epsilon << "mm]\n"
    //      << "will load as is" << endl;
  } else {
    cout << "plan_im has irregular slice spacing [epsilon="
         << epsilon << "]" << endl;
    for (int i = 0; i < IM_input_header.slice_count; i++) {
      cout << IM_z_positions[i];
      if (i < (IM_input_header.slice_count - 1)) {
        cout << " " <<IM_z_diffs[i];
      }
      else {
        cout << " n/a";
      }
      cout << endl;
    }
    cout << "requires regularization" << endl;
  }

  image3D.setDataType( Image<float>::UnsignedShort );

  if (regular_sliced == true) {
    // read image data
    PIXELTYPE* scans;
    try {
      scans = PlanIO::readScans(filename.c_str());
    }
    catch (PlanIMIOException e) {
      cerr << e.getMessage() << endl;
      return;
    }

    // load plan_im into Image<float>
    double temp_scale[3];
    double temp_origin[3];
    loadPlanIM_reg(image3D, IM_input_header, scans, temp_scale, temp_origin);

    // delete plan_im image data buffer
    delete [] scans;

    image3D.setSpacing(temp_scale[0],temp_scale[1],temp_scale[2]);
    image3D.setOrigin(temp_origin[0],temp_origin[1],temp_origin[2]);
    image3D.setOrientation(Image<float>::LAS);
   
  }

  else {
    // get new spacing parameters
    // set the PlanIm parameters of ImageIO
  
    guessParameters(IM_input_header.slice_count, IM_z_positions, IM_z_diffs,
		    IM_z_zero, IM_slice_thickness, IM_num_slices, 
		    IM_background_fill);
    throw bException("Load_PlanIM irregularly sliced");

  }
}

// Called once the user has set the new reslicing paramater 
// If there is no user interface part, the parameters used here
// will be the one obtained from guessParameters

void ImageIO::LoadPlanIMIrreg(string filename, Image<float> &image3D){
  
  double temp_scale[3];
  double temp_origin[3];

// Fill image3D with data corresponding to the new reslice

  loadPlanIM_irreg(image3D, filename.c_str(), IM_z_zero, IM_slice_thickness,
                   IM_num_slices,  IM_background_fill, temp_scale,  temp_origin);
  
// We set the header

 image3D.setSpacing(temp_scale[0],temp_scale[1],temp_scale[2]);

 image3D.setOrigin(temp_origin[0],temp_origin[1],temp_origin[2]);
}



/**********************************************************************/
/*************************   SAVE PLANIM  *****************************/
/**********************************************************************/


void ImageIO::getMinMax(float &minret, float &maxret,Image<float> image3D)
{
  // u_char  cval,cmin,cmax,*cptr;
//   u_short sval,smin,smax,*sptr;
  float   fval,fmin,fmax,*fptr;
  int     i,sz;

   //vector<int> dimension = image3D.getDimensions();

  vector<int> dimension(3);

  dimension[0]= image3D.getSizeX();
  dimension[1]= image3D.getSizeY();
  dimension[2]= image3D.getSizeZ();
   
   sz = (image3D.getSizeX())*(image3D.getSizeY())*(image3D.getSizeZ());
   
//   switch(data_type) {
//   case ItXVolume::UnsignedChar:
//     cptr = u_char_dat.data();
//     cmin = cmax = *cptr;
//     for(i=0;i<sz;i++,cptr++) {
//       cval = *cptr;
//       if(cmin > cval) cmin = cval;
//       if(cmax < cval) cmax = cval;
//     }
//     minret = (float)cmin;
//     maxret = (float)cmax;
//     break;
//   case ItXVolume::UnsignedShort:
//     sptr = u_short_dat.data();
//     smin = smax = *sptr;
//     for(i=0;i<sz;i++,sptr++) {
//       sval = *sptr;
//       if(smin > sval) smin = sval;
//       if(smax < sval) smax = sval;
//     }
//     minret = (float)smin;
//     maxret = (float)smax;
//     break;
//   case ItXVolume::Float:

    fptr = image3D.getDataPointer();
    fmin = fmax = *fptr;
    for(i=0;i<sz;i++,fptr++) {
      fval = *fptr;
      if(fmin > fval) fmin = fval;
      if(fmax < fval) fmax = fval;
    }
    minret = fmin;
    maxret = fmax;

//     break;
//   default:
//     break;
//   }
}


void ImageIO::convertToUnsignedShort(Image<float> image3D,Array3D<u_short> &u_short_dat)
{
  u_short *sptr;
  //  u_char  *cptr;
  float    fmin,fmax,frange,*fptr;
  int      i,j,k;

  //vector<int> dimension = image3D.getDimensions();
  vector<int> dimension(3);
  dimension[0]= image3D.getSizeX();
  dimension[1]= image3D.getSizeY();
  dimension[2]= image3D.getSizeZ();
  //  Vector3D<double> scale = image3D.getVoxelScale();

  if(image3D.getDataType() != Image<float>::UnsignedShort) {
    u_short_dat.resize(dimension[0],dimension[1],dimension[2]);
    if(u_short_dat.isEmpty()) {
      throw bException("ERROR: ImageIO::convertToUnsignedShort Out of memory!");
    }
  }

  //  switch(image3D.getDataType()) {
//    case Image<float>::UnsignedShort:
//      break;
//    case Image<float>::UnsignedChar:
//     sptr = u_short_dat.data();
//     cptr = u_char_dat.data();
//     for(k=0;k<dimension[0];k++)
//       for(j=0;j<dimension[1];j++)
// 	for(i=0;i<dimension[2];i++,cptr++,sptr++) 
// 	  *sptr = (u_short) *cptr;
//     u_char_dat.setDim(0,0,0);
//      break;
//    case Image<float>::Float:

    this->getMinMax(fmin,fmax,image3D);
    frange = fmax - fmin;
    if(frange == 0.0) frange = 1.0;
    fptr = image3D.getDataPointer();
    sptr = u_short_dat.getDataPointer();
    for(k=0;k<dimension[0];k++)
      for(j=0;j<dimension[1];j++)
	for(i=0;i<dimension[2];i++,fptr++,sptr++) 
	  *sptr = (u_short)((*fptr-fmin)/frange * 
			    (float)USHRT_MAX + 0.5);

    //  break;
//    default: break;
//    }
   image3D.setDataType( Image<float>::UnsignedShort);

}

void ImageIO::convertToShort(Image<float> image3D,Array3D<short> &short_dat)
{
  short *sptr;
  //  u_char  *cptr;
  float    fmin,fmax,frange,*fptr;
  int      i,j,k;

  //vector<int> dimension = image3D.getDimensions();
  vector<int> dimension(3);
  dimension[0]= image3D.getSizeX();
  dimension[1]= image3D.getSizeY();
  dimension[2]= image3D.getSizeZ();
  //  Vector3D<double> scale = image3D.getVoxelScale();

  if(image3D.getDataType() != Image<float>::Short) {
    short_dat.resize(dimension[0],dimension[1],dimension[2]);
    if(short_dat.isEmpty()) {
      throw bException("ERROR: ImageIO::convertToShort Out of memory!");
    }
  }

  //  switch(image3D.getDataType()) {
//    case Image<float>::UnsignedShort:
//      break;
//    case Image<float>::UnsignedChar:
//     sptr = u_short_dat.data();
//     cptr = u_char_dat.data();
//     for(k=0;k<dimension[0];k++)
//       for(j=0;j<dimension[1];j++)
// 	for(i=0;i<dimension[2];i++,cptr++,sptr++) 
// 	  *sptr = (u_short) *cptr;
//     u_char_dat.setDim(0,0,0);
//      break;
//    case Image<float>::Float:

    this->getMinMax(fmin,fmax,image3D);
    frange = fmax - fmin;
    if(frange == 0.0) frange = 1.0;
    fptr = image3D.getDataPointer();
    sptr = short_dat.getDataPointer();
    for(k=0;k<dimension[0];k++){
		for(j=0;j<dimension[1];j++){
		  for(i=0;i<dimension[2];i++,fptr++,sptr++) {
	 // *sptr = (short)((*fptr-fmin)/frange * 
	 //		    (float)SHRT_MAX + 0.5);
			  *sptr = (short) (*fptr);
	}}}

    //  break;
//    default: break;
//    }
   image3D.setDataType( Image<float>::Short);

}











//savePlan_im(const char *fname, float offset_z)

void ImageIO::SavePlanIM(string filename,Image<float> &image3D, float offset_z)
{
  
  plan_im_header header;
  int fd,i,j,k,loop;
  int slicecnt;
  short *out_scans;
  float fmin,fmax,pz;
  
  for(i=0;i<4;i++)
    for(j=0;j<4;j++)
      header.pixel_to_patient_TM[i][j] = 0.0;
	for(k=0 ; k<4 ; k++)
		header.pixel_to_patient_TM[k][k] = 1.0;

  getMinMax(fmin,fmax,image3D);

  //vector<int> dimension = image3D.getDimensions();

    vector<int> dimension(3);

  dimension[0]= image3D.getSizeX();
  dimension[1]= image3D.getSizeY();
  dimension[2]= image3D.getSizeZ();
  Vector3D<double> scale = image3D.getSpacing();
  
  strcpy(header.unit_number,"no unit");
  strcpy(header.patient_name,"no name");
  strcpy(header.comment,"no comment");

  header.x_dim            = dimension[0];
  header.y_dim            = dimension[1];
  header.slice_count      = dimension[2];
  header.resolution       = dimension[0];// should be x
  header.x_size           = scale.x;//libplanio version 6
  header.y_size           = scale.y;//libplanio version 6
  header.pixel_size       = scale.x;
  header.table_height     = 0;
  header.date.day         = 1;
  header.date.month       = 1;
  header.date.year        = 2002;
  header.date.dow         = 0;
  header.machine_id       = (enum scanners)0;
  header.patient_position = (enum position_list)0;
  header.whats_first      = (enum entry_list)0;
  header.pixel_type       = (enum pixel_types)0; 
  header.min              = (int)fmin;
  header.max              = (int)fmax;
  header.pixel_to_patient_TM[0][0] = scale.x;
  header.pixel_to_patient_TM[1][1] = scale.y;
  header.pixel_to_patient_TM[3][3] = 1.0;
  header.pixel_to_patient_TM[3][0] = image3D.getOrigin().x;
  header.pixel_to_patient_TM[3][1] = image3D.getOrigin().y;
  
  pz = scale.z;
  
  for(i=0;i<dimension[2];i++) {
    header.per_scan[i].gantry_angle = 0.0;
    header.per_scan[i].table_angle  = 0.0;	
    header.per_scan[i].offset_ptrs  = 0;
    header.per_scan[i].scan_number  = i;
    header.per_scan[i].z_position   = image3D.getOrigin().z   + i*pz;
  }

#ifdef WIN32
  fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC|O_BINARY, 0664);
#else
  fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0664);
#endif

  if (fd < 0) {
    throw bException("SavePlanIM File open error ");
  }
  
  if (write_image_header(fd, &header)) {
    throw bException("SavePlanIM Write header error ");
  }
  
  slicecnt = header.x_dim*header.y_dim;
  
  Array3D<short> short_dat;

  convertToShort(image3D, short_dat);
  out_scans = short_dat.getDataPointer();

  if(!out_scans){
    throw bException("SavePlanIM Error accessing u_short_data");
  }
 
  for (loop = 0; loop < header.slice_count; loop++) {
    header.per_scan[loop].offset_ptrs = lseek(fd, 0, SEEK_CUR);
    if (write_scan_xy(fd, (short int *)(out_scans + loop*slicecnt),
		      header.per_scan[loop].offset_ptrs,
 		      header.x_dim, header.y_dim)) {
      throw bException("SavePlanIM Write Scan error ");
    }
  }

  lseek(fd, 0, 0);  
  write_image_header(fd, &header);
  close(fd);
  
}






/**********************************************************************/
/**********************************************************************/
/*                                                                    */
/*                             DICOM                                  */
/*                                                                    */
/**********************************************************************/
/**********************************************************************/



/**********************************************************************/
/*************************   LOAD DICOM  ******************************/
/**********************************************************************/

// Load Dicom open the dicom file and set the DICOM parameters of Image IO
// (DICOM and nb_DICOM) from this parameter the user will be able to choose
// the dicom he wants.

void ImageIO::LoadDicom(string filename){

  nb_DICOM=dicomImage.OpenDICOMFile((char *)filename.c_str());
  
}

bool ImageIO::CheckSpacingSelectedDicom(int selected_DICOM)
{
_selected_DICOM = selected_DICOM;
if (selected_DICOM>-1){
    
    dicomImage.SelectImage(selected_DICOM);
	UniformSpacing = dicomImage.CheckUniformSpacing();
	}
else throw bException("Load_DICOM ERROR");
return UniformSpacing;
}
 
void ImageIO::LoadSelectedDicom(Image<float> &image3D, float newSpacing){
  std::cerr << "ImageIO::LoadSelectedDicom" << std::endl;
  if (_selected_DICOM>-1){
  
// We set the header
  
    image3D.resize(dicomImage.getsizeX(),dicomImage.getsizeY(),dicomImage.getsizeZ());
	
	image3D.setOrigin((-dicomImage.getsizeX()*dicomImage.PixsizeX())/2.0,-dicomImage.getsizeY()*dicomImage.PixsizeY()/2.0,dicomImage.get_Zpos(0));
	image3D.setOrigin(image3D.getOrigin());
	image3D.setSpacing(dicomImage.PixsizeX(),dicomImage.PixsizeY(),dicomImage.PixsizeZ());
	image3D.setSpacing(image3D.getSpacing());
	image3D.setDataType( Image<float>::UnsignedShort );
  image3D.setOrientation(Image<float>::strToImageOrientation(dicomImage.get_orient()));
  // TODO: apparently, orientation is assumed to be LAS
  //  we need to check if the dicom is in this position, and
  //  if not then we need to reorder our data.

   //  sprintf(header.unit_number,dicomImage.get_unit_number());
//     sprintf(header.patient_name,dicomImage.get_name());
//     sprintf(header.comment,dicomImage.get_comment());
//     sprintf(header.machine_id,dicomImage.get_machine_ID());
//     sprintf(header.image_type,dicomImage.get_image_type());
//     sprintf(header.modality,dicomImage.get_modality());
//     sprintf(header.study_time,dicomImage.get_study_time());
//     header.z_offset=dicomImage.get_Zoffset();
//     header.min=dicomImage.get_min();
//     header.max=dicomImage.get_max();
    

// To get the date

   //  char *date_piece;
//     char date[12];
//     sprintf(date,dicomImage.get_Date());
//     date_piece=strtok((char *)date,"/");   
//     header.date.day=atoi(date_piece);
//     date_piece = strtok(NULL,"/");
//     header.date.month=atoi(date_piece);
//     date_piece = strtok(NULL,"/");
//     header.date.year=atoi(date_piece);

// Here we fill image3D... at last !

    Array3D<u_short> u_short_dat;
    
    if((image3D.getSize().z>0)&&(image3D.getSize().y>0)&&(image3D.getSize().x>0)){
      u_short_dat.resize(image3D.getSize().x,image3D.getSize().y,image3D.getSize().z);
      dicomImage.LoadTheImage(u_short_dat.getDataPointer());
    }
    
    short temp;
    for (unsigned int k=0;k<image3D.getSize().x;k++){
		for (unsigned int j=0;j<image3D.getSize().y;j++){
			for (unsigned int i=0;i<image3D.getSize().z;i++){
				temp = (u_short) u_short_dat(k,j,i);
				//temp = (u_short) u_short_dat(k,image3D.getSize().y-1-j,i);
				//temp = (u_short) u_short_dat(k,image3D.getSize().y-1-j,image3D.getSize().z-1-i);
				image3D.set(k, j, i, temp);
			}
		}
    }

    image3D.toOrientation( Image<float>::RPS );

    if (UniformSpacing ==  false)
	{

	_dicomReslicing(image3D,newSpacing);
	image3D.setSpacing(dicomImage.PixsizeX()/10,dicomImage.PixsizeY()/10,newSpacing/10);	
	}
  }
  else throw bException("Load_DICOM ERROR");

	
}

// To load dicomContour

void ImageIO::LoadDicomContour(DICOMcontour *dcont ){
  
  dcont->set_ref(dicomImage.getObject()); 
  dcont->OpenDICOMFile();
  
  
}

//reslice the image with the newSpacing
void ImageIO::_dicomReslicing(Image<float> &image3D, float newSpacing)
{
	//IMPORTANT : we need a newSpacing positive in this function
	// the orientation is not important, that's why we use abs()
	
	//determine the new number of slices
	//create the new image resliced
	
	unsigned int ImageSizeX = image3D.getSizeX();
	unsigned int ImageSizeY = image3D.getSizeY();

 	//float oldSpacing = dicomImage.PixsizeZ();
	float firstSlice = dicomImage.get_Zpos(0);
	float lastSlice = dicomImage.get_Zpos((image3D.getSizeZ() - 1));
	
	int newNb_Slice = int( ( fabs(lastSlice) - fabs(firstSlice) ) / fabs(newSpacing) );
	newNb_Slice+=1;//to count the first slice
	
	Vector3D<unsigned int> newSize(ImageSizeX,ImageSizeY,newNb_Slice);
	Image<float> image3DResliced(newSize);

	/*int newZ = 0 ; 
	
	for (newZ = 0; newZ < newNb_Slice; newZ++){
		float oldZ = (newZ*newSpacing)/oldSpacing;

		int oldz1 = floor(oldZ);
		int oldz2 = oldz1+1;
				for (int y=0 ; y < ImageSizeY ; y++){
					for (int x=0 ; x < ImageSizeX ; x++){
						image3DResliced.set(x,y,newZ,(image3D.get(x,y,oldz1)*(oldz2-oldZ)+image3D.get(x,y,oldz2)*(oldZ-oldz1)));
				}}
			
	}*/

	float currentSlice_Zpos = dicomImage.get_Zpos(0);
	float threshold = currentSlice_Zpos;
	float distance;
	
	int z = 0;
	int newZ = 0 ; 
	distance = fabs(currentSlice_Zpos) - fabs(threshold);	
			
	do{//loop for the new image
		
		do{
			//same slice
			if(distance==0.0){
				for (unsigned int y=0 ; y < ImageSizeY ; y++){
					for (unsigned int x=0 ; x < ImageSizeX ; x++){
						image3DResliced.set(x,y,newZ,image3D.get(x,y,z));
				}}
				threshold = fabs(threshold) + fabs(newSpacing);
				newZ++;
			}
			
			//need to create an interpolate slice
			if(distance>0){
				for (unsigned int y=0 ; y < ImageSizeY ;y++){
					for (unsigned int x=0 ; x < ImageSizeX ; x++){
						
						//linear interpolation
						//weight
						float w1 = fabs(threshold  - fabs(dicomImage.get_Zpos(z-1)));
						float w2 = fabs(threshold - fabs(currentSlice_Zpos));
						//intensity
						float i1 = image3D.get(x,y,z-1);
						float i2 = image3D.get(x,y,z);
						
						float newIntensity = ((i1 * w2) + (i2 * w1))/ (w1 + w2);
						image3DResliced.set(x,y,newZ,newIntensity);		
				}}
				threshold = fabs(threshold) + fabs(newSpacing);
				newZ++;
			}
			distance = fabs(currentSlice_Zpos) - fabs(threshold) ;	
			
		}while (distance>=0);
		z++;
		currentSlice_Zpos = fabs(dicomImage.get_Zpos(z));
		distance = fabs(currentSlice_Zpos) - fabs(threshold) ;
	}while( newZ < newNb_Slice );
	


	//reset image3D
	image3D.resize(ImageSizeX,ImageSizeY,newNb_Slice);
	image3D.setData(image3DResliced);
}


/**********************************************************************/
/**********************************************************************/
/*                                                                    */
/*                             RTOG                                   */
/*                                                                    */
/**********************************************************************/
/**********************************************************************/


/**********************************************************************/
/*************************   LOAD RTOG ********************************/
/**********************************************************************/

void ImageIO::LoadRTOGHeader(string filename){
}

void ImageIO::LoadRTOGScan(){
}

void ImageIO::LoadRTOGStructure(int structnum){
}

void ImageIO::LoadRTOGDose(){
}


/**********************************************************************/
/*************************   SAVE RTOG ********************************/
/**********************************************************************/

void ImageIO::SaveRTOG(string RTOGDir)
{ // need to pass scan, dose, structures
}
