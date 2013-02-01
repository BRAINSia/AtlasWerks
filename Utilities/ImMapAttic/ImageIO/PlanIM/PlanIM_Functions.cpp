
// Here we define some functions to load PlanIM images


#include "PlanIM_Functions.h"
#include <iostream>

using std::cout;
using std::endl;


extern void loadPlanIM_reg( Array3D<float>& image3D, const plan_im_header& header,
                            const PIXELTYPE* scans,double* temp_scale,
                            double* temp_origin){  
  // copy over header data
  image3D.resize(header.x_dim, header.y_dim, header.slice_count);

  float x_scale = header.pixel_to_patient_TM[0][0];
  float y_scale = header.pixel_to_patient_TM[1][1];

  /*float x_scale = header.x_size;
  float y_scale = header.y_size;*/

  float x_origin = header.pixel_to_patient_TM[3][0];
  float y_origin = header.pixel_to_patient_TM[3][1];

  temp_scale[0]=x_scale; //cm->mm
  temp_scale[1]=y_scale; //cm->mm

  temp_origin[0]=x_origin; //cm->mm
  temp_origin[1]=y_origin; //cm->mm


  if (header.slice_count >= 2) {
    float slice_dist = header.per_scan[1].z_position -
                       header.per_scan[0].z_position;
    temp_scale[2]=slice_dist; //cm->mm
	temp_origin[2]=header.per_scan[0].z_position;
  }
  else {
    temp_scale[2]=0;
	temp_origin[2]=header.per_scan[0].z_position;
  }

  // copy over image data
  int index = 0;
  double temp,color_min,color_max;
  color_min=color_max=0;
  for (int z = 0; z < header.slice_count; z++) {
    for (int y = 0; y < header.y_dim; y++) {
      for (int x = 0; x < header.x_dim; x++) {
	temp= scans[index++];
 	image3D.set(x, y, z, temp);
	if (temp>color_max) color_max=temp;
	if (temp<color_min) color_min=temp;
	
      }
    }
  }
}


extern void loadPlanIM_irreg( Array3D<float>& image3D, const char* filename,
		float z_zero, float slice_thickness, 
		int num_slices, int background_fill,
		double* temp_scale, double* temp_origin) {
  // read input plan_im
  plan_im_header input_header;
  PIXELTYPE* input_scans;
  try {
    PlanIO::read(filename, input_header, input_scans);
  }
  catch (PlanIMIOException e) {
    cout << e.getMessage() << endl;
    return;
  }

  // update regular_header & regular_scans
  plan_im_header regular_header = input_header;
  regular_header.slice_count = num_slices;
  for (int i = 0; i < num_slices; i++) {
    regular_header.per_scan[i].z_position = i * slice_thickness + z_zero;
  }    
  PIXELTYPE* regular_scans = PlanIO::allocateScans(regular_header);

  // resample image
  PlanIO::resample(input_header, input_scans, regular_header, regular_scans, background_fill);  

  // load plan_im into Array3D
  loadPlanIM_reg(image3D, regular_header, regular_scans,temp_scale, temp_origin);

  // delete plan_im image data buffer
  delete [] input_scans;
  delete [] regular_scans;
}

extern int floatCompare(const void* p1, const void* p2) {
  float f1 = *((float*) p1);
  float f2 = *((float*) p2);
  if (f1 > f2) {
    return 1;
  }
  if (f1 < f2) {
    return -1;
  }
  return 0;
}

extern void guessParameters(int input_slice_count, float* z_positions, float* z_diffs,
		     float& z_zero, float& slice_thickness, int& output_slice_count, PIXELTYPE background_fill) {
  // start at first input slice
  z_zero = z_positions[0];
  
  // thickness is second smallest difference between input slices
  float min_diff = z_diffs[0];
  float min_2_diff = z_diffs[0];
  for (int i = 1; i < input_slice_count - 1; i++) {
    if (z_diffs[i] < min_diff) {
      min_2_diff = min_diff;
      min_diff = z_diffs[i];
    }
    else if (z_diffs[i] < min_2_diff) {
      min_2_diff = z_diffs[i];
    }
  }
  slice_thickness = min_2_diff;

  // add slices while in input scan range
  output_slice_count = 0;
  float current_position = z_zero;
  float max_position = z_positions[input_slice_count - 1];
  while (current_position <= max_position) {
    current_position += slice_thickness;
    output_slice_count++;
  }

  background_fill = 0;
}


extern float* getSlicePositions(const plan_im_header& header) {
  float* z_positions = new float[header.slice_count];
  for (int i = 0; i < header.slice_count; i++) {
    z_positions[i] = header.per_scan[i].z_position;
  }
  qsort(z_positions, header.slice_count, sizeof(float), floatCompare);
  return z_positions;
}

extern float* getSliceDiffs(const plan_im_header& header) {
  float* z_positions = getSlicePositions(header);
  float* z_diffs = new float[header.slice_count];
  for (int i = 1; i < header.slice_count; i++) {
    z_diffs[i-1] = z_positions[i] - z_positions[i-1];
  }
  // last diff is bogus, set to zero
  z_diffs[header.slice_count - 1] = 0;
  return z_diffs;
}


