#ifndef SLICE_FUNCTIONS_H
#define SLICE_FUNCTIONS_H


#include "PlanIO.h"
#include <stdlib.h>
#include <stdio.h>
#include <Array3D.h>
#include <iostream>
#include <math.h>
//#include "/uc/kuhn/mysnap/VoxData.h"


extern void loadPlanIM_reg(Array3D<float>& image3D, const plan_im_header& header, const PIXELTYPE* scans,double* temp_scale, double* temp_origin);

extern void loadPlanIM_irreg(Array3D<float>& image3D, const char* filename,
		float z_zero, float slice_thickness, 
		       int num_slices, int background_fill,double* temp_scale, double* temp_origin); 




extern int floatCompare(const void* p1, const void* p2);

extern void guessParameters(int input_slice_count, float* z_positions, float* z_diffs,
			    float& z_zero, float& slice_thickness, int& output_slice_count, PIXELTYPE background_fill);

extern float* getSlicePositions(const plan_im_header& header);

extern float* getSliceDiffs(const plan_im_header& header);



#endif

