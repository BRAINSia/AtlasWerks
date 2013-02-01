#include "PlanIO.h"
#include <iostream>
#include <float.h>

#ifdef WIN32

#include <io.h>

#endif

using std::cout;
using std::cerr;
using std::endl;


/*
 * allocate memory necessary for header's image data
 */
PIXELTYPE* PlanIO::allocateScans(const plan_im_header& header) {
  int pix_per_scan = header.x_dim * header.y_dim;
  int pix_total = pix_per_scan * header.slice_count;
  return new PIXELTYPE[pix_total];
}

/*
 * read image data from file descriptor into array according
 * to parameters in header
 */  
int PlanIO::readScans(int fdes, 
		      const plan_im_header& header,
		      PIXELTYPE* scans) {
  int pix_per_scan = header.x_dim * header.y_dim;
  for (int i = 0; i < header.slice_count; i++) {
    if (read_scan_xy(fdes, scans + i*pix_per_scan,
		     header.per_scan[i].offset_ptrs,
		     header.x_dim, header.y_dim)) {
      return i;
    }
  }
  return header.slice_count;
}

plan_im_header PlanIO::readHeader(const char* filename) {
  plan_im_header header;
  #ifdef WIN32 
  int fdes = open(filename, O_RDONLY|O_BINARY, 0);
  #else 
  int fdes = open(filename, O_RDONLY, 0);
  #endif
  if (fdes < 0) {
    throw PlanIMIOException("PlanIO::read: cannot open file for reading.");
  }
  if (read_image_header(fdes, &header)) {
    throw PlanIMIOException("PlanIO::read: error reading header.");
  }
  return header;
}

PIXELTYPE* PlanIO::readScans(const char* filename) {
  plan_im_header header = readHeader(filename);
  PIXELTYPE* scans = allocateScans(header);

  #ifdef WIN32 
  int fdes = open(filename, O_RDONLY|O_BINARY, 0);
  #else 
  int fdes = open(filename, O_RDONLY, 0);
  #endif
  if (fdes < 0) {
    throw PlanIMIOException("PlanIO::read: cannot open file for reading.");
  }
  if (readScans(fdes, header, scans) != header.slice_count) {
    throw PlanIMIOException("PlanIO::read: error reading scans.");
  }
  close(fdes);

  return scans;
}

/* 
 * read header and image data from file
 */
void PlanIO::read(const char* file,
		  plan_im_header& header,
		  PIXELTYPE*& scans) {
  #ifdef WIN32 
  int fdes = open(file, O_RDONLY|O_BINARY, 0);
  #else 
  int fdes = open(file, O_RDONLY, 0);
  #endif
  if (fdes < 0) {
    throw PlanIMIOException("PlanIO::read: cannot open file for reading.");
  }
  if (read_image_header(fdes, &header)) {
    throw PlanIMIOException("PlanIO::read: error reading header.");
  }
  
  scans = allocateScans(header);
  if (readScans(fdes, header, scans) != header.slice_count) {
    throw PlanIMIOException("PlanIO::read: error reading scans.");
  }
  close(fdes);
}

/* 
 * write header and image data to file
 */
void PlanIO::write(const char* file,
		   const plan_im_header& header,
		   const PIXELTYPE* scans) {
  // we need to disregard const for write_image_header and write_scan_xy
  plan_im_header header_copy = (plan_im_header) header;
  PIXELTYPE* scans_copy = (PIXELTYPE*) scans;

  // open file
  int fdes = open(file, O_RDWR | O_CREAT | O_TRUNC, 0664);
  if (fdes < 0) {
    throw PlanIMIOException("PlanIO::write: cannot open file for writing.");
  }
 
  // write header w/ bogus offset pointers
  if (write_image_header(fdes, &header_copy)) {
    throw PlanIMIOException("PlanIO::write: error writing header.");
  }
  
  // write image data, save offset pointers
  int pix_per_slice = header.x_dim * header.y_dim;
  for (int i = 0; i < header.slice_count; i++) {
    header_copy.per_scan[i].offset_ptrs = lseek(fdes, 0, SEEK_CUR);
    if (write_scan_xy(fdes, scans_copy + i * pix_per_slice,
		      header_copy.per_scan[i].offset_ptrs,
		      header_copy.x_dim, header_copy.y_dim)) {
      throw PlanIMIOException("PlanIO::write: error writing image data.");
    }
  }

  // rewind and write header with true offset pointers
  lseek(fdes, 0, 0);
  if (write_image_header(fdes, &header_copy)) {
    throw PlanIMIOException("PlanIO::write: error writing header.");
  } 
  close(fdes);
}

void PlanIO::resample(const plan_im_header& input_header,
                      const PIXELTYPE* const input_scans,
		      const plan_im_header& regular_header,
                      PIXELTYPE* regular_scans,
		      PIXELTYPE background_fill) {
  if (PLANIO_DEBUG_OUTPUT) {
    std::cout << "[slice(z), input_slice_le(z), input_slice_ge(z), operation]"
              << std::endl;
  } 
  int outdex = 0;
  for (int i = 0; i < regular_header.slice_count; i++) {
    // get next slice position to interpolate
    float z = regular_header.per_scan[i].z_position;

    // get closest input slice before and after z
    // don't assume input_header slice positions are sorted

    // le is "<=", closest input slice distance w/ position <= z
    float min_dist_le = FLT_MAX;
    // ge is >=, closest input slice distance w/ position >= z
    float min_dist_ge = FLT_MAX; 
    int le_index = -1, ge_index = -1; // index of closest slices
    bool found_le = false, found_ge = false;
    for (int j = 0; j < input_header.slice_count; j++) {
      float z_in = input_header.per_scan[j].z_position;
      if ((z_in <= z) && ((z - z_in) < min_dist_le)) {
	// found closest le so far
	min_dist_le = z - z_in;
	le_index = j;
	found_le = true;
      }
      if ((z_in >= z) && ((z_in - z) < min_dist_ge)) {
	// found closest ge so far
	min_dist_ge = z_in - z;
	ge_index = j;
	found_ge = true;
      }
    }

    if (PLANIO_DEBUG_OUTPUT) {
      cerr << "[" << i << "(" << z << "), ";
      if (found_le) {
	cerr << le_index << "(" << input_header.per_scan[le_index].z_position << "), ";
      }
      else {
	cerr << "n/a, ";
      }

      if (found_ge) {
	cerr << ge_index << "(" << input_header.per_scan[ge_index].z_position << "), ";
      }
      else {
	cerr << "n/a, ";
      }
    }

    if (!found_le || !found_ge) {
      if (PLANIO_DEBUG_OUTPUT) {
	cerr << "background(" << background_fill << ")]" << endl;
      }
      // write background value to all pixels in slice
      for (int i = 0, n = regular_header.x_dim * regular_header.y_dim; i < n; i++) {
	regular_scans[outdex++] = background_fill;
      }
    }
    else if (le_index == ge_index) {
      if (PLANIO_DEBUG_OUTPUT) {
	cerr << "copy]" << endl;
      }
      // copy verbatum an input slice
      int input_index = ge_index * input_header.x_dim * input_header.y_dim;
      for (int i = 0, n = regular_header.x_dim * regular_header.y_dim; i < n; i++) {
	regular_scans[outdex++] = input_scans[input_index++];
      }
    }
    else { 
      // do interpolation between input slices
      float d_le = z - input_header.per_scan[le_index].z_position; // dist between z and closest le slice
      float d_total = input_header.per_scan[ge_index].z_position - // dist between closest ge and le slices
	input_header.per_scan[le_index].z_position;
      float weight_ge = d_le / d_total; 
      float weight_le = 1.0 - weight_ge;

      if (PLANIO_DEBUG_OUTPUT) {
	cerr << "interpolate(weight_le=" << weight_le 
	     << ", weight_ge=" << weight_ge << ")]" << endl;
      }

      int le_pixel_index = le_index * input_header.x_dim * input_header.y_dim;
      int ge_pixel_index = ge_index * input_header.x_dim * input_header.y_dim;
      for (int i = 0, n = regular_header.x_dim * regular_header.y_dim; i < n; i++) {
	float val = weight_le * ((float) input_scans[le_pixel_index++]) 
	  + weight_ge * ((float) input_scans[ge_pixel_index++]);
	// round instead of truncate (i.e. + .5)
	regular_scans[outdex++] = (PIXELTYPE) (val + .5);
      }
    }
  }
}
