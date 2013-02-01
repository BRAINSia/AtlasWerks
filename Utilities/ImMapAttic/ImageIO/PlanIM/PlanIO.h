#ifndef PlanIO_h
#define PlanIO_h

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#ifndef WIN32
#include <unistd.h>
#endif
#include "plan_file_io.h"
#include "plan_strings.h"
#include "gen.h"
#include <plan_im.h>
//#include <lib3d.h>
#include <libplanio.h>
#include <iostream>
//#include <strings.h>
#include <climits>

class PlanIMIOException {
private:
  const char* message;

public:
  PlanIMIOException(const char* message) {
    this->message = message;
  }

  const char* getMessage() const {
    return strdup(message);
  }
};

const bool PLANIO_DEBUG_OUTPUT = false;
class PlanIO {
public:
  static PIXELTYPE* allocateScans(const plan_im_header& header);
  static int readScans(int fdes, 
		       const plan_im_header& header, 
		       PIXELTYPE* scans);
  static void read(const char* file, 
		   plan_im_header& header, 
		   PIXELTYPE*& scans);
  static plan_im_header readHeader(const char* filename);
  static PIXELTYPE* readScans(const char* filename);
  static void write(const char* file, 
		    const plan_im_header& header,
		    const PIXELTYPE* scans);
  static void resample(const plan_im_header& input_header, const PIXELTYPE* const input_scans,
		       const plan_im_header& regular_header, PIXELTYPE* regular_scans,
		       PIXELTYPE background_fill);
};

#endif
