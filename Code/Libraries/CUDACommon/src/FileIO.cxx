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

#include <fstream>
#include <math.h>
#include "FileIO.h"
#include <sstream>
#include <stdexcept>
#include <stdlib.h>


void writePGM(char* rawFile, unsigned char* data, int width, int height) {
    std::ofstream fo(rawFile, std::ios::binary);
    fo << "P5"  << std::endl;
    fo << width << " " << height << " " << 255 << std::endl;
    fo.write((char*)data, width * height);
    fo.close();
}

////////////////////////////////////////////////////////////////////////////////
// My Hacking version to read the data 
////////////////////////////////////////////////////////////////////////////////
void readRawVolumeData(char* rawFile, float* data, int len){
    fprintf(stderr, "Reading file %s ", rawFile);
    FILE* pFile = fopen(rawFile, "rb");
    
    uint result = fread(data, sizeof(float), len , pFile);

    if (result != (uint)len)
        fprintf(stderr, "Reading error ");
    fclose(pFile);
    fprintf(stderr, ".. done \n");
}


void writeRawVolumeData_f(char* rawFile, float* data, int len) {
    std::ofstream fo(rawFile, std::ios::binary);
    fo.write((char*)data, len * sizeof(data[0]));
    fo.close();
}

////////////////////////////////////////////////////////////////////////////////
// My Hacking version to write data to file
////////////////////////////////////////////////////////////////////////////////
void writeToNrrd(float* data, int w, int h, int l, char *name) {
    char buffer[255];
    // Write the header file
    sprintf(buffer,"%s.nhdr",name);
    std::ofstream fo(buffer, std::ios::binary);
    sprintf(buffer,"%s.raw",name);

    fo << "NRRD0004" << std::endl;
    fo << "# Complete NRRD file format specification at: "<< std::endl;
    fo << "# http://teem.sourceforge.net/nrrd/format.html"<< std::endl;
    fo << "content: resample "<< std::endl;
    fo << "type: float"<< std::endl;
    fo << "dimension: 3"<< std::endl;
    fo << "space: left-posterior-superior"<< std::endl;
    fo << "sizes: " <<w << " " << h << " " << l << std::endl;
    fo << "space directions: (1,0,0) (0,1,0) (0,0,1) "<< std::endl;
    fo << "centerings: cell cell cell"<< std::endl;
    fo << "kinds: domain domain domain"<< std::endl;
    fo << "endian: little"<< std::endl;
    fo << "encoding: raw"<< std::endl;
    fo << "space origin: (0,0,0)"<< std::endl;
    fo << "data file: " << buffer << std::endl;
    fo.close();

    unsigned int len = w * h * l;
    std::ofstream fo1(buffer, std::ios::binary);
    fo1.write((char*)data, len * sizeof(float));
    fo1.close();
}
