#include "nddrUtils.h"
#include <cudaInterface.h>
#include <fstream>

void readRawVolumeData(char* rawFile, float* data, int len){
    fprintf(stderr, "Reading file %s ", rawFile);
    FILE* pFile = fopen(rawFile, "rb");
    
    unsigned int result = fread(data, sizeof(float), len , pFile);

    if (result != len)
        fprintf(stderr, "Reading error ");
    fclose(pFile);
    fprintf(stderr, ".. done \n");
}

void writeToNrrd(float* data, int w, int h, int l, const char *name) {
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

void writeDeviceToNrrd(float* d_data, int w, int h, int l, const char *name) {
    float* h_data = new float [w * h * l];
    copyArrayFromDevice(h_data, d_data, w * h * l);
    writeToNrrd(h_data, w, h, l, name);
    delete []h_data;
}

void writeDeviceToNrrd(float* d_data, const Vector3Di& size, const char *name) {
    writeDeviceToNrrd(d_data, size.x, size.y, size.z, name);
}
