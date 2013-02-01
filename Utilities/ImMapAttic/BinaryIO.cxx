#include "BinaryIO.h"

bool BinaryIO::checkClassAssertions() {
  return (sizeof(char) == 1 &&
	  sizeof(short) == 2 &&
	  sizeof(int) == 4 &&
	  sizeof(float) == 4 &&
	  sizeof(double) == 8);
}

BinaryIO::Endian BinaryIO::computeHostEndian() {
  int x = 1;
  if (*(char*) &x == 1) {
    return little_endian;
  }
  else {
    return big_endian;
  }
}

BinaryIO::BinaryIO() {
  setIOEndian(big_endian);
  setHostEndian();
  _num_bytes_written = 0;
  _num_bytes_read = 0;

  if (!checkClassAssertions()) {
    std::cerr << "BinaryIO: WARNING: integral type size assertions do not hold!" << std::endl;
  }
}

int BinaryIO::getNumBytesRead() const {
  return _num_bytes_read;
}

int BinaryIO::getNumBytesWritten() const {
  return _num_bytes_written;
}

void BinaryIO::setIOEndian(const Endian& e) {
  _io_endian = e;
} 

void BinaryIO::setHostEndian(const Endian& e) {
  _host_endian = e;
} 

void BinaryIO::setHostEndian() {
  _host_endian = computeHostEndian();
}

BinaryIO::Endian BinaryIO::getIOEndian() const {
  return _io_endian;
} 

BinaryIO::Endian BinaryIO::getHostEndian() const {
  return _host_endian;
} 

void BinaryIO::writeDouble(const double& d, std::ostream& out) {
  if (_io_endian != _host_endian) {
    double tmp = swabDouble(d);
    out.write((char*)&tmp, sizeof(tmp));
  }
  else {
    out.write((char*)&d, sizeof(d));    
  }
  _num_bytes_written += sizeof(d);
}

void BinaryIO::writeFloat(const float& f, std::ostream& out) {
  if (_io_endian != _host_endian) {
    float tmp = swabFloat(f);
    out.write((char*)&tmp, sizeof(tmp));
  }
  else {
    out.write((char*)&f, sizeof(f));    
  }
  _num_bytes_written += sizeof(f);
}

void BinaryIO::writeInt(const int i, std::ostream& out) {
  if (_io_endian != _host_endian) {
    int tmp = swabInt(i);
    out.write((char*)&tmp, sizeof(tmp));
  }
  else {
    out.write((char*)&i, sizeof(i));    
  }
  _num_bytes_written += sizeof(i);
}

void BinaryIO::writeShort(const short s, std::ostream& out) {
  if (_io_endian != _host_endian) {
    short tmp = swabShort(s);
    out.write((char*)&tmp, sizeof(tmp));
  }
  else {
    out.write((char*)&s, sizeof(s));    
  }
  _num_bytes_written += sizeof(s);
}

void BinaryIO::writeChar(const char c, std::ostream& out) {
  out.write(&c, sizeof(c));    
  _num_bytes_written += sizeof(c);
}

void BinaryIO::write(const char* const buff, const int length, std::ostream& out) {
  out.write(buff, length);
  _num_bytes_written += length;
}

double BinaryIO::readDouble(std::istream& in) {
  double tmp;
  in.read((char*)&tmp, sizeof(tmp));
  _num_bytes_read += sizeof(tmp);
  if (_io_endian != _host_endian) {
    return swabDouble(tmp);
  }
  else {
    return tmp;
  }
}

float BinaryIO::readFloat(std::istream& in) {
  float tmp;
  in.read((char*)&tmp, sizeof(tmp));
  _num_bytes_read += sizeof(tmp);
  if (_io_endian != _host_endian) {
    return swabFloat(tmp);
  }
  else {
    return tmp;
  }
}
 
int BinaryIO::readInt(std::istream& in) {
  int tmp;
  in.read((char*)&tmp, sizeof(tmp));
  _num_bytes_read += sizeof(tmp);
  if (_io_endian != _host_endian) {
    return swabInt(tmp);
  }
  else {
    return tmp;
  }
}

short BinaryIO::readShort(std::istream& in) {
  short tmp;
  in.read((char*)&tmp, sizeof(tmp));
  _num_bytes_read += sizeof(tmp);
  if (_io_endian != _host_endian) {
    return swabShort(tmp);
  }
  else {
    return tmp;
  }
}

char BinaryIO::readChar(std::istream& in) {
  char tmp;
  in.read((char*)&tmp, sizeof(tmp));
  _num_bytes_read += sizeof(tmp);
  return tmp;
}

void BinaryIO::read(char* buff, const int length, std::istream& in) {
  _num_bytes_read += length;
  in.read((char*)buff, length);
}

//
// byte swapping routines
//

// changes byte order for short
// from big-endain to little-endian
// or vice versa
//
// byte numbers...
// 1 2 -> 2 1
short BinaryIO::swabShort(const short& in) {
  short out;
  char* p_in = (char*) &in;
  char* p_out = (char*) &out;
  p_out[0] = p_in[1];
  p_out[1] = p_in[0];
  return out;
}

unsigned short BinaryIO::swabUShort(const unsigned short& in) {
  unsigned short out;
  unsigned int temp = in; 
  char* p_in = (char*) &temp;
  char* p_out = (char*) &out;
  p_out[0] = p_in[1];
  p_out[1] = p_in[0];
  return out;
}

// changes byte order for int
// from big-endian to little-endian
// or vice versa
//
// byte numbers...
// 1 2 3 4 -> 4 3 2 1
int BinaryIO::swabInt(const int& in) {
  int out;
  char* p_in = (char*) &in;
  char* p_out = (char*) &out;
  p_out[0] = p_in[3];
  p_out[1] = p_in[2];
  p_out[2] = p_in[1];
  p_out[3] = p_in[0];
  return out;
}

// changes byte order for float
// from big-endian to little-endian
// or vice versa
//
// byte numbers...
// 1 2 3 4 -> 4 3 2 1
float BinaryIO::swabFloat(const float& in) {
  float out;
  char* p_in = (char*) &in;
  char* p_out = (char*) &out;
  p_out[0] = p_in[3];
  p_out[1] = p_in[2];
  p_out[2] = p_in[1];
  p_out[3] = p_in[0];
  return out;
}

// changes byte order for double
// from big-endian to little-endian
// or vice versa
//
// byte numbers...
// 1 2 3 4 5 6 7 8 -> 8 7 6 5 4 3 2 1
double BinaryIO::swabDouble(const double& in) {
  double out;
  char* p_in = (char*) &in;
  char* p_out = (char*) &out;
  p_out[0] = p_in[7];
  p_out[1] = p_in[6];
  p_out[2] = p_in[5];
  p_out[3] = p_in[4];
  p_out[4] = p_in[3];
  p_out[5] = p_in[2];
  p_out[6] = p_in[1];
  p_out[7] = p_in[0];
  return out;  
}
