#ifndef BinaryIO_h
#define BinaryIO_h

#include <iostream>

/*
 * This class is used to hide bite swapping for binary
 * file i/o.  For instance, pcs are little-endian and (some? most? all?) 
 * suns are big-endian.  Thus, to read and write the same binary data
 * files, an endian-ness must be decided for the data file and one
 * platform will have to swap bytes (i.e. change endian-ness) before
 * writing or after reading integral data types to/from the binary file.
 *
 * See below class declaration for specific byte swapping routines.
 *
 * When an instance of the class is created, it will automatically
 * decide wheather the machine is little or big endian (although you
 * can override this).
 * The default i/o endianness is big - you can change
 * this also.
 *
 * Typical use...
 *
 * ofstream output("myfile", ios::binary);
 * BinaryIO bio; 
 * bio.setIOEndian(BinaryIO::little_endian); 
 * bio.writeDouble(d, output);
 * bio.writeInt(i, output);
 * bio.write(char_array, length, output);
 * output.close();
 *
 * each instance keeps track of how many bytes have
 * been read or written, I use this mainly for
 * debugging (e.g. what is the byte address of this
 * variable in the file)
 *
 *
 * This class assumes that...
 * char is 1 byte (sizeof(char) == 1)
 * short is 2 bytes (sizeof(short) == 2)
 * int is 4 bytes (sizeof(int) == 4)
 * float is 4 bytes (sizeof(float) == 4)
 * double is 8 bytes (sizeof(double) == 8)
 *
 * This class will not work if these do not
 * hold.  Right now, there is a check made in
 * the constructor and a warning is printed
 * if these asertions do not hold.  
 *
 * bcd 31Jan03
 *
 */

class BinaryIO {
 public:
  enum Endian {big_endian, little_endian};

  BinaryIO();

  // routines to check host configuration
  static Endian computeHostEndian();
  static bool checkClassAssertions();

  // get info about or configure instance
  int getNumBytesRead() const;
  int getNumBytesWritten() const;
  void setIOEndian(const Endian& e);
  void setHostEndian(const Endian& e);
  void setHostEndian();
  Endian getIOEndian() const;
  Endian getHostEndian() const;

  // byte swapping routines
  static short swabShort(const short& in);
  static unsigned short swabUShort(const unsigned short& in);
  static int swabInt(const int& in);
  static float swabFloat(const float& in);
  static double swabDouble(const double& in);

  // write a _blank_ to the ostream, byte swapping as necessary
  void writeDouble(const double& d, std::ostream& out);
  void writeFloat(const float& f, std::ostream& out);
  void writeInt(const int i, std::ostream& out);
  void writeShort(const short s, std::ostream& out);
  void writeChar(const char c, std::ostream& out);
  void write(const char* const buff, const int length, std::ostream& out);

  // read a _blank_ from the istream, byte swapping as necessary
  double readDouble(std::istream& in);
  float readFloat(std::istream& in);
  int readInt(std::istream& in);
  short readShort(std::istream& in);
  char readChar(std::istream& in);
  void read(char* buff, const int length, std::istream& in);

 private:
  Endian _io_endian;     // what to write and read as
  Endian _host_endian;   // endianness of the machine 
  int _num_bytes_written;
  int _num_bytes_read;
};

#endif
