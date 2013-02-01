
%module DataTypes

%{
#include "Vector3D.h"
#include "Matrix3D.h"
#include "AffineTransform3D.h"
#include "Array3D.h"
#include "Image.h"

//#include "Vector2D.h"
//#include "Array2D.h"
//#include "Image2D.h"

%}

%feature("autodoc","1");

%define CloneFunc(type)
%extend type {
  void clone(const type &other) {
    (*$self) = other;
  }
};
%enddef

// ################################################################
// Vector3D
// ################################################################

%include "Vector3D.h"

%define Vector3DPrintExtension(format, type)
%extend Vector3D<type> {
  char *__str__() {
    static char tmp[1024];
    sprintf(tmp,"Vector(format,format,format)", $self->x,$self->y,$self->z);
    return tmp;
  }
};
%enddef

%template(dVector3D) Vector3D<double>;
Vector3DPrintExtension(%g,double)
%template(fVector3D) Vector3D<float>;
Vector3DPrintExtension(%g,float)
%template(iVector3D) Vector3D<int>;
Vector3DPrintExtension(%d,int)
%template(uiVector3D) Vector3D<unsigned int>;
Vector3DPrintExtension(%d,unsigned int)

CloneFunc(Vector3D<double>)
CloneFunc(Vector3D<float>)
CloneFunc(Vector3D<int>)
CloneFunc(Vector3D<unsigned int>)

%define Vector3DToList(type)
%extend Vector3D< type > {
%pythoncode %{
def tolist(self):
  """Convert a Vector3D to a multidimensional list"""
  vec_list = [self.x, self.y, self.z];
  return vec_list
%}
};
%enddef

Vector3DToList(double)
Vector3DToList(float)
Vector3DToList(int)
Vector3DToList(unsigned int)

%define Vector3DFromList(type)
%extend Vector3D< type > {
%pythoncode %{
def fromlist(self, lst):
  """Convert a list to a Vector3D"""
  self.x = lst[0];
  self.y = lst[1];
  self.z = lst[2];
%}
};
%enddef

Vector3DFromList(double)
Vector3DFromList(float)
Vector3DFromList(int)
Vector3DFromList(unsigned int)

// ################################################################
// Vector2D
// ################################################################

%include "Vector2D.h"

%define Vector2DPrintExtension(format, type)
%extend Vector2D<type> {
  char *__str__() {
    static char tmp[1024];
    sprintf(tmp,"Vector(format,format)", $self->x,$self->y);
    return tmp;
  }
};
%enddef

%template(dVector2D) Vector2D<double>;
Vector2DPrintExtension(%g,double)
%template(fVector2D) Vector2D<float>;
Vector2DPrintExtension(%g,float)
%template(iVector2D) Vector2D<int>;
Vector2DPrintExtension(%d,int)
%template(uiVector2D) Vector2D<unsigned int>;
Vector2DPrintExtension(%d,unsigned int)

CloneFunc(Vector2D<double>)
CloneFunc(Vector2D<float>)
CloneFunc(Vector2D<int>)
CloneFunc(Vector2D<unsigned int>)

%define Vector2DToList(type)
%extend Vector2D< type > {
%pythoncode %{
def tolist(self):
  """Convert a Vector2D to a multidimensional list"""
  vec_list = [self.x, self.y];
  return vec_list
%}
};
%enddef

Vector2DToList(double)
Vector2DToList(float)
Vector2DToList(int)
Vector2DToList(unsigned int)

%define Vector2DFromList(type)
%extend Vector2D< type > {
%pythoncode %{
def fromlist(self, lst):
  """Convert a list to a Vector2D"""
  self.x = lst[0];
  self.y = lst[1];
%}
};
%enddef

Vector2DFromList(double)
Vector2DFromList(float)
Vector2DFromList(int)
Vector2DFromList(unsigned int)

// ################################################################
//  Matrix3D
// ################################################################

%include "Matrix3D.h"

%template(fMatrix3D) Matrix3D<float>;

%extend Matrix3D<float>{
  char *__str__() {
    static char tmp[1024];
    sprintf(tmp, "[ %f %f %f ]\n[ %f %f %f ]\n[ %f %f %f ]\n", 
	    $self->a[0], $self->a[1], $self->a[2], 
	    $self->a[3], $self->a[4], $self->a[5], 
	    $self->a[6], $self->a[7], $self->a[8]); 
    return tmp;
  }
};

%template(dMatrix3D) Matrix3D<double>;

CloneFunc(Matrix3D<double>)

%extend Matrix3D< double > {
  char *__str__() {
    static char tmp[1024];
    sprintf(tmp, "[ %lf %lf %lf ]\n[ %lf %lf %lf ]\n[ %lf %lf %lf ]\n", 
	    $self->a[0], $self->a[1], $self->a[2], 
	    $self->a[3], $self->a[4], $self->a[5], 
	    $self->a[6], $self->a[7], $self->a[8]); 
    return tmp;
  }

};

%define Matrix3DToList(type)
%extend Matrix3D< type > {
%pythoncode %{
def tolist(self):
  """Convert a Vector3D to a multidimensional list"""
  mat_list = [[0, 0, 0],[0, 0, 0],[0, 0, 0]];
  for m in range(3):
    for n in range(3):
      mat_list[m][n] = self.get(m,n)
  return mat_list
%}
};
%enddef

Matrix3DToList(float)
Matrix3DToList(double)

%define Matrix3DFromList(type)
%extend Matrix3D< type > {
%pythoncode %{
def fromlist(self, lst):
  """Convert a list to a Matrix3D"""
  for m in range(3):
    for n in range(3):
      self.set(m,n,lst[m][n])
%}
};
%enddef

Matrix3DFromList(float)
Matrix3DFromList(double)

// ################################################################
//  Affine3D
// ################################################################

%include "AffineTransform3D.h"

%template(fAffine3D) AffineTransform3D<float>;
%extend AffineTransform3D< float > {
  char *__str__() {
    static char tmp[1024];
    sprintf(tmp, "Matrix\n[ %f %f %f ]\n[ %f %f %f ]\n[ %f %f %f ]\nVector\n[ %f %f %f ]\n", 
	    $self->matrix.a[0], $self->matrix.a[1], $self->matrix.a[2], 
	    $self->matrix.a[3], $self->matrix.a[4], $self->matrix.a[5], 
	    $self->matrix.a[6], $self->matrix.a[7], $self->matrix.a[8],
	    $self->vector.x, $self->vector.y, $self->vector.z); 
    return tmp;
  }
  
};

%template(dAffine3D) AffineTransform3D<double>;

CloneFunc(AffineTransform3D<double>)

%extend AffineTransform3D< double > {
  char *__str__() {
    static char tmp[1024];
    sprintf(tmp, "Matrix\n[ %lf %lf %lf ]\n[ %lf %lf %lf ]\n[ %lf %lf %lf ]\nVector\n[ %lf %lf %lf ]\n", 
	    $self->matrix.a[0], $self->matrix.a[1], $self->matrix.a[2], 
	    $self->matrix.a[3], $self->matrix.a[4], $self->matrix.a[5], 
	    $self->matrix.a[6], $self->matrix.a[7], $self->matrix.a[8],
	    $self->vector.x, $self->vector.y, $self->vector.z); 
    return tmp;
  }

};

%define AffineTransform3DToList(type)
%extend AffineTransform3D< type > {
%pythoncode %{
def tolist(self):
  """Convert a Affine3D to a 4x4 array"""
  a = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  for m in range(3):
    for n in range(3):
      a[m][n] = self.matrix.get(m,n)
  for m in range(3):
     a[m][3] = self.vector.get(m)
  return a
%}
};
%enddef

AffineTransform3DToList(float)
AffineTransform3DToList(double)

%define AffineTransform3DFromList(type)
%extend AffineTransform3D< type > {
%pythoncode %{
def fromlist(self, lst):
  """Convert a list to a Matrix3D"""
  for m in range(3):
    for n in range(3):
      self.matrix.set(m,n,lst[m][n])

  for m in range(3):
    self.vector.set(m,lst[m][3])
%}
};
%enddef

AffineTransform3DFromList(float)
AffineTransform3DFromList(double)

// ################################################################
// Array3D
// ################################################################

%include "Array3D.h"

%define Array3DPrintExtension(type)
%extend Array3D< type > {
  char *__str__() {
    static char tmp[1024];
    Vector3D<unsigned int> size = $self->getSize();
    sprintf(tmp, "Array3D< type > size (%d, %d, %d)", size.x, size.y, size.z);
    return tmp;
  }
};
%enddef

%exception get{
  try {
    $action
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_IndexError, const_cast<char*>(e.what()));
    return NULL;
  }
}

%template(fArray3D) Array3D<float>;

CloneFunc(Array3D<float>)

Array3DPrintExtension(float)

%extend Array3D< float > {
  float get(unsigned int xIndex,
	    unsigned int yIndex,
	    unsigned int zIndex) const
  {
    float val = $self->get(xIndex, yIndex, zIndex);
    return val;
  }

  float get(unsigned int elementIndex) const
  {
    float val = $self->get(elementIndex);
    return val;
  }
};

// generalize classname so it can be Array3D or Image
%define Array3DToList(classname, type)
%extend classname< type > {
%pythoncode %{
def tolist(self):
  """Convert a RealArray3D to a multidimensional list"""
  imsize = self.getSize()
  # set up multidimensional array
  im_as_list = [0]*imsize.x
  for xi in range(imsize.x):
    im_as_list[xi] = [0]*imsize.y
    for yi in range(imsize.y):
      im_as_list[xi][yi] = [0]*imsize.z
  # fill array
  for zi in range(imsize.z):
    for yi in range(imsize.y):
      for xi in range(imsize.x):
        im_as_list[xi][yi][zi] = self.get(xi,yi,zi)
  return im_as_list
%}
};
%enddef

Array3DToList(Array3D,float)

%define Array3DFromList(classname, type)
%extend classname< type > {
%pythoncode %{
def fromlist(self, lst):
  """Convert a multidimensional list to a RealArray3D"""
  imsize = uiVector3D(len(lst),len(lst[0]),len(lst[0][0]))
  self.resize(imsize)
  for zi in range(imsize.z):
    for yi in range(imsize.y):
      for xi in range(imsize.x):
        self.set(xi,yi,zi,lst[xi][yi][zi])
%}
};
%enddef

Array3DFromList(Array3D, float)

%template(fVectorField) Array3D<Vector3D<float> >;

CloneFunc(Array3D<Vector3D<float> >)

Array3DPrintExtension(Vector3D<float>)

%extend Array3D< Vector3D< float > > {
  Vector3D< float > 
    get(unsigned int xIndex,
	unsigned int yIndex,
	unsigned int zIndex) const
  {
    Vector3D< float > v = $self->get(xIndex, yIndex, zIndex);
    return v;
  }
};

%define VectorFieldToList(type)
%extend Array3D< Vector3D< type > > {
%pythoncode %{
def tolist(self):
  """Convert a VectorField to a multidimensional list"""
  sz = self.getSize()
  # set up multidimensional array
  vf_as_list = [0]*sz.x
  for xi in range(sz.x):
    vf_as_list[xi] = [0]*sz.y
    for yi in range(sz.y):
      vf_as_list[xi][yi] = [0]*sz.z
      for zi in range(sz.z):
        vf_as_list[xi][yi][zi] = [0]*3
  # fill array
  for zi in range(sz.z):
    for yi in range(sz.y):
      for xi in range(sz.x):
        v = self(xi,yi,zi)
        vf_as_list[xi][yi][zi][0] = v.x
        vf_as_list[xi][yi][zi][1] = v.y
        vf_as_list[xi][yi][zi][2] = v.z
  return vf_as_list
%}
};
%enddef

VectorFieldToList(float)

%define VectorFieldFromList(type)
%extend Array3D< Vector3D< type > > {
%pythoncode %{
def fromlist(self, l):
  """Convert a multidimensional list to a Array3D"""
  sz = uiVector3D(len(l),len(l[0]),len(l[0][0]))
  self.resize(sz)
  for zi in range(sz.z):
    for yi in range(sz.y):
      for xi in range(sz.x):
        v = fVector3D()
        v.x = l[xi][yi][zi][0]
        v.y = l[xi][yi][zi][1]
        v.z = l[xi][yi][zi][2]
        self.set(xi,yi,zi,v)
%}
};
%enddef

VectorFieldFromList(float)

// ################################################################
// Array2D
// ################################################################

%include "Array2D.h"

%define Array2DPrintExtension(type)
%extend Array2D< type > {
  char *__str__() {
    static char tmp[1024];
    Vector2D<unsigned int> size = $self->getSize();
    sprintf(tmp, "Array2D< type > size (%d, %d)", size.x, size.y);
    return tmp;
  }
};
%enddef

%template(fArray2D) Array2D<float>;

CloneFunc(Array2D<float>)

Array2DPrintExtension(float)

%define VectorField2DToList(type)
%extend Array2D< Vector2D< type > > {
%pythoncode %{
def tolist(self):
  """Convert a VectorField2D to a multidimensional list"""
  sz = self.getSize()
  # set up multidimensional array
  vf_as_list = [0]*sz.x
  for xi in range(sz.x):
    vf_as_list[xi] = [0]*sz.y
    for yi in range(sz.y):
      vf_as_list[xi][yi] = [0]*2
  # fill array
  for yi in range(sz.y):
    for xi in range(sz.x):
      v = self(xi,yi)
      vf_as_list[xi][yi][0] = v.x
      vf_as_list[xi][yi][1] = v.y
  return vf_as_list
%}
};
%enddef

%define VectorField2DFromList(type)
%extend Array2D< Vector2D< type > > {
%pythoncode %{
def fromlist(self, l):
  """Convert a multidimensional list to a Array2D"""
  sz = uiVector2D(len(l),len(l[0]),len(l[0][0]))
  self.resize(sz)
  for yi in range(sz.y):
    for xi in range(sz.x):
      v = fVector2D()
      v.x = l[xi][yi][0]
      v.y = l[xi][yi][1]
      self.set(xi,yi,v)
%}
};
%enddef


// fVectorFieldSlice is a slice of a 3D vector field, it's not a 2D
// vector field, so let's add a little bit of code to turn it into one
// Note it makes no sense to have this for anything but vector fields
%define VectorFieldSliceFlattenExtension(type)
%extend Array2D<Vector3D< type > > {
%pythoncode %{
  def flatten(self, slicedir):
    """Flatten a vector field slice into a 2D vector field"""
    sz = self.getSize()
    vf = fVectorField2D(sz)
    for yi in range(sz.y):
      for xi in range(sz.x):
        vx = self.get(xi,yi)
        if slicedir == 0:
          vf.set(xi,yi,fVector2D(vx.y,vx.z))
        elif slicedir == 1:
          vf.set(xi,yi,fVector2D(vx.x,vx.z))
        elif slicedir == 2:
          vf.set(xi,yi,fVector2D(vx.x,vx.y))
    return vf
%}
};
%enddef

%define VectorField2DComponentExtension(type)
%extend Array2D<Vector2D<type> > {
  %pythoncode %{
  def getComponent(self,comp):
    """Take a vector field and return an fArray2D of a single component"""
    sz = self.getSize()
    vf = fArray2D(sz)
    for yi in range(sz.y):
      for xi in range(sz.x):
        vf.set(xi,yi, self.get(xi,yi).tolist[comp]);
    return vf
  %}
 };
%enddef

%template(fVectorField2D) Array2D<Vector2D<float> >;
CloneFunc(Array2D<Vector2D<float> >)
Array2DPrintExtension(Vector2D<float>)
VectorField2DToList(float)
VectorField2DFromList(float)
VectorField2DComponentExtension(float)
%template(fVectorFieldSlice) Array2D<Vector3D< float > >;
CloneFunc(Array2D<Vector3D<float> >)
VectorFieldSliceFlattenExtension(float)
Array2DPrintExtension(Vector3D<float>)

// ################################################################
//  Image
// ################################################################

%include "Image.h"

%template(fImage) Image<float>;
CloneFunc(Image<float>)

%extend Image< float > {
  char *__str__() {
    static char tmp[1024];
    Vector3D<unsigned int> size = $self->getSize();
    Vector3D<double> origin = $self->getOrigin();
    Vector3D<double> spacing = $self->getSpacing();
    sprintf(tmp, "Image< float > size (%d, %d, %d) origin (%g %g %g) spacing (%g %g %g)", 
	    size.x, size.y, size.z, 
	    origin.x, origin.y, origin.z,
	    spacing.x, spacing.y, spacing.z);
    return tmp;
  }
};

Array3DToList(Image, float)
Array3DFromList(Image, float)


//%include "Vector2D.h"
//%include "Array2D.h"
//%include "Image2D.h"
