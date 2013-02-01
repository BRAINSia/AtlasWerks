#include "VectorField3D.hxx"

// Compute gradient of u with cyclic boundary
// Assumes gradU is already allocated
extern void grad(const Image3D & u, VectorField3D & gradU);

// Pull back of an image by a vector field
extern void pullBack(const Image3D & im, const VectorField3D & v,
                     Image3D & result);

// Pull back of an image by a vector field
extern void pullBackClamp(const Image3D & im, const VectorField3D & v,
                          Image3D & result);

// Pull back of an image and a vector field by a vector field
extern void pullBack(const Image3D & im, const VectorField3D & v,
                     Image3D & resultI, const VectorField3D & u,
                     VectorField3D & resultU);

// Push an image forward by a vector field
extern void pushForward(const Image3D & im, const VectorField3D & v,
                        Image3D & result);

// Compute translational component of a vector field
extern void translationalComponent(const VectorField3D & v,
                                   Real & xTrans, Real & yTrans, Real & zTrans);
