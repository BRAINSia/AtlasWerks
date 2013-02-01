/* ================================================================
 *
 * AtlasWerks Project
 *
 * Copyright (c) Sarang C. Joshi. All rights reserved.  See
 * Copyright.txt or for details.
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notice for more information.
 *
 * ================================================================ */

#include "JointlyVaryingVectorFields.h"
#include "AtlasWerksTypes.h"

#include "HField3DUtils.h"

JointlyVaryingVectorFields::VectorField()
  : _numFlows(0), _sizeX(0), _sizeY(0), _sizeZ(0)
{
}

inline
void
JointlyVaryingVectorFields::jointlyTrilerpPoint(Vector3D<double> &pt, Vector3D<double> &out, 
                                                std::vector<double> paramcoords, std::vector<int> vsteps, std::vector<double> weights) const
{
  // Interpolate all dimensions at once
  // Note this only does BACKGROUND_STRATEGY_PARTIAL_ZERO

  // a (much) faster version of the floor function
  int floorX = static_cast<int>(pt.x);
  int floorY = static_cast<int>(pt.y);
  int floorZ = static_cast<int>(pt.z);
  if (pt.x < 0 && pt.x != static_cast<int>(pt.x)) --floorX;
  if (pt.y < 0 && pt.y != static_cast<int>(pt.y)) --floorY;
  if (pt.z < 0 && pt.z != static_cast<int>(pt.z)) --floorZ;

  // this is not truly ceiling, but floor + 1, which is usually ceiling    
  int ceilX = floorX + 1;
  int ceilY = floorY + 1;
  int ceilZ = floorZ + 1;

  const double t = pt.x - floorX;
  const double u = pt.y - floorY;
  const double v = pt.z - floorZ;
  const double oneMinusT = 1.0 - t;
  const double oneMinusU = 1.0 - u;
  const double oneMinusV = 1.0 - v;

  //
  // ^
  // |  v3   v2       -z->        v4   v5
  // y           --next slice-->      
  // |  v0   v1                   v7   v6
  //
  //      -x->
  //   

  VectorFieldCoordType v0X=0, v0Y=0, v0Z=0;
  VectorFieldCoordType v1X=0, v1Y=0, v1Z=0;
  VectorFieldCoordType v2X=0, v2Y=0, v2Z=0;
  VectorFieldCoordType v3X=0, v3Y=0, v3Z=0;
  VectorFieldCoordType v4X=0, v4Y=0, v4Z=0;
  VectorFieldCoordType v5X=0, v5Y=0, v5Z=0;
  VectorFieldCoordType v6X=0, v6Y=0, v6Z=0;
  VectorFieldCoordType v7X=0, v7Y=0, v7Z=0;

  bool inside = true;

  if(backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
    // wrap
    if(floorX < 0){
      floorX = _sizeX + floorX;
      if(ceilX < 0) ceilX = _sizeX + ceilX;
    }
    if(ceilX >= _sizeX){
      ceilX = ceilX % _sizeX;
      if(floorX >= _sizeX) floorX = floorX % _sizeX;
    }
    if(floorY < 0){
      floorY = _sizeY + floorY;
      if(ceilY < 0) ceilY = _sizeY + ceilY;
    }
    if(ceilY >= _sizeY){
      ceilY = ceilY % _sizeY;
      if(floorY >= _sizeY) floorY = floorY % _sizeY;
    }
    if(floorZ < 0){
      floorZ = _sizeZ + floorZ;
      if(ceilZ < 0) ceilZ = _sizeZ + ceilZ;
    }
    if(ceilZ >= _sizeZ){
      ceilZ = ceilZ % _sizeZ;
      if(floorZ >= _sizeZ) floorZ = floorZ % _sizeZ;
    }
  }
else if(backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
    // clamp
    if(floorX < 0){
      floorX = 0;
      if(ceilX < 0) ceilX = 0;
    }
    if(ceilX >= _sizeX){
      ceilX = _sizeX-1;
      if(floorX >= _sizeX) floorX = _sizeX-1;
    }
    if(floorY < 0){
      floorY = 0;
      if(ceilY < 0) ceilY = 0;
    }
    if(ceilY >= _sizeY){
      ceilY = _sizeY-1;
      if(floorY >= _sizeY) floorY = _sizeY-1;
    }
    if(floorZ < 0){
      floorZ = 0;
      if(ceilZ < 0) ceilZ = 0;
    }
    if(ceilZ >= _sizeZ){
      ceilZ = _sizeZ-1;
      if(floorZ >= _sizeZ) floorZ = _sizeZ-1;
    }
  }else{
    inside = (floorX >= 0 && ceilX < _sizeX &&
	      floorY >= 0 && ceilY < _sizeY &&
	      floorZ >= 0 && ceilZ < _sizeZ);
  }

  if (inside)
    {
      //
      // coordinate is inside volume, fill in 
      // eight corners of cube
      //
    for (unsigned int k=0;k < _numFlows;++k)
      {
      // We'll sum using the weights given to us.
      // To determine which VField to use we're given a vector of ints
      // a negative value means use the reverse vector field instead
      // of fwd.
      // Note these are 1-indexed, because of confusion at 0

      h = (vsteps[k] > 0 ? (*this)[k].vfwd[vsteps[k]-1] : (*this)[k].v[vsteps[k]-1]);

      v0X += h(floorX, floorY, floorZ).x;
      v0Y += h(floorX, floorY, floorZ).y;
      v0Z += h(floorX, floorY, floorZ).z;

      v1X += h(ceilX, floorY, floorZ).x;
      v1Y += h(ceilX, floorY, floorZ).y;
      v1Z += h(ceilX, floorY, floorZ).z;

      v2X += h(ceilX, ceilY, floorZ).x;
      v2Y += h(ceilX, ceilY, floorZ).y;
      v2Z += h(ceilX, ceilY, floorZ).z;

      v3X += h(floorX, ceilY, floorZ).x;
      v3Y += h(floorX, ceilY, floorZ).y;
      v3Z += h(floorX, ceilY, floorZ).z;

      v4X += h(floorX, ceilY, ceilZ).x;
      v4Y += h(floorX, ceilY, ceilZ).y;
      v4Z += h(floorX, ceilY, ceilZ).z;

      v5X += h(ceilX, ceilY, ceilZ).x;
      v5Y += h(ceilX, ceilY, ceilZ).y;
      v5Z += h(ceilX, ceilY, ceilZ).z;

      v6X += h(ceilX, floorY, ceilZ).x;
      v6Y += h(ceilX, floorY, ceilZ).y;
      v6Z += h(ceilX, floorY, ceilZ).z;

      v7X += h(floorX, floorY, ceilZ).x;
      v7Y += h(floorX, floorY, ceilZ).y;
      v7Z += h(floorX, floorY, ceilZ).z;
      }
    }
  else{
      //
      // coordinate is not inside volume; initialize cube
      // corners to identity/zero then set any corners of cube that
      // fall on volume boundary
      //
  v0X = 0; v0Y = 0; v0Z = 0;
  v1X = 0; v1Y = 0; v1Z = 0;
  v2X = 0; v2Y = 0; v2Z = 0;
  v3X = 0; v3Y = 0; v3Z = 0;
  v4X = 0; v4Y = 0; v4Z = 0;
  v5X = 0; v5Y = 0; v5Z = 0;
  v6X = 0; v6Y = 0; v6Z = 0;
  v7X = 0; v7Y = 0; v7Z = 0;

      bool floorXIn = floorX >= 0 && floorX < _sizeX;
      bool floorYIn = floorY >= 0 && floorY < _sizeY;
      bool floorZIn = floorZ >= 0 && floorZ < _sizeZ;
	
      bool ceilXIn = ceilX >= 0 && ceilX < _sizeX;
      bool ceilYIn = ceilY >= 0 && ceilY < _sizeY;
      bool ceilZIn = ceilZ >= 0 && ceilZ < _sizeZ;
	
      if (floorXIn && floorYIn && floorZIn)
	{
        for (unsigned int k=0;k < _numFlows;++k)
          {
	  v0X += h(floorX, floorY, floorZ).x;
	  v0Y += h(floorX, floorY, floorZ).y;
	  v0Z += h(floorX, floorY, floorZ).z;	  
          }
	}
      if (ceilXIn && floorYIn && floorZIn)
	{
	  v1X = h(ceilX, floorY, floorZ).x;
	  v1Y = h(ceilX, floorY, floorZ).y;
	  v1Z = h(ceilX, floorY, floorZ).z;
	}
      if (ceilXIn && ceilYIn && floorZIn)
	{
	  v2X = h(ceilX, ceilY, floorZ).x;
	  v2Y = h(ceilX, ceilY, floorZ).y;
	  v2Z = h(ceilX, ceilY, floorZ).z;
	}
      if (floorXIn && ceilYIn && floorZIn)
	{
	  v3X = h(floorX, ceilY, floorZ).x;
	  v3Y = h(floorX, ceilY, floorZ).y;
	  v3Z = h(floorX, ceilY, floorZ).z;
	}
      if (floorXIn && ceilYIn && ceilZIn)
	{
	  v4X = h(floorX, ceilY, ceilZ).x;
	  v4Y = h(floorX, ceilY, ceilZ).y;
	  v4Z = h(floorX, ceilY, ceilZ).z;	  
	}
      if (ceilXIn && ceilYIn && ceilZIn)
	{
	  v5X = h(ceilX, ceilY, ceilZ).x;
	  v5Y = h(ceilX, ceilY, ceilZ).y;
	  v5Z = h(ceilX, ceilY, ceilZ).z;	  
	}
      if (ceilXIn && floorYIn && ceilZIn)
	{
	  v6X = h(ceilX, floorY, ceilZ).x;
	  v6Y = h(ceilX, floorY, ceilZ).y;
	  v6Z = h(ceilX, floorY, ceilZ).z;	  
	}
      if (floorXIn && floorYIn && ceilZIn)
	{
	  v7X = h(floorX, floorY, ceilZ).x;
	  v7Y = h(floorX, floorY, ceilZ).y;
	  v7Z = h(floorX, floorY, ceilZ).z;	  
	}
    }
    
  //
  // do trilinear interpolation
  //
    
  //
  // this is the basic trilerp function...
  //
  //     h = 
  //       v0 * (1 - t) * (1 - u) * (1 - v) +
  //       v1 * t       * (1 - u) * (1 - v) +
  //       v2 * t       * u       * (1 - v) +
  //       v3 * (1 - t) * u       * (1 - v) +
  //       v4 * (1 - t) * u       * v       +
  //       v5 * t       * u       * v       +
  //       v6 * t       * (1 - u) * v       +
  //       v7 * (1 - t) * (1 - u) * v;
  //
  // the following nested version saves 30 multiplies.
  //
    
  hx = 
    oneMinusT * (oneMinusU * (v0X * oneMinusV + v7X * v)  +
		 u         * (v3X * oneMinusV + v4X * v)) +
    t         * (oneMinusU * (v1X * oneMinusV + v6X * v)  +
		 u         * (v2X * oneMinusV + v5X * v));
    
  hy = 
    oneMinusT * (oneMinusU * (v0Y * oneMinusV + v7Y * v)  +
		 u         * (v3Y * oneMinusV + v4Y * v)) +
    t         * (oneMinusU * (v1Y * oneMinusV + v6Y * v)  +
		 u         * (v2Y * oneMinusV + v5Y * v));
    
  hz = 
    oneMinusT * (oneMinusU * (v0Z * oneMinusV + v7Z * v)  +
		 u         * (v3Z * oneMinusV + v4Z * v)) +
    t         * (oneMinusU * (v1Z * oneMinusV + v6Z * v)  +
		 u         * (v2Z * oneMinusV + v5Z * v));
}

Vector3D<double> jointlyFlowPoint(Vector3D<double> spatialcoords, 
                                  std::vector<double> paramcoords, unsigned int numsteps,
                                  double endradius = 0.0) const
{
  // Flow from (s,t) to endradius*(s,t)/sqrt(s^2+t^2)

  std::vector<double> weights;
}

void JointlyVaryingVectorFields::jointlyFlow(Vector3D<double> spatialcoords, std::vector<double> paramcoords,
                   VectorField &h, unsigned int numsteps, double endradius = 0.0) const
{
}
