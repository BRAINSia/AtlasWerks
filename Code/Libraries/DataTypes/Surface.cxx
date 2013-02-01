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

#include "Surface.h"
#include "Vector3D.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <numeric>
#include <cassert>

Surface
::Surface() 
{}

Surface
::Surface(const Surface& rhs)
  : vertices(rhs.vertices),
    normals(rhs.normals),
    unitNormals(rhs.unitNormals),
    facets(rhs.facets)
{
}

Surface& 
Surface
::operator=(const Surface& rhs) 
{
  if (&rhs != this)
    {
      vertices.assign(rhs.vertices.begin(), rhs.vertices.end());
      facets.assign(rhs.facets.begin(), rhs.facets.end());
      normals.assign(rhs.normals.begin(), rhs.normals.end());
      unitNormals.assign(rhs.unitNormals.begin(), rhs.unitNormals.end());
    }      
  return *this;
}

void
Surface
::clear()
{
  vertices.clear();
  normals.clear();
  unitNormals.clear();
  facets.clear();
}

void 
Surface
::getAxisExtrema(Vector3D<double>& min, 
		 Vector3D<double>& max) const 
{
  assert(!vertices.empty());

  // initialize extrema to first vertex
  min = max = vertices[0];

  // loop through the remaining vertices looking for extrema
  for (ConstVertexIter vertIter = vertices.begin(); 
       vertIter != vertices.end(); 
       ++vertIter) 
    {
      if (vertIter->x < min.x) min.x = vertIter->x;
      if (vertIter->y < min.y) min.y = vertIter->y;
      if (vertIter->z < min.z) min.z = vertIter->z;

      if (vertIter->x > max.x) max.x = vertIter->x;
      if (vertIter->y > max.y) max.y = vertIter->y;
      if (vertIter->z > max.z) max.z = vertIter->z;
    }
}

//
// !! this does not return the true centroid!!!
//
Vector3D<double>
Surface
::getCentroid() const 
{
  assert(!vertices.empty());
  return std::accumulate(vertices.begin(), 
			 vertices.end(), 
			 Vertex(0, 0, 0))
    / vertices.size();
}

void 
Surface
::scale(const Vector3D<double>& scaleFactor)
{
  for (VertexIter vertIter = vertices.begin();
       vertIter != vertices.end();
       ++vertIter)
    {
      *vertIter *= scaleFactor;
    }
}

void 
Surface
::scale(const double& sx, 
	const double& sy, 
	const double& sz) 
{
  for (VertexIter vertIter = vertices.begin(); 
       vertIter != vertices.end(); 
       ++vertIter)
    {
      vertIter->scale(sx, sy, sz);
    }		    
}  

void 
Surface
::applyAffineTransform(const AffineTransform3D<double>& transform)
{
  for (VertexIter vertIter = vertices.begin(); 
       vertIter != vertices.end(); 
       ++vertIter)
  {
    transform.transformVector(*vertIter);
  }		    
}

void 
Surface
::translate(const Vector3D<double>& t) 
{
  for (VertexIter vertIter = vertices.begin(); 
       vertIter != vertices.end(); 
       ++vertIter)
    {
      *vertIter += t;
    }		    
}

void 
Surface
::translate(const double& tx, 
	    const double& ty, 
	    const double& tz) 
{
  for (VertexIter vertIter = vertices.begin(); 
       vertIter != vertices.end(); 
       ++vertIter)
    {
      vertIter->translate(tx, ty, tz);
    }		    
}

void 
Surface
::computeNormals() {
  if (normals.size() != vertices.size()) 
    {
      normals.resize(vertices.size(), Normal(0, 0, 0));
    }

  if (facetNormals.size() != facets.size()) 
  {
      facetNormals.reserve(facets.size());
  }

  for (ConstFacetIter facetIter = facets.begin(); 
       facetIter != facets.end(); 
       ++facetIter) 
    {
      // get triangle edges
      Vector3D<double> edge01 = 
	vertices[(*facetIter)[1]]-vertices[(*facetIter)[0]];
      Vector3D<double> edge12 = 
	vertices[(*facetIter)[2]]-vertices[(*facetIter)[1]];
      Vector3D<double> edge20 = 
	vertices[(*facetIter)[0]]-vertices[(*facetIter)[2]];

      // get normal vector for facet
      // will be the same (except for sign) regardless of
      // what pair is taken
      Normal facetNormal = -edge01.cross(edge20);
      facetNormals.push_back(facetNormal);
      double facetNormalLength = facetNormal.length();
    
      // compute interior angle at each vertex (to use as weight)
      double angle0 = fabs(asin(facetNormalLength 
				/ (edge01.length()*edge20.length())));
      double angle1 = fabs(asin(facetNormalLength  
				/ (edge12.length()*edge01.length())));
      double angle2 = fabs(asin(facetNormalLength 
				/ (edge20.length()*edge12.length())));
    
      // increment normal at each vertex with this facets contribution
      facetNormal.normalize();
      normals[(*facetIter)[0]] += (facetNormal * angle0);
      normals[(*facetIter)[1]] += (facetNormal * angle1);
      normals[(*facetIter)[2]] += (facetNormal * angle2);
  }
}

void 
Surface
::computeUnitNormals() 
{
  computeNormals();
  unitNormals.assign(normals.begin(), normals.end());
  for (NormalIter iter = unitNormals.begin();
       iter != unitNormals.end();
       ++iter)
    {
      iter->normalize();
    }	 
}

void
Surface
::readBYU(std::istream &input) 
{
  if (input.fail())
    {
      throw std::runtime_error("invalid input stream");
    }

  //
  // read header info
  //
  int num_part;
  int num_vert;
  int num_poly;
  int num_connections;
  
  input >> num_part >> num_vert >> num_poly >> num_connections;
  
  if (input.fail() ||
      num_part < 0 ||
      num_vert < 0 ||
      num_poly < 0 ||
      num_connections < 0) 
    {
      throw std::runtime_error("error reading first line parameters.");
    }

  if (num_part != 1) 
    {
      throw std::runtime_error("more than one part.");
    }

  // read first and last polygon numbers
  // just skip over them for now
  int *first_poly = new int[num_part];
  int *last_poly = new int[num_part];
  int i=0;
  for (i = 0; i < num_part; i++) 
    {
      input >> first_poly[i] >> last_poly[i];
    }
  delete [] first_poly;
  delete [] last_poly;

  if (input.fail()) 
    {
      throw std::runtime_error("error reading part boundaries.");
    }  
  
  //
  // read vertices
  //
  vertices.resize(num_vert);
  for (VertexIter vertIter = vertices.begin();
       vertIter != vertices.end();
       ++vertIter)
    {
      input >> vertIter->x >> vertIter->y >> vertIter->z;
    }

  if (input.fail()) 
    {
      throw std::runtime_error("error reading vertices.");
    }

  //
  // read polygon lists
  //
  facets.resize(num_poly, Facet());
  for (FacetIter facetIter = facets.begin();
       facetIter != facets.end();
       ++facetIter)
    {
      int vertexIndex;
      while (input >> vertexIndex)
	{
	  if (vertexIndex > 0) 
	    {
	      facetIter->push_back(vertexIndex - 1);
	    }
	  else if (vertexIndex < 0) 
	    {
	      facetIter->push_back((-vertexIndex) - 1);
	      break;
	    }
	  else 
	    {
	      throw std::runtime_error("vertex index is 0.");
	    }
	}
    }
  if (input.fail()) 
    {
      throw std::runtime_error("error reading facets.");
    }
}

void
Surface
::readBYU(const std::string& filename)
{
  std::ifstream input(filename.c_str(), std::ios::binary);
  if (input.bad()) 
    {
      throw std::runtime_error("error opening file");
    }
  readBYU(input);
  input.close();
}

void
Surface
::readOFF(std::istream &input) 
{
  if (input.fail())
    {
      throw std::runtime_error("invalid input stream");
    }

  //
  // read header info
  //
  std::string keyword;
  int num_vert;
  int num_poly;
  int num_connections;
  
  input >> keyword >> num_vert >> num_poly >> num_connections;
  
  if (input.fail() ||
      num_vert < 0 ||
      num_poly < 0 ||
      num_connections < 0) 
    {
      throw std::runtime_error("error reading first line parameters.");
    }
  
  //
  // read vertices
  //
  vertices.resize(num_vert);
  for (VertexIter vertIter = vertices.begin();
       vertIter != vertices.end();
       ++vertIter)
    {
      input >> vertIter->x >> vertIter->y >> vertIter->z;
    }

  if (input.fail()) 
    {
      throw std::runtime_error("error reading vertices.");
    }

  //
  // read polygon lists
  //
  facets.resize(num_poly, Facet());
  for (FacetIter facetIter = facets.begin();
       facetIter != facets.end();
       ++facetIter)
    {
      int numVertsInFacet;
      input >> numVertsInFacet;
      facetIter->resize(numVertsInFacet, 0);
      for (Facet::iterator vertIter = facetIter->begin(); 
           vertIter != facetIter->end(); ++vertIter) {
        input >> (*vertIter);
      }
      if (facetIter->size() != 3) {
          int Vert0 = facetIter->front();
          facetIter->resize(3);
          (*facetIter)[0] = Vert0;
          (*facetIter)[1] = 0;
          (*facetIter)[2] = 1;
      }
      int dummy;
      for (int i = 0; i < 4; ++i) input >> dummy;
    }
  if (input.fail()) 
    {
      throw std::runtime_error("error reading facets.");
    }
}

void
Surface
::readOFF(const std::string& filename)
{
  std::ifstream input(filename.c_str(), std::ios::binary);
  if (input.bad()) 
    {
      throw std::runtime_error("error opening file");
    }
  readOFF(input);
  input.close();
}

void
Surface
::writeBYU(std::ostream &outFile) const 
{
  //
  // IMPORTANT NOTE: this only works for triangular surfaces
  //

  if (outFile.fail())
    {
      throw std::runtime_error("invalid input stream");
    }  

  // 
  // write numPart, numVert, numPoly, numConnections
  // 
  outFile << "1"               << " " 
	  << vertices.size()   << " " 
	  << facets.size()     << " "
	  << facets.size() * 3 << std::endl;
  if (outFile.fail())
    {
      throw std::runtime_error("error writing byu header");
    }  

  //
  // write part boundaries
  //
  outFile << "1 " << facets.size() << std::endl;
  if (outFile.fail())
    {
      throw std::runtime_error("error writing byu part boundaries");
    }  

  //
  // write the vertices
  //
  for (ConstVertexIter vertexIter = vertices.begin();
       vertexIter != vertices.end();
       ++vertexIter)
    {
      outFile << vertexIter->x << " "
	      << vertexIter->y << " "
	      << vertexIter->z << std::endl;
    }
  if (outFile.fail())
    {
      throw std::runtime_error("error writing byu vertices");
    }  

  //
  // write the facets, last vertex index in each row should be negative
  //
  for (ConstFacetIter facetIter = facets.begin();
       facetIter != facets.end();
       ++facetIter)
    {
      outFile << (*facetIter)[0] + 1 << " "
	      << (*facetIter)[1] + 1 << " "
	      << "-" << (*facetIter)[2] + 1 << std::endl;
    }
  if (outFile.fail())
    {
      throw std::runtime_error("error writing byu facets");
    }  
}

void
Surface
::writeBYU(const std::string& filename) const 
{
  std::ofstream output(filename.c_str(), std::ios::binary);
  if (output.bad()) 
    {
      throw std::runtime_error("error opening file");
    }
  writeBYU(output);  
  output.close();
}
