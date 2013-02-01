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

#ifndef SURFACE_H
#define SURFACE_H

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include <string>
#include <vector>
#include "Vector3D.h"
#include <iosfwd>

#include <AffineTransform3D.h>

//
// a simple surface class for triangular
// surfaces with normal and unit normal computation
//
// bcd 2003
//

class Surface 
{
public:
  typedef Vector3D<double>             Vertex;
  typedef std::vector<Vertex>          VertexList;
  typedef VertexList::iterator         VertexIter;
  typedef VertexList::const_iterator   ConstVertexIter;

  typedef Vector3D<double>             Normal;
  typedef std::vector<Normal>          NormalList;
  typedef NormalList::iterator         NormalIter;
  typedef NormalList::const_iterator   ConstNormalIter;

  typedef std::vector<unsigned int>    Facet;
  typedef std::vector<Facet>           FacetList;
  typedef FacetList::iterator          FacetIter;
  typedef FacetList::const_iterator    ConstFacetIter;

  VertexList    vertices;     
  NormalList    normals;      // normal at each vertex
  NormalList    unitNormals;  
  NormalList    facetNormals;  
  FacetList     facets;       // vertex lists
  
  Surface();
  Surface(const Surface& rhs);
  Surface& operator=(const Surface& rhs);

  void clear();

  unsigned int addVertex(const Vertex& vertex); 
  unsigned int addVertex(const double& x,
			 const double& y,
			 const double& z);

  unsigned int addFacet(const Facet& facet);
  unsigned int addFacet(unsigned int vertex1, 
			unsigned int vertex2, 
			unsigned int vertex3);

  unsigned int numVertices() const;
  unsigned int numFacets()   const;
  
  // min gets the minimum x, y, and z vertex list values
  // max gets the maximum x, y, and z vertex list values
  void getAxisExtrema(Vector3D<double>& min, 
		      Vector3D<double>& max) const;

  Vector3D<double> getCentroid() const;

  void scale(const Vector3D<double>& scaleFactor);
  void scale(const double& sx, 
	     const double& sy, 
	     const double& sz);

  void applyAffineTransform(const AffineTransform3D<double>& transform);

  void translate(const Vector3D<double>& t);
  void translate(const double& tx, 
		 const double& ty,
		 const double& tz);

  // must be called before accessing normals
  void computeNormals();
  void computeUnitNormals();
  
  // file io
  void readBYU(const std::string& filename);
  void readBYU(std::istream &input = std::cin);
  void readOFF(const std::string& filename);
  void readOFF(std::istream &input = std::cin);
  void writeBYU(const std::string& filename) const;
  void writeBYU(std::ostream &output = std::cout) const;
};
#include "Surface.inl"
#endif
