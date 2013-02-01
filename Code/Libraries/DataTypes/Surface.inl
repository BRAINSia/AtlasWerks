#ifndef SURFACE_INL
#define SURFACE_INL

inline
unsigned int 
Surface
::addVertex(const Vertex& vertex) 
{ 
  return addVertex(vertex.x, vertex.y, vertex.z);
}

inline
unsigned int 
Surface
::addVertex(const double& x,
	    const double& y,
	    const double& z)
{
  unsigned int i = 0;
  Vertex currVertex(x,y,z);
  for (ConstVertexIter vertIter = vertices.begin(); 
       vertIter != vertices.end(); 
       ++vertIter, ++i) 
    {
      if (*vertIter == currVertex)
	{
	  return i;
	}
    }
  vertices.push_back(currVertex);
  return vertices.size() - 1;
}

inline
unsigned int 
Surface
::addFacet(const Facet& facet)
{
  facets.push_back(facet);
  return facets.size() - 1;
}

inline
unsigned int 
Surface
::addFacet(unsigned int vertex1, 
	   unsigned int vertex2, 
	   unsigned int vertex3)
{
  Facet f(3);
  f[0] = vertex1;
  f[1] = vertex2;
  f[2] = vertex3;
  facets.push_back(f);
  return facets.size() - 1;
}

inline
unsigned int 
Surface
::numVertices() const 
{ 
  return vertices.size(); 
}

inline
unsigned int 
Surface
::numFacets() const 
{ 
  return facets.size();   
}
#endif
