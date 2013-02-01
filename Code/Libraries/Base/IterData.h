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

#ifndef IterData_h
#define IterData_h

#include <string>
#include <map>
using namespace std;

typedef std::map<std::string,std::string> MessageMap;


class Iteration
{
	public:
	int iter;
	float error;
	string name;;
	Iteration()
		: iter(0), error(0)
	{
		this->name = "Iteration";
	}
	Iteration(int iter, float error)
	{
		this->iter = iter;
		this->error = error;
	}
};
class IterData
{
	public:
	string m_name;
	int currentIter;
	list<Iteration> iterations;
	void saveData(const char* fileName)
	{
	        TiXmlDocument doc;
		string s;
		TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );
		doc.LinkEndChild( decl );

		TiXmlElement * root = new TiXmlElement("IterData");
		doc.LinkEndChild( root );

		// block: CurrentIter
		{
			TiXmlElement * current = new TiXmlElement( "CurrentIter" );
			root->LinkEndChild( current );

			current->SetAttribute("currentIter",currentIter);
		}

		// block Iterations
		{
			TiXmlElement * iteratorsNode = new TiXmlElement( "Iterator" );
			root->LinkEndChild( iteratorsNode );

			list<Iteration>::iterator tor;
			for(tor=iterations.begin(); tor!=iterations.end(); tor++)
			{
				const Iteration& i=*tor;
				TiXmlElement * iteration;
				iteration = new TiXmlElement("Iteration");
				iteratorsNode->LinkEndChild(iteration);
				iteration->SetAttribute("iter",i.iter);
				iteration->SetAttribute("error",i.error);
			}
		}
		doc.SaveFile(fileName);
	}

	IterData()
	{
		this->m_name = "IterData";
	}
	void setCurrentIter(int iter)
	{
		this->currentIter = iter;
	}
	void clearIterations()
	{
		iterations.clear();
	}
	void setValues(int iter, float error)
	{
		iterations.push_back(Iteration(iter,error));
	}
};
/*
void IterData::saveData(const char* fileName)
{
	TiXmlDocument doc;
	string s;
 	TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );  
	doc.LinkEndChild( decl );

	TiXmlElement * root = new TiXmlElement("IterData");  
	doc.LinkEndChild( root ); 

	// block: CurrentIter
	{
		TiXmlElement * current = new TiXmlElement( "CurrentIter" );  
		root->LinkEndChild( current );

		current->SetAttribute("currentIter",currentIter);
	}

	// block Iterations
	{
		TiXmlElement * iteratorsNode = new TiXmlElement( "Iterator" );  
		root->LinkEndChild( iteratorsNode ); 

		list<Iteration>::iterator tor;
		for(tor=iterations.begin(); tor!=iterations.end(); tor++)
		{
			const Iteration& i=*tor;
			TiXmlElement * iteration;
			iteration = new TiXmlElement("Iteration");
			iteratorsNode->LinkEndChild(iteration);
			iteration->SetAttribute("iter",i.iter);
			iteration->SetAttribute("error",i.error);
		}
	}
	doc.SaveFile(fileName);
}
*/
#endif
