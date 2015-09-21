/*For NN lab*/


#ifndef ARFFREADER_H
#define ARFFREADER_H

#include <string>
#include <fstream>
#include "Attribute.h"
#include <vector>
#include <map>


using namespace std;

class ArffReader
{
	public:
	//void read(string fileName);
	string getAttributeName(int i);
	vector<string> getAttributeNames();
	int getTargetIndex();//also, number of features
	vector<Attribute>* nextInstance();
	AttributeType getTypeForAttribute(int i) const;
	vector<string>& getValuesForDiscreteAttribute(int i);
	
	ArffReader(string fileName);
	ArffReader(const ArffReader&);
	~ArffReader();
	
	private:
	int targetIndex;
	vector<string> attributeNames;
	vector<AttributeType> attributeTypes; 
	ifstream in;
	
	void readHeaders();
	string trim(string in);
	
	map<int,vector<string>*> valuesForDiscreteAttributes;
	
	
};


#endif
