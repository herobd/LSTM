
#include <assert.h>
#include <vector>
#include "ArffReader.h"
#include <string>
#include <assert.h>
#include <stdlib.h>
#include <cstddef>
//#include "DiscreteAttribute.h"
//#include "ContinuousAttribute.h"
#include <iostream>

#define BUFFER_MAX 3000

using namespace std;

ArffReader::ArffReader(string fileName)
{
	in.open(fileName.c_str());
	assert(!in.fail());
	valuesForDiscreteAttributes = map<int,vector<string>*>();
	readHeaders();
}

ArffReader::ArffReader(const ArffReader&)
{
	
}

ArffReader::~ArffReader()
{
	for (int i = 0; i < valuesForDiscreteAttributes.size(); i++)
	{
		delete valuesForDiscreteAttributes[i];
	}
}

string ArffReader::getAttributeName(int i)
{
	return attributeNames[i];
}

int ArffReader::getTargetIndex()
{
	return targetIndex;
}

vector<string> ArffReader::getAttributeNames()
{
	return attributeNames;
}

vector<Attribute>* ArffReader::nextInstance()
{
	char buffer [BUFFER_MAX];
	string line;
	while (in.good())
	{
		in.getline(buffer,BUFFER_MAX);
		line = string(buffer);
		int commentLoc = line.find_first_of('%');
		line = line.substr(0,commentLoc);
		//cout << "read-in line: " << line << endl;
		if (line.length() > 0)
		{
			line = trim(line);
			vector<Attribute>* toReturn = new vector<Attribute>();
			for (int i=0; i <= targetIndex; i++)
			{
				assert(line.length() > 0);
				int comma = line.find_first_of(',');
				//cout << "line is " << line << ", comma is " << comma << endl;
				string line2 = line.substr(comma+1);
				string attr = line.substr(0,comma);
				
				line = string(line2);
				//cout << "[" << i << "]" << attr << ", ";
				if (attr.compare("?")==0)
				{
					toReturn->push_back(Attribute(attributeTypes[i]));
				}
				else if (attributeTypes[i]==DISCRETE)
				{
					toReturn->push_back(Attribute(attr));
				}
				else if (attributeTypes[i]==CONTINUOUS)
				{
					toReturn->push_back(Attribute(atof(attr.c_str())));
				}
			}
			return toReturn;
		}
	}
	return NULL;
}

void ArffReader::readHeaders()
{
	cout << "readHeaders <start>" << endl;
	char buffer [BUFFER_MAX];
	string line;
	//int numOfAttributes = 0;
	
	while (in.good())
	{
		in.getline(buffer,BUFFER_MAX);
		line = trim(string(buffer));
		//cout << "read-in line: " << line << endl;
		if (line.length() > 0)
		{
			//cout << "looking at: " << line.substr(0,9) << endl;
			if (line.substr(0,9).compare("@RELATION")==0 || line.substr(0,9).compare("@relation")==0)
			{
				string relationName = line.substr(9);
				cout << "Reading relation: " << relationName << endl;
				break;
			}
		}
	}
	vector<string> names= vector<string>();
	vector<AttributeType> types = vector<AttributeType>();
	while (in.good())
	{
		
		in.getline(buffer,BUFFER_MAX);
		line = trim(string(buffer));
		if (line.length() > 0)
		{
			//cout << "line: " << line << endl;
			//cout << "looking at: " << line.substr(0,10) << endl;
			//cout << "or at: " << line.substr(0,5) << endl;
			if (line.substr(0,10).compare("@ATTRIBUTE")==0 || line.substr(0,10).compare("@attribute")==0)
			{
				int first_ = line.find_first_of(' ');
				int next_ = line.find_first_of(' ',first_+1);
				int nextT = line.find_first_of('\t',first_+1);
				//cout << "first:" << first_ << ", next_:" << next_ << ", nextT:" << nextT << endl;
				if (nextT < next_ || next_==-1)
					next_ = nextT;
				string attrName = trim(line.substr(first_+1,next_-first_-1));
				//cout << "name: " << attrName << endl;
				string attrType = trim(line.substr(next_+1));
				//cout << "type: " << attrType << endl;
				names.push_back(attrName);
				if (attrType.compare("continuous") == 0 ||
					attrType.compare("CONTINUOUS") == 0 ||
					attrType.compare("real") == 0 ||
					attrType.compare("REAL") == 0 ||
					attrType.compare("numeric") == 0 ||
					attrType.compare("NUMERIC") == 0)
				{
					types.push_back(CONTINUOUS);
					//cout << "Added continuous attr: " << attrName << endl;
				}
				else
				{
					types.push_back(DISCRETE);
					//cout << "Added discrete attr: " << attrName << endl;
					
					// actually read in the values
					int openBracket = line.find_first_of('{');
					int closeBracket = line.find_last_of('}');
					string valueList = line.substr(openBracket+1,closeBracket-(openBracket+1));
					vector<string>* valueVector = new vector<string>();
					while (1)
					{
						//cout << "List is " << valueList << endl;
						int nextComma = valueList.find_first_of(',');
						if (nextComma == -1 || nextComma == string::npos)
						{
							valueVector->push_back(trim(valueList));
							break;
						}
						else
						{
							valueVector->push_back(trim(valueList.substr(0,nextComma)));
							valueList = valueList.substr(nextComma+1);
						}
					}
					//cout << "Put " << valueVector << " at index " << types.size()-1 << endl;
					valuesForDiscreteAttributes[types.size()-1] = valueVector;
					//valuesForDiscreteAttributes.push_back(valueVector);
				}
				
			}
			else if (line.substr(0,5).compare("@DATA")==0 || line.substr(0,5).compare("@data")==0)
			{
				break;
			}
		}
		attributeNames = names;
		attributeTypes = types;
		assert(names.size() == types.size());
		targetIndex = types.size()-1;
	}
	
}

string ArffReader::trim(string in)
{
	//trim function from http://stackoverflow.com/questions/479080/trim-is-not-part-of-the-standard-c-c-library
	size_t s = in.find_first_not_of(" \n\r\t");
   	size_t e = in.find_last_not_of (" \n\r\t");
	if(( string::npos == s) || ( string::npos == e))
	      return  "";
    	else
	      return in.substr(s, e-s+1);
}

AttributeType ArffReader::getTypeForAttribute(int i) const
{
	return attributeTypes[i];
}

vector<string>& ArffReader::getValuesForDiscreteAttribute(int i)
{
	return *(valuesForDiscreteAttributes[i]);
}
