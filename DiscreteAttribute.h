
#ifndef DISATTR_H
#define DISATTR_H


//#include <assert.h>
#include <string>
#include "Attribute.h"

class DiscreteAttribute: public Attribute
{
	public:
	std::string getValue();
	
	DiscreteAttribute(std::string myValue);
	
	DiscreteAttribute();
	
	private:
	std::string value;
	
};


#endif
