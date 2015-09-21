#ifndef CONATTR_H
#define CONATTR_H

#include "Attribute.h"
#include <limits>
#include <assert.h>

class ContinuousAttribute: public Attribute
{
	public:
	float getValue();
	
	ContinuousAttribute(float myValue);
	
	ContinuousAttribute();
	
	private:
	float value;
	
};

#endif
