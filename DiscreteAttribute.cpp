#include "DiscreteAttribute.h"
#include <assert.h>

std::string DiscreteAttribute::getValue()
{
	assert(known);
	return value;
}

DiscreteAttribute::DiscreteAttribute(std::string myValue)
{
	assert(myValue.length() > 0);
	value = std::string(myValue);
	type = DISCRETE;
	known = true;
}

DiscreteAttribute::DiscreteAttribute()
{
	known = false;
	value = "?";
	type = DISCRETE;
}
