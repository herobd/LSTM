#include "ContinuousAttribute.h"

float ContinuousAttribute::getValue()
{
	assert(known);
	return value;
}

ContinuousAttribute::ContinuousAttribute(float myValue)
{
	value = myValue;
	type = CONTINUOUS;
	known = true;
}

ContinuousAttribute::ContinuousAttribute()
{
	known = false;
	type = CONTINUOUS;
	value = std::numeric_limits<float>::min();
}
