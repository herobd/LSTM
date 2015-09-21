#include "Attribute.h"
#include <assert.h>

AttributeType Attribute::getType() const
{
	return type;
}
	
bool Attribute::isKnown() const
{
	return known;
}

float Attribute::getCValue() const
{
	assert(known && type == CONTINUOUS);
	return cValue;
}

Attribute::Attribute(float myValue)
{
	cValue = myValue;
	type = CONTINUOUS;
	known = true;
}

Attribute::Attribute(const Attribute& toCopy)
{
	type = toCopy.getType();
	known = toCopy.isKnown();
	if (type == DISCRETE && known)
	{
		dValue = toCopy.getDValue();
	}
	else if (type == CONTINUOUS && known)
	{
		cValue = toCopy.getCValue();
	}
	
}

std::string Attribute::getDValue() const
{
	assert(known && type == DISCRETE);
	return dValue;
}

Attribute::Attribute(std::string myValue)
{
	assert(myValue.length() > 0);
	dValue = std::string(myValue);
	type = DISCRETE;
	known = true;
}

Attribute::Attribute(AttributeType mytype)
{
	type = mytype;
	known = false;
}
