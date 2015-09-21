/*Brian Davis
 * ID3 project
 * 
 * Attribute
 *This class  defines the generalization of the value of a particular attribute of a 
 *learning instance.
 */
 
#ifndef ATTR_H
#define ATTR_H

#include <string>

enum AttributeType { BINARY, CONTINUOUS, DISCRETE };

class Attribute
{
	public:
	AttributeType getType() const;
	bool isKnown() const;
	float getCValue() const;
	std::string getDValue() const;
	Attribute(float myValue);
	Attribute(std::string myValue);
	Attribute(AttributeType mytype);
	Attribute(const Attribute& toCopy);
	
	protected:
	AttributeType type;
	float cValue;
	std::string dValue;
	bool known;
};


#endif
