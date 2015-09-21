#include <string>
#include "ArffReader.h"
#include "Attribute.h"
#include <vector>
#include <iostream>
#include <fstream>
#include "LSTM.h"
#include <string.h>

using namespace std;


int main(int argc, char* argv[])
{

	double learningRate = .1;
	//double criticalError = .001;//based on arbitrary running tests
	//bool useMomentumTerm;
	double momentumTerm=.5;
	string fileNameTrain = "";
	string fileNameTest = "";
	string outFileName = "out";
	string netFileName = "";
	string netFileNameLoad = "";
	bool loadNN=false;
	bool saveNN=false;
	int LIMIT =30;
	int numOfBlocks = 10;
	int numOfCellsInBlock = 5;
	int numOfHiddenNodes = 10;
		
	
	
	//read in arguments
	for (int i = 1; i < argc; i++)
	{
		char* arg = argv[i];
		if (strcmp(arg,"-m")==0)
		{
			//useMomentumTerm = true;
			arg = argv[++i];
			momentumTerm = atof(arg);
			cout << "Using momentum term " << momentumTerm << endl;
			continue;
		}
		else if (strcmp(arg,"-l")==0)
		{
			arg = argv[++i];
			learningRate = atof(arg);
			cout << "Learning rate set to " << learningRate << endl;
			continue;
		}
		else if (strcmp(arg,"-o")==0)
		{
			outFileName = argv[++i];
			cout << "OutFile set to " << outFileName << endl;
			continue;
		}
		else if (strcmp(arg,"-test")==0)
		{
			fileNameTest = argv[++i];
			cout << "Test file is " << fileNameTest << endl;
			continue;
		}
		else if (strcmp(arg,"-train")==0)
		{
			fileNameTrain = argv[++i];
			cout << "Train file is " << fileNameTrain << endl;
			continue;
		}
		else if (strcmp(arg,"-limit")==0)
		{
			arg = argv[++i];
			LIMIT = atof(arg);
			continue;
		}
		else if (strcmp(arg,"-blocks")==0)
		{
			arg = argv[++i];
			numOfBlocks = atoi(arg);
			continue;
		}
		else if (strcmp(arg,"-cells")==0)
		{
			arg = argv[++i];
			numOfCellsInBlock = atoi(arg);
			continue;
		}
		else if (strcmp(arg,"-hiddens")==0 || strcmp(arg,"-hidden")==0)
		{
			arg = argv[++i];
			numOfHiddenNodes = atoi(arg);
			continue;
		}
		else if (strcmp(arg,"-saveNN")==0)
		{
			saveNN=true;
			netFileName = argv[++i];
			continue;
		}

		else if (strcmp(arg,"-loadNN")==0)
		{
			loadNN=true;
			netFileNameLoad = argv[++i];
			continue;
		};
	}
	
	
	
	//READ IN DATA
	
	
	//Initailize navigators for the attribute input and output
	map<int/*attribute*/, map<string/*value*/, int/*inputNode*/>*> discreteAttributeNodeMap;
	map<int/*attribute*/, int/*inputNode*/> continuousAttributeNodeMap;
	
	ArffReader readerTrain(fileNameTrain);
	ArffReader readerTest(fileNameTest);
	
	cout << "Reader initailized." << endl;
	int numOfAttributes = readerTrain.getTargetIndex();
	//AttributeType attributeTypeKey[numOfAttributes];
	int currentNode = 0;
	int numOfNodesNeededForDiscreteValues = 0;
	
	for (int i = 0; i < numOfAttributes; i++)
	{
		AttributeType type = readerTrain.getTypeForAttribute(i);
		if (type == CONTINUOUS)
		{
			continuousAttributeNodeMap[i] = currentNode++;
		}
		else if (type == DISCRETE)
		{
			vector<string> values = readerTrain.getValuesForDiscreteAttribute(i);
			map<string, int> *valueToNodeMap = new map<string, int>();
			for (unsigned int j = 0; j < values.size(); j++)
			{
				(*valueToNodeMap)[values[j]] = currentNode++;
				numOfNodesNeededForDiscreteValues++;
			}
			discreteAttributeNodeMap[i] = valueToNodeMap;
		}
	}
	int numOfInputNodes = numOfNodesNeededForDiscreteValues + continuousAttributeNodeMap.size();
	//cout << "Num of nodes at layer 0 is " << numOfNodesInLayer[0] << endl;
	
	vector<string> outputNodeClassMap = readerTrain.getValuesForDiscreteAttribute(numOfAttributes);
	int numOfOutputNodes = outputNodeClassMap.size();
	//cout << "Num of nodes at output layer is " << numOfNodesInLayer[numOfHiddenLayers+1] << endl;
	
	map<string, int> classOutputNodeMap = map<string, int>();
	for (unsigned int i = 0; i < outputNodeClassMap.size(); i++)
	{
		classOutputNodeMap[outputNodeClassMap[i]] = i;
	}
	
	
	cout << "Reading training in data." << endl;
	vector<vector<double>*> instancesTrain = vector<vector<double>*>();
	while(1)
	{
		vector<Attribute>* instance = readerTrain.nextInstance();
		if (instance != NULL)
		{
			//translate the instance into a NN input;
			vector<double>* input = new vector<double>();
			//init to all zeros (needed for distcrete attributes)
			for (int i = 0; i < numOfInputNodes; i++)
			{
				input->push_back(0);
			}
			
			for (unsigned int i = 0; i < instance->size()-1; i++)
			{
				Attribute attribute = (*instance)[i];
				if (attribute.getType() == CONTINUOUS)
				{
					double value = attribute.getCValue();
					(*input)[continuousAttributeNodeMap[i]] = value;
					
				}
				else if (attribute.getType() == DISCRETE)
				{
					string value = attribute.getDValue();
					(*input)[(*discreteAttributeNodeMap[i])[value]] = 1;
				}
				else
				{
					cout << "ERROR, invalid attribute." << endl;
					return -1;
				}
			}
			//The correct activiation node for the classification is going to be the last value
			//in the vector. This won't actually be used, but it just to simplify the passing process.
			string classification = (*instance)[instance->size()-1].getDValue();
			input->push_back(classOutputNodeMap[classification]);
			delete instance;
			assert(input->size()-1 == numOfInputNodes);
			instancesTrain.push_back(input);
		}
		else
		{
			break;
		}
	}
	
	int numOfInstancesTrain = instancesTrain.size();
	cout << "Read in finished, " << numOfInstancesTrain << " training instances found." << endl;
	
	
	cout << "Reading Test in data." << endl;
	vector<vector<double>*> instancesTest = vector<vector<double>*>();
	while(1)
	{
		vector<Attribute>* instance = readerTest.nextInstance();
		if (instance != NULL)
		{
			//translate the instance into a NN input;
			vector<double>* input = new vector<double>();
			//init to all zeros (needed for distcrete attributes)
			for (int i = 0; i < numOfInputNodes; i++)
			{
				input->push_back(0);
			}
			
			for (unsigned int i = 0; i < instance->size()-1; i++)
			{
				Attribute attribute = (*instance)[i];
				if (attribute.getType() == CONTINUOUS)
				{
					double value = attribute.getCValue();
					(*input)[continuousAttributeNodeMap[i]] = value;
					
				}
				else if (attribute.getType() == DISCRETE)
				{
					string value = attribute.getDValue();
					(*input)[(*discreteAttributeNodeMap[i])[value]] = 1;
				}
				else
				{
					cout << "ERROR, invalid attribute." << endl;
					return -1;
				}
			}
			//The correct activiation node for the classification is going to be the last value
			//in the vector. This won't actually be used, but it just to simplify the passing process.
			string classification = (*instance)[instance->size()-1].getDValue();
			input->push_back(classOutputNodeMap[classification]);
			delete instance;
			assert(input->size()-1 == numOfInputNodes);
			instancesTest.push_back(input);
		}
		else
		{
			break;
		}
	}
	
	int numOfInstancesTest = instancesTest.size();
	cout << "Read in finished, " << numOfInstancesTest << " Test instances found." << endl;
	
	LSTM* lstm;
	
	if (loadNN)
	{
		lstm = new LSTM(netFileNameLoad);
	}
	else
		lstm = new LSTM(numOfInputNodes, numOfOutputNodes, numOfBlocks, numOfCellsInBlock, numOfHiddenNodes,
			learningRate, momentumTerm);
	ofstream outfile;
	outfile.open(outFileName);
	lstm->train(instancesTrain, instancesTest, LIMIT, outfile, netFileName);
	double tacc = lstm->test(instancesTest);
	cout << "Final accuracy on TESTing data: " <<tacc<<endl;
	outfile << "Final accuracy on TESTing data: " <<tacc<<endl;
	outfile.close();
	
	return 1;
}
