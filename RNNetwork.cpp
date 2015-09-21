
#include "RNNetwork.h"
#include <assert.h>
#include <random>
#include <math.h>
#include <iostream>

#define LAST_LAYER numOfHiddenLayers+1


using namespace std;
//weights[layerfrom][nodefrom][nodeto]   rWeights[layerfrom][nodefrom][layerto][nodeto]
NNetwork::NNetwork(int numOfHiddenLayers, int* numOfNodesInLayer, map<int, map<int, vector<tuple<int,int> > > >* recurMap, bool useMomentumTerm, double momentumTerm, double learningRate, int k)
{
	
	this->numOfHiddenLayers = numOfHiddenLayers;
	this->numOfNodesInLayer = numOfNodesInLayer;
	this->recurMap = recurMap;
	this->useMomentumTerm = useMomentumTerm;
	this->momentumTerm = momentumTerm;
	this->learningRate = learningRate;
	this->k = k-1;
	
	//initailize three-dimensional weights array
	weights = new double**[numOfHiddenLayers+1];//[layer from][node from][node to]
	for (int i = 0; i < numOfHiddenLayers+1; i++)
	{
		int firstLayer = numOfNodesInLayer[i];
		int secondLayer = numOfNodesInLayer[i+1];
		double** layerWeights = new double*[firstLayer];
		for (int layer = 0; layer < firstLayer; layer++)
		{
			layerWeights[layer] = new double[secondLayer];
			
		}
		weights[i] = layerWeights;
	}
	
	bWeights = new double*[numOfHiddenLayers+1];
	oldBWeights = new double*[numOfHiddenLayers+1];
	for (int i=0; i<numOfHiddenLayers+1; i++)
	{
		bWeights[i] = new double[numOfNodesInLayer[i+1]];
		oldBWeights[i] = new double[numOfNodesInLayer[i+1]];
	}
	
	rWeights = new map<int,map<int,map<int,map<int,double> > > >;
	oldRWeights = new map<int,map<int,map<int,map<int,double> > > >;
	 
	
	previousWeightDeltas = new double***[k+1];
	weightDeltas = new double***[k+1];
	previousBWeightDeltas = new double**[k+1];
	bWeightDeltas = new double**[k+1];
	for (int t=k; t>=0; t--)
	{
		previousWeightDeltas[t] = new double**[numOfHiddenLayers+1];
		weightDeltas[t] = new double**[numOfHiddenLayers+1];
		previousBWeightDeltas[t] = new double*[numOfHiddenLayers+1];
		bWeightDeltas[t] = new double*[numOfHiddenLayers+1];
		for (int layer = 0; layer < numOfHiddenLayers+1; layer++)
		{
			int firstLayer = numOfNodesInLayer[layer];
			int secondLayer = numOfNodesInLayer[layer+1];
			double** layerWeightDeltas = new double*[firstLayer];
			double** layerWeightDeltas2 = new double*[firstLayer];
			for (int node = 0; node < firstLayer; node++)
			{
				layerWeightDeltas[node] = new double[secondLayer];
				layerWeightDeltas2[node] = new double[secondLayer];
				for (int node2 = 0; node2 < secondLayer; node2++)
				{
					layerWeightDeltas[node][node2] = 0;//init to zero
				}
			}
			previousWeightDeltas[t][layer] = layerWeightDeltas;
			weightDeltas[t][layer] = layerWeightDeltas2;
			previousBWeightDeltas[t][layer] = new double[secondLayer];
			bWeightDeltas[t][layer] = new double[secondLayer];
		}
	}
	
	previousRWeightDeltas = new map<int,map<int,map<int,map<int,double> > > >[k+1];
	rWeightDeltas = new map<int,map<int,map<int,map<int,double> > > >[k+1];
	
#if TRACE
	presetWeights();
#else
	initRandomWeights();
#endif
	
	out = new double**[this->k+2];//[t][layer][node]
	for (int t=this->k+1; t>=0; t--)
	{
		out[t] = new double*[numOfHiddenLayers+2];
		for (int layer = 0; layer < numOfHiddenLayers+2; layer++)
		{
			out[t][layer] = new double[numOfNodesInLayer[layer]];
		}
	}
	
	net = new double**[this->k+1];//[t][layer][node]
	for (int t=this->k; t>=0; t--)
	{
		net[t] = new double*[numOfHiddenLayers+2];
		for (int layer = 0; layer < numOfHiddenLayers+2; layer++)
		{
			net[t][layer] = new double[numOfNodesInLayer[layer]];
		}
	}
	
	nodeDeltas = new double**[this->k+1];//[t][layer][node]
	for (int t=this->k; t>=0; t--)
	{
		nodeDeltas[t] = new double*[numOfHiddenLayers+2];
		for (int layer = 0; layer < numOfHiddenLayers+2; layer++)//skips 0 as thats the input
		{
			nodeDeltas[t][layer] = new double[numOfNodesInLayer[layer]];
		}
	}
	
	
	
	oldWeights = new double**[numOfHiddenLayers+1];
	for (int layer = 0; layer < numOfHiddenLayers+1; layer++)
	{
		int firstLayer = numOfNodesInLayer[layer];
		int secondLayer = numOfNodesInLayer[layer+1];
		double** layerWeights = new double*[firstLayer];
		for (int node = 0; node < firstLayer; node++)
		{
			layerWeights[node] = new double[secondLayer];
		}
		oldWeights[layer] = layerWeights;
	}
}

////////////
////LOAD////
////////////
NNetwork::NNetwork(string fileName)
{
	ifstream in;
	in.open(fileName);
	string line;
	getline (in,line);
	this->numOfHiddenLayers = stoi(line);
	this->numOfNodesInLayer = new int[numOfHiddenLayers+2];
	for (int i=0; i<numOfHiddenLayers+2; i++)
	{
		getline (in,line);
		numOfNodesInLayer[i]=stoi(line);
	}
	this->recurMap = new map<int, map<int, vector<tuple<int,int> > > >();
	
	this->useMomentumTerm = true;
	getline (in,line);
	this->momentumTerm = stod(line);
	getline (in,line);
	this->learningRate = stod(line);
	getline (in,line);
	this->k = stoi(line);
	
	//initailize three-dimensional weights array
	weights = new double**[numOfHiddenLayers+1];//[layer from][node from][node to]
	for (int i = 0; i < numOfHiddenLayers+1; i++)
	{
		int firstLayer = numOfNodesInLayer[i];
		int secondLayer = numOfNodesInLayer[i+1];
		double** layerWeights = new double*[firstLayer];
		for (int layer = 0; layer < firstLayer; layer++)
		{
			layerWeights[layer] = new double[secondLayer];
			for (int layer2 = 0; layer2 < secondLayer; layer2++)
			{
				getline (in,line);
				layerWeights[layer][layer2] = stod(line);
			}
		}
		weights[i] = layerWeights;
	}
	
	bWeights = new double*[numOfHiddenLayers+1];
	oldBWeights = new double*[numOfHiddenLayers+1];
	for (int i=0; i<numOfHiddenLayers+1; i++)
	{
		bWeights[i] = new double[numOfNodesInLayer[i+1]];
		oldBWeights[i] = new double[numOfNodesInLayer[i+1]];
		for (int node = 0; node < numOfNodesInLayer[i+1]; node++)
		{
			getline (in,line);
			bWeights[i][node] = stod(line);
		}
	}
	
	rWeights = new map<int,map<int,map<int,map<int,double> > > >;
	oldRWeights = new map<int,map<int,map<int,map<int,double> > > >;
	 
	
	previousWeightDeltas = new double***[k+1];
	weightDeltas = new double***[k+1];
	previousBWeightDeltas = new double**[k+1];
	bWeightDeltas = new double**[k+1];
	for (int t=k; t>=0; t--)
	{
		previousWeightDeltas[t] = new double**[numOfHiddenLayers+1];
		weightDeltas[t] = new double**[numOfHiddenLayers+1];
		previousBWeightDeltas[t] = new double*[numOfHiddenLayers+1];
		bWeightDeltas[t] = new double*[numOfHiddenLayers+1];
		for (int layer = 0; layer < numOfHiddenLayers+1; layer++)
		{
			int firstLayer = numOfNodesInLayer[layer];
			int secondLayer = numOfNodesInLayer[layer+1];
			double** layerWeightDeltas = new double*[firstLayer];
			double** layerWeightDeltas2 = new double*[firstLayer];
			for (int node = 0; node < firstLayer; node++)
			{
				layerWeightDeltas[node] = new double[secondLayer];
				layerWeightDeltas2[node] = new double[secondLayer];
				for (int node2 = 0; node2 < secondLayer; node2++)
				{
					layerWeightDeltas[node][node2] = 0;//init to zero
				}
			}
			previousWeightDeltas[t][layer] = layerWeightDeltas;
			weightDeltas[t][layer] = layerWeightDeltas2;
			previousBWeightDeltas[t][layer] = new double[secondLayer];
			bWeightDeltas[t][layer] = new double[secondLayer];
		}
	}
	
	previousRWeightDeltas = new map<int,map<int,map<int,map<int,double> > > >[k+1];
	rWeightDeltas = new map<int,map<int,map<int,map<int,double> > > >[k+1];
	
	
	out = new double**[k+2];//[t][layer][node]
	for (int t=k+1; t>=0; t--)
	{
		out[t] = new double*[numOfHiddenLayers+2];
		for (int layer = 0; layer < numOfHiddenLayers+2; layer++)
		{
			out[t][layer] = new double[numOfNodesInLayer[layer]];
			
			//test
			/*for (int node=0; node<numOfNodesInLayer[layer]; node++)
			{
				assert(out[t][layer] != NULL);
				out[t][layer][node] = 0;
			}*/
		}
	}
	//cout << "out[0]: " << out[0] << ", out[1]: " << out[1] << endl;
	//cout << "out[0][0]: " << out[0][0] << ", out[0][1]: " << out[0][1] << endl;
	
	net = new double**[k+1];//[t][layer][node]
	for (int t=k; t>=0; t--)
	{
		net[t] = new double*[numOfHiddenLayers+2];
		for (int layer = 0; layer < numOfHiddenLayers+2; layer++)
		{
			net[t][layer] = new double[numOfNodesInLayer[layer]];
		}
	}
	
	nodeDeltas = new double**[k+1];//[t][layer][node]
	for (int t=k; t>=0; t--)
	{
		nodeDeltas[t] = new double*[numOfHiddenLayers+2];
		for (int layer = 0; layer < numOfHiddenLayers+2; layer++)//skips 0 as thats the input
		{
			nodeDeltas[t][layer] = new double[numOfNodesInLayer[layer]];
		}
	}
	
	
	
	oldWeights = new double**[numOfHiddenLayers+1];
	for (int layer = 0; layer < numOfHiddenLayers+1; layer++)
	{
		int firstLayer = numOfNodesInLayer[layer];
		int secondLayer = numOfNodesInLayer[layer+1];
		double** layerWeights = new double*[firstLayer];
		for (int node = 0; node < firstLayer; node++)
		{
			layerWeights[node] = new double[secondLayer];
		}
		oldWeights[layer] = layerWeights;
	}
	
	//assert(out[0][0] != NULL);
}

NNetwork::~NNetwork()
{
	delete[] previousRWeightDeltas;
	delete[] rWeightDeltas;
	
	for (int t=k; t>=0; t--)
	{
		for (int layer = 0; layer < numOfHiddenLayers+2; layer++)
		{
			delete[] out[t][layer];
			delete[] net[t][layer];
			delete[] nodeDeltas[t][layer];
		}
		delete[] out[t];
		delete[] net[t];
		delete[] nodeDeltas[t];
	}
	for (int layer = 0; layer < numOfHiddenLayers+2; layer++)
	{
		delete[] out[k+1][layer];
	}
	delete[] out[k+1];
	
	delete[] out;
	delete[] net;
	delete[] nodeDeltas;
	
	for (int i = 0; i < LAST_LAYER; i++)
	{
		for (int j = 0; j < numOfNodesInLayer[i]; j++)
		{
			delete[] oldWeights[i][j];
		}
		delete[] oldWeights[i];
	}
	delete[] oldWeights;

	for (int i = 0; i < LAST_LAYER; i++)
	{
		for (int j = 0; j < numOfNodesInLayer[i]; j++)
		{
			delete[] weights[i][j];
		}
		delete[] weights[i];
	}
	delete[] weights;
	
	for (int t=k; t>=0; t--)
	{
		for (int i = 0; i < LAST_LAYER; i++)
		{
			for (int j = 0; j < numOfNodesInLayer[i]; j++)
			{
				delete[] previousWeightDeltas[t][i][j];
				delete[] weightDeltas[t][i][j];
			}
			delete[] weightDeltas[t][i];
			delete[] previousWeightDeltas[t][i];
			delete[] bWeightDeltas[t][i];
			delete[] previousBWeightDeltas[t][i];
		}
		delete[] weightDeltas[t];
		delete[] previousWeightDeltas[t];
		delete[] bWeightDeltas[t];
		delete[] previousBWeightDeltas[t];
	}
	
	delete[] weightDeltas;
	delete[] previousWeightDeltas;
	delete[] bWeightDeltas;
	delete[] previousBWeightDeltas;
	
	for (int i=0; i<numOfHiddenLayers+1; i++)
	{
		delete[] bWeights[i];
		delete[] oldBWeights[i];
	}
	delete[] bWeights;
	delete[] oldBWeights;
	delete rWeights;
	delete oldRWeights;
	delete recurMap;
	
	delete[] numOfNodesInLayer;
}

//This doesn't compute an exact mean of 0 for each layer, rather it simply relies of the standard deviation equation to gain an apro	zimate 
void NNetwork::initRandomWeights()
{
	//unsigned seed = 38838;
	default_random_engine generator;
	normal_distribution<double> distribution(0.0,0.2);
	
	for (int layerFrom = 0; layerFrom <= numOfHiddenLayers; layerFrom++)
	{
		for (int nodeFrom = 0; nodeFrom < numOfNodesInLayer[layerFrom]; nodeFrom++)
		{
			for (int nodeTo = 0; nodeTo < numOfNodesInLayer[layerFrom+1]; nodeTo++)
			{
				weights[layerFrom][nodeFrom][nodeTo] = distribution(generator);
				//oldWeights[layerFrom][nodeFrom][nodeTo] = 0;
			}
			
			
		}
	}
	
	//bWeights & rWeights
	for (int layerFrom = 0; layerFrom <= numOfHiddenLayers; layerFrom++)
	{
			for (int nodeTo = 0; nodeTo < numOfNodesInLayer[layerFrom+1]; nodeTo++)
			{
				bWeights[layerFrom][nodeTo] = distribution(generator);
				
				
				if ((*recurMap)[layerFrom][nodeTo].size() > 0)
				{
					for (tuple<int,int> recurConn: (*recurMap)[layerFrom][nodeTo])
					{
						int rLayer=get<0>(recurConn);
						int rNode=get<1>(recurConn);
						(*rWeights)[rLayer][rNode][layerFrom][nodeTo] = distribution(generator);
					}
				}
			}
		
	}
	
}

void NNetwork::presetWeights()
{
	weights[0][0][0] = -1.0;
	weights[0][0][1] = -1.0;
	weights[1][0][0] = 1.0;
	weights[1][1][0] = 1.0;
	
	bWeights[0][0] = 0;
	bWeights[0][1] = 0;
	bWeights[1][0] = 0;
	
	(*rWeights)[1][0][0][0]=.6;
	(*rWeights)[1][0][0][1]=.6;
	(*rWeights)[1][1][0][0]=.6;
	(*rWeights)[1][1][0][1]=.6;
}

vector<double> NNetwork::runOn(const vector<double> &input)
{
	assert((signed int) input.size()-1 == numOfNodesInLayer[0]);
	//assert(out[0][0] != 0x0 && out[0][0][88]!=12345);
	
	//cout <<"loop start"<<endl;
	for (int inNode = 0; inNode < numOfNodesInLayer[0]; inNode++)
	{
		//assert(out[0][0] != 0x0);
		out[0][0][inNode] = input[inNode];
		
		//cout << "loop out[0]: " << out[0] << ", out[1]: " << out[1];
		//cout << " /// out[0][0]: " << out[0][0] << ", out[0][1]: " << out[0][1] << endl;
	}
	
	
	run();
	
	
	
	//return
	vector<double> toReturn = vector<double>(numOfNodesInLayer[LAST_LAYER]);
	for (int outNode = 0; outNode < numOfNodesInLayer[LAST_LAYER]; outNode++)
	{
		toReturn[outNode] = out[0][LAST_LAYER][outNode];
	}
	
	swapOuts();
	
	
	return toReturn;
}

void NNetwork::burnInOn(const vector<vector<double> > &input)
{
	assert((signed int) input[0].size()-1 == numOfNodesInLayer[0]);
#if (!TRACE)
	assert(input.size() > k);
#endif
	
	//setup t=1
	//if (k>0)
	for (int layer = 0; layer < numOfHiddenLayers+2; layer++)
	{
		for (int node=0; node<numOfNodesInLayer[layer]; node++)
		{
			out[1][layer][node]=RECUR_START;
		}
	}
	
	for (int t=0; t<=k-TRACE; t++)
	{
		for (int inNode = 0; inNode < numOfNodesInLayer[0]; inNode++)
		{
			out[0][0][inNode] = input[t][inNode];
		}
		//printOut();
		
		run();
		//cout << "burned" << endl;
		swapOuts();
	}
	
	
	
	//no return
}

 	
vector<double> NNetwork::trainOn(const vector<double> &testInstance, int targetActivationNode)
{
	assert((signed int)testInstance.size()-1 == numOfNodesInLayer[0]);
	
	//Initailize the storage arrays.
	
	
	///
	
	//initailize the input data into the storage
	for (int inNode = 0; inNode < numOfNodesInLayer[0]; inNode++)
	{
		out[0][0][inNode] = testInstance[inNode];
	}
	
	run();
	
	////-Backpropogate-////
	
	
	copyWeightsTo();
	
	//
	BPTT(targetActivationNode);
	////
	
	
	//PRINT
	// cout << "Weights:" << endl;
	// for (int i = 0; i < 4; i++)
	// {
	// 	for (int j = 0; j < 3; j++)
	// 	{
	// 		cout << "From in" << i << " to hidden" << j << " = " << weights[0][i][j] << ",  ";
	// 	}
	// }
	// cout << endl;
	
	// for (int i = 0; i < 3; i++)
	// {
	// 	for (int j = 0; j < 3; j++)
	// 	{
	// 		cout << "From hidden" << i << " to out" << j << " = " << weights[1][i][j] << ",  ";
	// 	}
	// }
	// cout << endl;
	// cout << endl;
	
	
	//save previous
	
	double**** temp = previousWeightDeltas;
	previousWeightDeltas = weightDeltas;
	weightDeltas=temp;
	
	double*** temp2 = previousBWeightDeltas;
	previousBWeightDeltas = bWeightDeltas;
	bWeightDeltas=temp2;
	
	map<int,map<int,map<int,map<int,double> > > >* temp3 = previousRWeightDeltas;
	previousRWeightDeltas = rWeightDeltas;
	rWeightDeltas=temp3;
	
	
	vector<double> toReturn = vector<double>(numOfNodesInLayer[LAST_LAYER]);
	for (int outNode = 0; outNode < numOfNodesInLayer[LAST_LAYER]; outNode++)
	{
		toReturn[outNode] = out[0][LAST_LAYER][outNode];
	}
	
	swapOuts();

	return toReturn;
}

double* NNetwork::trainOnRetD(const vector<double> &testInstance, int targetActivationNode)
{
	assert((signed int)testInstance.size()-1 == numOfNodesInLayer[0]);
	
	//Initailize the storage arrays.
	
	
	///
	
	//initailize the input data into the storage
	for (int inNode = 0; inNode < numOfNodesInLayer[0]; inNode++)
	{
		out[0][0][inNode] = testInstance[inNode];
	}
	
	run();
	
	////-Backpropogate-////
	
	
	copyWeightsTo();
	
	//
	BPTT(targetActivationNode);
	////
	
	
	//set up ret
	for (int nodeTo = 0; nodeTo < numOfNodesInLayer[0]; nodeTo++)
	{
		nodeDeltas[0][0][nodeTo]=0;
		//		"previous" in the sense that we've already backpropogated it
		for (int nodeOnPreviousLayer = 0; nodeOnPreviousLayer < numOfNodesInLayer[1]; nodeOnPreviousLayer++) 
		{
			nodeDeltas[0][0][nodeTo] += nodeDeltas[0][1][nodeOnPreviousLayer] * oldWeights[0][nodeTo][nodeOnPreviousLayer];
		}
		// deal with "previous" recurrent layer +nodeDeltas
		
		nodeDeltas[0][0][nodeTo]*= out[0][0][nodeTo] * (1-out[0][0][nodeTo]);
	}
	
	//save previous
	
	double**** temp = previousWeightDeltas;
	previousWeightDeltas = weightDeltas;
	weightDeltas=temp;
	
	double*** temp2 = previousBWeightDeltas;
	previousBWeightDeltas = bWeightDeltas;
	bWeightDeltas=temp2;
	
	map<int,map<int,map<int,map<int,double> > > >* temp3 = previousRWeightDeltas;
	previousRWeightDeltas = rWeightDeltas;
	rWeightDeltas=temp3;
	
	
	
	
	swapOuts();

	return nodeDeltas[0][0];
}

//double*** [time step][layer][node]
void NNetwork::run()
{
	//resetNet();
	// for (int t=k; t>=0; t--)
	// {
		int t=0;
		for (int layer = 0; layer < LAST_LAYER; layer++)
		{
		
			for (int nodeTo = 0; nodeTo < numOfNodesInLayer[layer+1]; nodeTo++)
			{
				net[t][layer+1][nodeTo]=0;
				for (int nodeFrom = 0; nodeFrom < numOfNodesInLayer[layer]; nodeFrom++)
				{
					net[t][layer+1][nodeTo] += weights[layer][nodeFrom][nodeTo] * out[t][layer][nodeFrom];
				}
				//bias node
				net[t][layer+1][nodeTo] += bWeights[layer][nodeTo] * BIAS_OUT;
				
				
				if ((*recurMap).count(layer) && (*recurMap)[layer].count(nodeTo))
				{
					
					for (tuple<int,int> recurConn: (*recurMap)[layer][nodeTo])
					{
						int rLayer=get<0>(recurConn);
						int rNode=get<1>(recurConn);
						//if (t==k)
						//{
						//	net[t][layer+1][nodeTo] += (*rWeights)[rLayer][rNode][layer][nodeTo] * RECUR_START;
						//}
						//else
						{
							net[t][layer+1][nodeTo] += (*rWeights)[rLayer][rNode][layer][nodeTo] * out[t+1][rLayer][rNode];
							
							//cout << "\tnode[1]["<<rLayer<<"]["<<rNode<<"] out="<<out[t+1][rLayer][rNode]<<endl;
							//cout << "sum[0]["<<layer+1<<"]["<<nodeTo<<"] add " << (*rWeights)[rLayer][rNode][layer][nodeTo] * out[t+1][rLayer][rNode] << endl;
						}
					}
				}
				
				out[t][layer+1][nodeTo] = activationFunction(net[t][layer+1][nodeTo]);
				
				//cout << "node[0]["<<layer+1<<"]["<<nodeTo<<"] sum="<<net[t][layer+1][nodeTo]<<" out="<<out[t][layer+1][nodeTo]<<endl;
			}
		}
		
		
	// }
}



double NNetwork::activationFunction(double net) const
{
	//This is currently implemented using the sigmoid function
	return 1.0 / (1.0 + exp(-1.0*net));
}

/*double NNetwork::afPrime(double net) const
{
	double f = activationFunction(net);
	return f * (1 - f);
}*/

void NNetwork::copyWeightsTo()
{
	/*for (int layer = 0; layer < numOfHiddenLayers+1; layer++)
	{
		int firstLayer = numOfNodesInLayer[layer];
		int secondLayer = numOfNodesInLayer[layer+1];
		//double** layerWeights = new double*[firstLayer];
		for (int node = 0; node < firstLayer; node++)
		{
			//layerWeights[node] = new double[secondLayer];
			for (int node2 = 0; node2 < secondLayer; node2++)
			{
				//layerWeights[node][node2] = weights[layer][node][node2];
				oldWeights[layer][node][node2] = weights[layer][node][node2];
			}
		}
		//here[layer] = layerWeights;
	}
	
	for (int layer=1; layer<numOfHiddenLayers+2; layer++)
	{
		for (int node = 0; node < numOfNodesInLayer[layer]; node++)
		{
			oldBWeights = bWeights[layer][node];
		}
	}*/
	double*** temp1;
	double** temp2;
	map<int,map<int,map<int,map<int,double> > > >* temp3;
	
	temp1 = oldWeights;
	oldWeights =weights;
	weights = temp1;
	
	temp2 = oldBWeights;
	oldBWeights = bWeights;
	bWeights = temp2;
	
	temp3 = oldRWeights;
	oldRWeights = rWeights;
	rWeights = temp3;
	
	setWeights();
}

void NNetwork::backpropOutLayer(int t,int targetActivationNode)
{

	for (int nodeTo = 0; nodeTo < numOfNodesInLayer[LAST_LAYER]; nodeTo++)
	{
		if (t==0)
		{
#if TRACE
			nodeDeltas[t][LAST_LAYER][nodeTo] = ((.9)- out[t][LAST_LAYER][nodeTo]) * out[t][LAST_LAYER][nodeTo] * (1-out[t][LAST_LAYER][nodeTo]);
#else
			nodeDeltas[t][LAST_LAYER][nodeTo] = (((nodeTo==targetActivationNode)?1:0)- out[t][LAST_LAYER][nodeTo]) * out[t][LAST_LAYER][nodeTo] * (1-out[t][LAST_LAYER][nodeTo]);
#endif
		}
		else
		{
			nodeDeltas[t][LAST_LAYER][nodeTo]=0;
			/*if ((*recurMap).count(LAST_LAYER) && (*recurMap)[LAST_LAYER].count(nodeTo))
			{
				for (auto recur : (*recurMap)[LAST_LAYER][nodeTo])
				{
					nodeDeltas[t][LAST_LAYER][nodeTo] += nodeDeltas[t-1][get<0>(recur)][get<1>(recur)] * (*oldRWeights)[get<0>(recur)][get<1>(recur)][LAST_LAYER][nodeTo];
				}
			}*/
			
			if ((*rWeights).count(LAST_LAYER) && (*rWeights)[LAST_LAYER].count(nodeTo))
			{
				for (auto& recurToLayerM1 : (*rWeights)[LAST_LAYER][nodeTo])
				{
					for (auto& recurToNode : recurToLayerM1.second)
					{
						nodeDeltas[t][LAST_LAYER][nodeTo] += nodeDeltas[t-1][recurToLayerM1.first+1][recurToNode.first] * (*oldRWeights)[LAST_LAYER][nodeTo][recurToLayerM1.first][recurToNode.first];
					}
				}
			}
			
			nodeDeltas[t][LAST_LAYER][nodeTo]*= out[t][LAST_LAYER][nodeTo] * (1-out[t][LAST_LAYER][nodeTo]);
		}
		
		for (int nodeFrom = 0; nodeFrom < numOfNodesInLayer[numOfHiddenLayers]; nodeFrom++)
		{
			weightDeltas[t][numOfHiddenLayers][nodeFrom][nodeTo] = learningRate * out[t][numOfHiddenLayers][nodeFrom] * nodeDeltas[t][LAST_LAYER][nodeTo];
			if (useMomentumTerm)
			{
				weightDeltas[t][numOfHiddenLayers][nodeFrom][nodeTo] += momentumTerm * previousWeightDeltas[t][numOfHiddenLayers][nodeFrom][nodeTo];
			}
			
			weights[numOfHiddenLayers][nodeFrom][nodeTo] += weightDeltas[t][numOfHiddenLayers][nodeFrom][nodeTo];
		}
		//bias
		bWeightDeltas[t][numOfHiddenLayers][nodeTo] = learningRate * BIAS_OUT * nodeDeltas[t][LAST_LAYER][nodeTo];
		if (useMomentumTerm)
		{
			bWeightDeltas[t][numOfHiddenLayers][nodeTo] += momentumTerm * previousBWeightDeltas[t][numOfHiddenLayers][nodeTo];
		}
		
		bWeights[numOfHiddenLayers][nodeTo] += bWeightDeltas[t][numOfHiddenLayers][nodeTo];		
		
		//recur
		if ((*recurMap)[numOfHiddenLayers][nodeTo].size() > 0)
		{
			for (tuple<int,int> recurConn: (*recurMap)[numOfHiddenLayers][nodeTo])
			{
				int rLayer=get<0>(recurConn);
				int rNode=get<1>(recurConn);
				//if (0==k)
				//{
				//	rWeightDeltas[t][rLayer][rNode][numOfHiddenLayers][nodeTo] = learningRate * RECUR_START * nodeDeltas[t][LAST_LAYER][nodeTo];
				//}
				//else
				{
					rWeightDeltas[t][rLayer][rNode][numOfHiddenLayers][nodeTo] = learningRate * out[t+1][rLayer][rNode] * nodeDeltas[t][LAST_LAYER][nodeTo];
					
				}
				if (useMomentumTerm)
				{
					rWeightDeltas[t][rLayer][rNode][numOfHiddenLayers][nodeTo] += momentumTerm * previousRWeightDeltas[t][rLayer][rNode][numOfHiddenLayers][nodeTo];
				}
				
				(*rWeights)[rLayer][rNode][numOfHiddenLayers][nodeTo] += rWeightDeltas[t][rLayer][rNode][numOfHiddenLayers][nodeTo];
			}
		}
	}
}

void NNetwork::backpropHiddenLayers(int t)
{
	double debug_sumDeltas=0;
	
	
	for (int layer = numOfHiddenLayers; layer > 0; layer--)
	{
		for (int nodeTo = 0; nodeTo < numOfNodesInLayer[layer]; nodeTo++)
		{
			nodeDeltas[t][layer][nodeTo]=0;
			//		"previous" in the sense that we've already backpropogated it
			for (int nodeOnPreviousLayer = 0; nodeOnPreviousLayer < numOfNodesInLayer[layer+1]; nodeOnPreviousLayer++) 
			{
				nodeDeltas[t][layer][nodeTo] += nodeDeltas[t][layer+1][nodeOnPreviousLayer] * oldWeights[layer][nodeTo][nodeOnPreviousLayer];
			}
			// deal with "previous" recurrent layer +nodeDeltas
			if (t!=0)
			{
				/*if ((*recurMap).count(layer) && (*recurMap)[layer].count(nodeTo))
				{
					for (auto recur : (*recurMap)[layer][nodeTo])
					{
						nodeDeltas[t][layer][nodeTo] += nodeDeltas[t-1][get<0>(recur)][get<1>(recur)] * (*oldRWeights)[get<0>(recur)][get<1>(recur)][layer][nodeTo];
						[0][1][0]
					}
				}*/
				
				if ((*rWeights).count(layer) && (*rWeights)[layer].count(nodeTo))
				{
					for (auto& recurToLayerM1 : (*rWeights)[layer][nodeTo])
					{
						for (auto& recurToNode : recurToLayerM1.second)
						{
							nodeDeltas[t][layer][nodeTo] += nodeDeltas[t-1][recurToLayerM1.first+1][recurToNode.first] * (*oldRWeights)[layer][nodeTo][recurToLayerM1.first][recurToNode.first];
						}
					}
				}
			}
			nodeDeltas[t][layer][nodeTo]*= out[t][layer][nodeTo] * (1-out[t][layer][nodeTo]);
			//cout << "nodeDeltas["<<t<<"]["<<layer<<"]["<<nodeTo<<"] = " << nodeDeltas[t][layer][nodeTo] << endl;
			
			for (int nodeFrom = 0; nodeFrom < numOfNodesInLayer[layer-1]; nodeFrom++)
			{
				weightDeltas[t][layer-1][nodeFrom][nodeTo] = learningRate * out[t][layer-1][nodeFrom] * nodeDeltas[t][layer][nodeTo];
				if (useMomentumTerm)
				{
					weightDeltas[t][layer-1][nodeFrom][nodeTo] += momentumTerm * previousWeightDeltas[t][layer-1][nodeFrom][nodeTo];
				}
				
				weights[layer-1][nodeFrom][nodeTo] += weightDeltas[t][layer-1][nodeFrom][nodeTo];
				
				debug_sumDeltas+=weightDeltas[t][layer-1][nodeFrom][nodeTo];
			}
			
			//bias
			bWeightDeltas[t][layer-1][nodeTo] = learningRate * BIAS_OUT * nodeDeltas[t][layer][nodeTo];
			if (useMomentumTerm)
			{
				bWeightDeltas[t][layer-1][nodeTo] += momentumTerm * previousBWeightDeltas[t][layer-1][nodeTo];
			}
			bWeights[layer-1][nodeTo] += bWeightDeltas[t][layer-1][nodeTo];
			debug_sumDeltas+=bWeightDeltas[t][layer-1][nodeTo];
			
			//recur
			if ((*recurMap)[layer-1][nodeTo].size() > 0)
			{
				for (tuple<int,int> recurConn: (*recurMap)[layer-1][nodeTo])
				{
					int rLayer=get<0>(recurConn);
					int rNode=get<1>(recurConn);
					// if (t==k)
					// {
					// 	rWeightDeltas[t][rLayer][rNode][layer-1][nodeTo] = learningRate * RECUR_START * nodeDeltas[t][layer][nodeTo];
					// }
					// else
					{
						rWeightDeltas[t][rLayer][rNode][layer-1][nodeTo] = learningRate * out[t+1][rLayer][rNode] * nodeDeltas[t][layer][nodeTo];
						
					}
					if (useMomentumTerm)
					{
						rWeightDeltas[t][rLayer][rNode][layer-1][nodeTo] += momentumTerm * previousRWeightDeltas[t][rLayer][rNode][layer-1][nodeTo];
					}
					
					
					(*rWeights)[rLayer][rNode][layer-1][nodeTo] += rWeightDeltas[t][rLayer][rNode][layer-1][nodeTo];
					
					debug_sumDeltas+=rWeightDeltas[t][rLayer][rNode][layer-1][nodeTo];
				}
			}
		}
		

			
		
		
	}
	//cout << "sum of deltas= " << debug_sumDeltas << endl;
}

void NNetwork::setWeights()
{
	//setup weights
	for (int layer = LAST_LAYER; layer > 0; layer--)
	{
		for (int nodeTo = 0; nodeTo < numOfNodesInLayer[layer]; nodeTo++)
		{
			
			
			
			for (int nodeFrom = 0; nodeFrom < numOfNodesInLayer[layer-1]; nodeFrom++)
			{
				
				weights[layer-1][nodeFrom][nodeTo] =oldWeights[layer-1][nodeFrom][nodeTo];
			}
			
			//bias
			bWeights[layer-1][nodeTo] =oldBWeights[layer-1][nodeTo];
			
			//recur
			if ((*recurMap)[layer-1][nodeTo].size() > 0)
			{
				for (tuple<int,int> recurConn: (*recurMap)[layer-1][nodeTo])
				{
					int rLayer=get<0>(recurConn);
					int rNode=get<1>(recurConn);
					
					
					(*rWeights)[rLayer][rNode][layer-1][nodeTo] = (*oldRWeights)[rLayer][rNode][layer-1][nodeTo];
				}
			}
		}
		

			
		
		
	}
}


void NNetwork::BPTT(int targetActivationNode)
{
	
	//resetNodeDeltas();
	
	
	for (int t=0; t<=k; t++)
	{
		backpropOutLayer(t,targetActivationNode);
		
		backpropHiddenLayers(t);
	}
	
	
	
}

void NNetwork::swapOuts()
{
	double** temp = out[k+1];
	
	for (int t = k+1; t>=1; t--)
	{
		out[t] = out[t-1];
	}
	out[0] = temp;
}

void NNetwork::printNetwork()
{
	//return;
	cout << "FF wieghts:" << endl;
	for (int layer=0; layer<=numOfHiddenLayers; layer++)
	{
		cout << "Starting layer " << layer << endl;
		for (int nodeFrom=0; nodeFrom < numOfNodesInLayer[layer]; nodeFrom++)
		{
			
			for (int nodeTo=0; nodeTo < numOfNodesInLayer[layer+1]; nodeTo++)
			{
				cout << "\tfrom node " << nodeFrom << " to node " << nodeTo << ": "<< weights[layer][nodeFrom][nodeTo] << endl;
			}
		}
	}
	
	cout << "Bais weights:" << endl;
	for (int layer=0; layer<=numOfHiddenLayers; layer++)
	{
		cout << "layer " << layer+1;
		for (int nodeTo=0; nodeTo < numOfNodesInLayer[layer+1]; nodeTo++)
		{
			cout << "\tto node " << nodeTo << ": " << bWeights[layer][nodeTo] << endl;
		}
	}
	
	cout << "Recurrent wieghts:" << endl;
	for (int layer=0; layer<=numOfHiddenLayers; layer++)
	{
			
		for (int nodeTo=0; nodeTo < numOfNodesInLayer[layer+1]; nodeTo++)
		{
			for (auto recur : (*recurMap)[layer][nodeTo])
			{
				double store = (*rWeights)[get<0>(recur)][get<1>(recur)][layer][nodeTo];
				cout << "\tfrom layer "<<get<0>(recur)<< " node " << get<1>(recur) << " to layer " << layer+1 << " node " << nodeTo << ": "<< store << endl;
			}
		}
		
	}
	
	cout << "Weight deltas:"<<endl;
	for (int t=0; t<=k; t++)
	{
		for (int layer=0; layer<=numOfHiddenLayers; layer++)
		{
			cout << "t" << t << " Starting layer " << layer << endl;
			for (int nodeFrom=0; nodeFrom < numOfNodesInLayer[layer]; nodeFrom++)
			{
			
				for (int nodeTo=0; nodeTo < numOfNodesInLayer[layer+1]; nodeTo++)
				{
					cout << "\tfrom node " << nodeFrom << " to node " << nodeTo << ": "<< previousWeightDeltas[t][layer][nodeFrom][nodeTo] << endl;
				}
			}
		}
	}
	
	cout << "Node net:"<<endl;
	for (int t=0; t<=k; t++)
	{
		for (int layer=1; layer<=LAST_LAYER; layer++)
		{
			cout << "t" << t << " Starting layer " << layer << endl;
			for (int nodeFrom=0; nodeFrom < numOfNodesInLayer[layer]; nodeFrom++)
			{
				cout << "\tat node " << nodeFrom << ": "<< nodeDeltas[t][layer][nodeFrom] << endl;
			
			}
		}
	}
	
	printOut();
}

void NNetwork::printOut()
{
	for (int t=0; t<=k+1; t++)
	{
		cout << "Out [" <<t<<"]:" << endl;
		for (int layer=0; layer<=numOfHiddenLayers+1; layer++)
		{
			cout << "Layer " << layer << endl;
			for (int node=0; node < numOfNodesInLayer[layer]; node++)
			{
				cout << "\tnode " << node << ": " <<out[t][layer][node] << endl;
			}
		}
	}
}


void NNetwork::saveNoRecur(string fileName)
{
	ofstream out;
	out.open(fileName, ios::out);
	
	
	out << numOfHiddenLayers <<endl;
	
	//numOfNodesInLayer
	for (int i=0; i<numOfHiddenLayers+2; i++)
		out << numOfNodesInLayer[i] << endl;
	//this->recurMap = recurMap;
	out << momentumTerm << endl;
	out << learningRate  << endl;
	out << k  << endl;
	
	//weights
	for (int i = 0; i < numOfHiddenLayers+1; i++)
	{
		int firstLayer = numOfNodesInLayer[i];
		int secondLayer = numOfNodesInLayer[i+1];
		for (int layer = 0; layer < firstLayer; layer++)
		{
			for (int layer2 = 0; layer2 < secondLayer; layer2++)
			{
				out << weights[i][layer][layer2] << endl;
			}
		}
	}
	
	//bWeights
	for (int i=0; i<numOfHiddenLayers+1; i++)
	{
		for (int node = 0; node < numOfNodesInLayer[i+1]; node++)
		{
			out << bWeights[i][node] << endl;
		}
	}
	
	//rWeights = new map<int,map<int,map<int,map<int,double> > > >;
	 
	
	
	out.close();

}
