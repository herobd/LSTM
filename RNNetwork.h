//Recurrent Nueral Network
//for CS 678 lab
//Brian Davis

#ifndef NNET_H
#define NNET_H

#include <vector>
#include <map>
#include <tuple>
#include <fstream>
using namespace std;




#define TRACE 0

#if TRACE
#define BIAS_OUT 0
#define RECUR_START 0
#else
#define BIAS_OUT 1
#define RECUR_START .5
#endif

class NNetwork
{
	private:
	double*** weights;//[layer from][node from][node to]
	double** bWeights;//[layer prev][nodeto]
	double**** previousWeightDeltas;//[t][layer from][node from][node to]
	double**** weightDeltas;//[t][layer from][node from][node to]
	double*** previousBWeightDeltas;//[t][layer prev][noteto]
	double*** bWeightDeltas;//[t][layer prev][noteto]
	int numOfHiddenLayers ;
	bool useMomentumTerm;
	int* numOfNodesInLayer;
	map<int, map<int, vector<tuple<int,int> > > >* recurMap;//vector of <layer,nodefrom> with recurrent connection to *[layer-1][nodeto]
	map<int,map<int,map<int,map<int,double> > > >* rWeights; //*[rLayer][rNode][layer-1][nodeTo]
	map<int,map<int,map<int,map<int,double> > > >* previousRWeightDeltas; //[t][rLayer][rNode][layer-1][nodeTo]
	map<int,map<int,map<int,map<int,double> > > >* rWeightDeltas; //[t][rLayer][rNode][layer-1][nodeTo]
	double momentumTerm;
	double learningRate;
	int k; //actually k-1
	
	//Holding arrays. These are reused frequently
	double*** out;//[t][layer][node]
	double*** net;//[t][layer][node]
	double*** nodeDeltas;//[t][layer][node]
	double*** oldWeights;//[layer from][node from][node to]
	double** oldBWeights;//[layer from][node to]
	map<int,map<int,map<int,map<int,double> > > >* oldRWeights; //*[rLayer][rNode][layer-1][nodeTo]
	
	
	//randomize weights
	void initRandomWeights();
	double activationFunction(double net) const;
	//double afPrime(double net) const;//derevitive of activation function
	void run();
	void copyWeightsTo();
	void backpropOutLayer(int t,int targetActivationNode);
	void backpropHiddenLayers(int t);
	
	void swapOuts();
	void BPTT(int targetActivationNode);
	void setWeights();
	
	public:
	NNetwork(int numOfHiddenLayers, int* numOfNodesInLayer, map<int, map<int, vector<tuple<int,int> > > >* recurMap, bool useMomentumTerm, double momentumTerm, double learningRate, int k);
	NNetwork(string fileName);
	~NNetwork();
	
	vector<double> runOn(const vector<double> &input);
	vector<double> trainOn(const vector<double> &testInstance, int targetActivationNode);
	double* trainOnRetD(const vector<double> &testInstance, int targetActivationNode);
	void burnInOn(const vector<vector<double> > &input);//where there are k (k-1) input examples 
	void printNetwork();
	void printOut();
	
	void presetWeights();
	
	void saveNoRecur(string fileName);
};

#endif
