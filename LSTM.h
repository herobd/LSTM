//LSTM Recurrent Nueral Network
//for CS 678 project
//Brian Davis

#ifndef LSTM_H
#define LSTM_H

#include <vector>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <random>
using namespace std;


#define t 1
#define TIME_SPAN 2


class LSTM
{
	public:
	LSTM(int numOfInputNodes, int numOfOutputNodes, int numOfBlocks, int numOfCellsInBlock, int numOfHiddenNodes,
		double learningRate, double momentumTerm);
	LSTM(string fileName);
	
	//Test on a test dataset
	double test(const vector<vector<double>*> &instances);
	
	//runs an instance through the network
	void runOn(vector<double>*instance);
	
	//train on a dataset
	void train(const vector<vector<double>*> &instances, const vector<vector<double>*> &testInstances, int maxIter, ofstream &outfile, string saveHere);
	
	//runs an instance through and updates wieghts
	double trainOn(vector<double>*instance, int correctActivationNode);
	
	//Burning in involves running a few samples to simulate already being in a run of temporal data
	void burnInOn(vector<vector<double>* > instances);
	
	
	void save(string fileName);



	private:
	double learningRate;
	double momentumTerm;
	int numOfInputNodes;
	int numOfOutputNodes;
	int numOfBlocks;
	int numOfCellsInBlock;
	int numOfHiddenNodes;
      
      //These are the weights between various nodes
      //naming convention: weight[from type][to type]
      //IN, OUT and FOR are the INPUT, OUTPUT and FORGET gates of the LSTM blocks
      //vectors provide bounds checking, which helped with debugging, obviously arrays would be more efficient
           
	vector< vector< vector<double> > > weightCellOutput;//[block][cell][n]
	vector< vector< vector<double> > > weightCellHidden;//[block][cell][n]
	vector< vector< vector<double> > > weightInputCell;//[n][block][cell]
	vector< vector< vector< vector<double> > > > weightCellCell;//[block][cell][block][cell]
	vector< vector< vector<double> > > weightHiddenCell;//[n][block][cell]
	vector< vector<double> > weightInputIN;//[n][block]
	vector< vector<double> > weightHiddenIN;//[n][block]
	vector< vector< vector<double> > > weightCellIN;//[block][cell][block]
	vector< vector<double> > weightInputFOR;//[n][block]
	vector< vector<double> > weightHiddenFOR;//[n][block]
	vector< vector< vector<double> > > weightCellFOR;//[block][cell][block]
	vector< vector<double> > weightHiddenOutput;//[n][n]
	vector< vector<double> > weightHiddenHidden;//[n][n]
	vector< vector< vector<double> > >  weightCellOUT;//[block][cell][block]
	vector< vector<double> > weightInputOUT;//[n][block]
	vector< vector<double> > weightHiddenOUT;//[n][block]
	vector< vector<double> > weightInputHidden;//[n][n]

	vector< vector<double> > weightBiasCell;//[block][cell]
	vector<double> weightBiasIN;//[block]
	vector<double> weightBiasOUT;//[block]
	vector<double> weightBiasFOR;//[block]
	vector<double> weightBiasHidden;//[n]
	vector<double> weightBiasOutput;//[n]
	
	//[t] is time, either present t=1, or past t=0
	
	//error
	vector<double> dOutput;//[n]
	vector<double> dInput;//[n]
	vector< vector<double> > dHidden;//[t][n]
	vector<double> dOUT;//[block]

	//output, what nodes emit
	vector< vector<double> > outOutput;//[t][n]
	vector< vector<double> > outHidden;//[t][n]
	vector< vector<double> > outInput;//[t][n]
	vector< vector< vector<double> > > outCell;//[t][block][cell]
	vector< vector<double> > outFOR;//[t][block]
	vector< vector<double> > outIN;//[t][block]
	vector< vector<double> > outOUT;//[t][block]

	//backprop error
	vector< vector< vector<double> > > deltaCellOutput;//[block][cell][n]
	vector< vector<double> > deltaHiddenOutput;//[n][n]
	vector< vector< vector<double> > > deltaInputCell;//[n][block][cell]
	vector< vector< vector< vector<double> > > > deltaCellCell;//[block][cell][block][cell]
	vector< vector< vector<double> > > deltaHiddenCell;//[n][block][cell]
	vector< vector<double> > deltaInputIN;//[n][block]
	vector< vector<double> > deltaHiddenIN;//[n][block]
	vector< vector< vector<double> > > deltaCellIN;//[block][cell][block]
	vector< vector<double> > deltaInputFOR;//[n][block]
	vector< vector<double> > deltaHiddenFOR;//[n][block]
	vector< vector< vector<double> > > deltaCellFOR;//[block][cell][block]
	vector< vector< vector<double> > > deltaCellHidden;//[block][cell][n]
	vector< vector<double> > deltaHiddenHidden;//[n][n]
	vector< vector<double> > deltaInputHidden;//[n][n]
	vector< vector< vector<double> > > deltaCellOUT;//[block][cell][block]
	vector< vector<double> > deltaHiddenOUT;//[n][block]
	vector< vector<double> > deltaInputOUT;//[n][block]
	
	vector<double> deltaBiasOutput;//[n]
	vector< vector<double> > deltaBiasCell;//[block][cell]
	vector<double> deltaBiasIN;//[bl]
	vector<double> deltaBiasFOR;//[blck]
	vector<double> deltaBiasHidden;//[n]
	vector<double> deltaBiasOUT;//[blc]
	
	vector< vector< vector< vector<double> > > > derInputCellIn;//[t][n][block][cell]
	vector< vector< vector< vector< vector<double> > > > > derCellOutCellIn;//[t][block][cell][block][cell]
	vector< vector< vector< vector<double> > > > derHiddenCellIn;//[t][n][block][cell]
	vector< vector< vector< vector<double> > > > derInputIN;//[t][n][block][cell]
	vector< vector< vector< vector<double> > > > derHiddenIN;//[t][n][block][cell]
	vector< vector< vector< vector< vector<double> > > > > derCellOutIN;//[t][block][cell][block][cell]
	vector< vector< vector< vector<double> > > > derInputFOR;//[t][n][block][cell]
	vector< vector< vector< vector<double> > > > derHiddenFOR;//[t][n][block][cell]
	vector< vector< vector< vector< vector<double> > > > > derCellOutFOR;//[t][block][cell][block][cell]
	
	vector< vector< vector<double> > > derBiasCellIn;//[t][block][cell]
	vector< vector< vector<double> > > derBiasIN;//[t][block][cell]
	vector< vector< vector<double> > > derBiasFOR;//[t][block][cell]
	
	vector< vector<double> > errorS;//[block][cell]
	vector< vector< vector<double> > > stateCell;//[t][block][cell]
	vector< vector<double> > netCell;//[block][cell]

	//activation funcctions in LSTM blocks (names derived from paper)
	double f (double x){return 1.0 / (1.0 + exp(-x));}
	double fp(double x){return x*(1.0-x);}

	double h (double x){return (2.0 / (1.0 + exp(-x)))-1.0;}
	double hpp(double x){return 2*f(x)*(1.0-f(x));}

	double g (double x){return (4.0 / (1.0 + exp(-x)))-2.0;}
	double gpp(double x){return 4*f(x)*(1.0-f(x));}
	
	void updateWeights();
	double totalWeightChange();
	double errorDifSqr(const vector<double> &result, unsigned int correctNode);
	bool isRight(const vector<double> &result, unsigned int correctNode);
	
	//swap present data to past and past to present to be overwritten
	void shiftTime();
	
	//for debugging purposes
	void checkWeightChange();
	
	
	void init();
	
	
};

#endif
