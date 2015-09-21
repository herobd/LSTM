
#include "LSTM.h"

LSTM::LSTM(int numOfInputNodes, int numOfOutputNodes, int numOfBlocks, int numOfCellsInBlock, int numOfHiddenNodes,
		double learningRate, double momentumTerm)
{
	this->learningRate=learningRate;
	this->momentumTerm=momentumTerm;
	this->numOfInputNodes=numOfInputNodes;
	this->numOfOutputNodes=numOfOutputNodes;
	this->numOfBlocks=numOfBlocks;
	this->numOfCellsInBlock=numOfCellsInBlock;
	this->numOfHiddenNodes=numOfHiddenNodes;
	
	init();
	
}

LSTM::LSTM(string fileName)
{
	ifstream in;
	in.open(fileName);
	assert(in.is_open());
	string line;
	getline (in,line);
	learningRate= stod(line);
	getline (in,line);
	momentumTerm= stod(line);
	getline (in,line);
	numOfInputNodes= stoi(line);
	getline (in,line);
	numOfOutputNodes= stoi(line);
	getline (in,line);
	numOfBlocks= stoi(line);
	getline (in,line);
	numOfCellsInBlock= stoi(line);
	getline (in,line);
	numOfHiddenNodes= stoi(line);
	
	init();
	
	for (int block=0; block<numOfBlocks; block++)
	{

		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
		{
			for (int n=0; n<numOfOutputNodes; n++)
			{
				getline (in,line);
				weightCellOutput[block][cell][n] = stod(line);// deltaCellOutput[block][cell][n];
			}
				
			for (int n=0; n<numOfHiddenNodes; n++)
			{
				getline (in,line);
				weightCellHidden[block][cell][n] = stod(line);// deltaCellHidden[block][cell][n];
			}
			
			for (int block2=0; block2<numOfBlocks; block2++)
			{
				getline (in,line);
				weightCellIN[block][cell][block2] = stod(line);// deltaCellIN[block][cell][block2];
				getline (in,line);
				weightCellOUT[block][cell][block2] = stod(line);// deltaCellOUT[block][cell][block2];
				getline (in,line);
				weightCellFOR[block][cell][block2] = stod(line);// deltaFOR[block][cell][block2];
				
				for (int cell2=0; cell2<numOfCellsInBlock; cell2++)
				{
					getline (in,line);
					weightCellCell[block][cell][block2][cell2] = stod(line);// deltaCellCell[block][cell][block2][cell2];
				}
			}
		}
	}
	
	for (int n=0; n<numOfInputNodes; n++)
	{

		for (int block=0; block<numOfBlocks; block++)
		{

			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				getline (in,line);
				weightInputCell[n][block][cell] = stod(line);// deltaInputCell[n][block][cell];
			}
			getline (in,line);
			weightInputIN[n][block] = stod(line);// deltaInputIN[n][block];
			getline (in,line);
			weightInputFOR[n][block] = stod(line);// deltaInputFOR[n][block];
			getline (in,line);
			weightInputOUT[n][block] = stod(line);// deltaInputOUT[n][block];
		}
		
		for (int n2=0; n2<numOfHiddenNodes; n2++)
		{
			getline (in,line);
			weightInputHidden[n][n2] = stod(line);// deltaInputHidden[n][n2];
			getline (in,line);
			deltaInputHidden[n][n2] = stod(line);// deltaInputHidden[n][n2] ;
		}
	}
	
	for (int n=0; n<numOfHiddenNodes; n++)
	{
		
		for (int n2=0; n2<numOfOutputNodes; n2++)
		{
			getline (in,line);
			weightHiddenOutput[n][n2] = stod(line);// deltaHiddenOutput[n][n2];
		}
		for (int n2=0; n2<numOfInputNodes; n2++)
		{
			getline (in,line);
			weightHiddenHidden[n][n2] = stod(line);// deltaHiddenHidden[n][n2];
		}
		
		for (int block=0; block<numOfBlocks; block++)
		{
			getline (in,line);
			weightHiddenFOR[n][block] = stod(line);// deltaHiddenFOR[n][block];
			getline (in,line);
			weightHiddenIN[n][block] = stod(line);// deltaHiddenIN[n][block];
			getline (in,line);
			weightHiddenOUT[n][block] = stod(line);// deltaHiddenOUT[n][block];
			
			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				getline (in,line);
				weightHiddenCell[n][block][cell] = stod(line);// deltaHiddenCell[n][block][cell];
			}
		}
	}
	
	for (int block=0; block<numOfBlocks; block++)
	{
		getline (in,line);
			weightBiasIN[block] = stod(line);// deltaBiasIN[block];
		getline (in,line);
			weightBiasOUT[block] = stod(line);// deltaBiasOUT[block];
		getline (in,line);
			weightBiasFOR[block] = stod(line);// deltaBiasFOR[block] ;
		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
		{
			getline (in,line);
			weightBiasCell[block][cell] = stod(line);// deltaBiasCell[block][cell];
		}
	}
	
	for (int n=0; n<numOfHiddenNodes; n++)
	{
		getline (in,line);
		weightBiasHidden[n] = stod(line);// deltaBiasHidden[n];
	}
	for (int n=0; n<numOfOutputNodes; n++)
	{
		getline (in,line);
		weightBiasOutput[n] = stod(line);
	}
	
	in.close();
}

void LSTM::burnInOn(vector<vector<double>* > instances)
{
	for (int i=0; i<numOfInputNodes; i++)
	{
		outInput[t][i]=instances[0]->at(i);
	}
	for (int index=1; index<instances.size(); index++)
	{
		runOn(instances[index]);
	}
}


void LSTM::init()
{
	default_random_engine generator;
	normal_distribution<double> distribution_0(0.0,0.15);
	normal_distribution<double> distribution_p(0.15,0.1);
	normal_distribution<double> distribution_n(-0.15,0.1);
	
	weightCellOutput.resize(numOfBlocks);//[block][cell][n]
	weightCellHidden.resize(numOfBlocks);//[block][cell][n]
	weightCellCell.resize(numOfBlocks);//[block][cell][block][cell]
	weightCellIN.resize(numOfBlocks);//[block][cell][block]
	weightCellOUT.resize(numOfBlocks);//[block][cell][block]
	weightCellFOR.resize(numOfBlocks);//[block][cell][block]
	
	deltaCellOutput.resize(numOfBlocks);//[block][cell][n]
	deltaCellHidden.resize(numOfBlocks);//[block][cell][n]
	deltaCellCell.resize(numOfBlocks);//[block][cell][block][cell]
	deltaCellIN.resize(numOfBlocks);//[block][cell][block]
	deltaCellOUT.resize(numOfBlocks);//[block][cell][block]
	deltaCellFOR.resize(numOfBlocks);//[block][cell][block]
	
	
	for (int block=0; block<numOfBlocks; block++)
	{
		weightCellOutput[block].resize(numOfCellsInBlock);//[cell][n]
		weightCellHidden[block].resize(numOfCellsInBlock);//[cell][n]
		weightCellCell[block].resize(numOfCellsInBlock);//[cell][block][cell]
		weightCellIN[block].resize(numOfCellsInBlock);//[cell][block]
		weightCellOUT[block].resize(numOfCellsInBlock);//[cell][block]
		weightCellFOR[block].resize(numOfCellsInBlock);//[cell][block]
		
		deltaCellOutput[block].resize(numOfCellsInBlock);//[cell][n]
		deltaCellHidden[block].resize(numOfCellsInBlock);//[cell][n]
		deltaCellCell[block].resize(numOfCellsInBlock);//[cell][block][cell]
		deltaCellIN[block].resize(numOfCellsInBlock);//[cell][block]
		deltaCellOUT[block].resize(numOfCellsInBlock);//[cell][block]
		deltaCellFOR[block].resize(numOfCellsInBlock);//[cell][block]
		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
		{
			weightCellOutput[block][cell].resize(numOfOutputNodes);//[n]
			deltaCellOutput[block][cell].resize(numOfOutputNodes);//[n]
			for (int n=0; n<numOfOutputNodes; n++)
				weightCellOutput[block][cell][n] = distribution_0(generator);
				
			weightCellHidden[block][cell].resize(numOfHiddenNodes);//[n]
			deltaCellHidden[block][cell].resize(numOfHiddenNodes);//[n]
			for (int n=0; n<numOfHiddenNodes; n++)
				weightCellHidden[block][cell][n] = distribution_0(generator);
			
			weightCellCell[block][cell].resize(numOfBlocks);//[block][cell]
			weightCellIN[block][cell].resize(numOfBlocks);//[block]
			weightCellOUT[block][cell].resize(numOfBlocks);//[block]
			weightCellFOR[block][cell].resize(numOfBlocks);//[block]
			
			deltaCellCell[block][cell].resize(numOfBlocks);//[block][cell]
			deltaCellIN[block][cell].resize(numOfBlocks);//[block]
			deltaCellOUT[block][cell].resize(numOfBlocks);//[block]
			deltaCellFOR[block][cell].resize(numOfBlocks);//[block]
			for (int block2=0; block2<numOfBlocks; block2++)
			{
				weightCellIN[block][cell][block2] = distribution_0(generator);
				weightCellOUT[block][cell][block2] = distribution_0(generator);
				weightCellFOR[block][cell][block2] = distribution_0(generator);
				
				weightCellCell[block][cell][block2].resize(numOfCellsInBlock);//[cell]
				deltaCellCell[block][cell][block2].resize(numOfCellsInBlock);//[cell]
				for (int cell2=0; cell2<numOfCellsInBlock; cell2++)
				{
					weightCellCell[block][cell][block2][cell2] = distribution_0(generator);
				}
			}
		}
	}
	
	weightInputCell.resize(numOfInputNodes);//[n][block][cell]
	weightInputIN.resize(numOfInputNodes);//[n][block]
	weightInputFOR.resize(numOfInputNodes);//[n][block]
	weightInputOUT.resize(numOfInputNodes);//[n][block]
	weightInputHidden.resize(numOfInputNodes);//[n][n]
	
	deltaInputCell.resize(numOfInputNodes);//[n][block][cell]
	deltaInputIN.resize(numOfInputNodes);//[n][block]
	deltaInputFOR.resize(numOfInputNodes);//[n][block]
	deltaInputOUT.resize(numOfInputNodes);//[n][block]
	deltaInputHidden.resize(numOfInputNodes);//[n][n]
	for (int n=0; n<numOfInputNodes; n++)
	{
		weightInputCell[n].resize(numOfBlocks);//[block][cell]
		weightInputIN[n].resize(numOfBlocks);//[block]
		weightInputFOR[n].resize(numOfBlocks);//[block]
		weightInputOUT[n].resize(numOfBlocks);//[block]
		
		deltaInputCell[n].resize(numOfBlocks);//[block][cell]
		deltaInputIN[n].resize(numOfBlocks);//[block]
		deltaInputFOR[n].resize(numOfBlocks);//[block]
		deltaInputOUT[n].resize(numOfBlocks);//[block]
		for (int block=0; block<numOfBlocks; block++)
		{
			weightInputCell[n][block].resize(numOfCellsInBlock);//[cell]
			deltaInputCell[n][block].resize(numOfCellsInBlock);//[cell]
			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				weightInputCell[n][block][cell] = distribution_0(generator);
			}
			weightInputIN[n][block] = distribution_0(generator);
			weightInputFOR[n][block] = distribution_0(generator);
			weightInputOUT[n][block] = distribution_0(generator);
		}
		
		weightInputHidden[n].resize(numOfHiddenNodes);//[n]
		deltaInputHidden[n].resize(numOfHiddenNodes);//[n]
		for (int n2=0; n2<numOfHiddenNodes; n2++)
		{
			weightInputHidden[n][n2] = distribution_0(generator);
			deltaInputHidden[n][n2] = distribution_0(generator);
		}
	}
	
	weightHiddenCell.resize(numOfHiddenNodes);//[n][block][cell]
	weightHiddenOutput.resize(numOfHiddenNodes);//[n][n]
	weightHiddenHidden.resize(numOfHiddenNodes);//[n][n]
	weightHiddenFOR.resize(numOfHiddenNodes);//[n][block]
	weightHiddenIN.resize(numOfHiddenNodes);//[n][block]
	weightHiddenOUT.resize(numOfHiddenNodes);//[n][block]
	
	deltaHiddenCell.resize(numOfHiddenNodes);//[n][block][cell]
	deltaHiddenOutput.resize(numOfHiddenNodes);//[n][n]
	deltaHiddenHidden.resize(numOfHiddenNodes);//[n][n]
	deltaHiddenFOR.resize(numOfHiddenNodes);//[n][block]
	deltaHiddenIN.resize(numOfHiddenNodes);//[n][block]
	deltaHiddenOUT.resize(numOfHiddenNodes);//[n][block]
	for (int n=0; n<numOfHiddenNodes; n++)
	{
		
		weightHiddenOutput[n].resize(numOfOutputNodes);//[n]
		deltaHiddenOutput[n].resize(numOfOutputNodes);//[n]
		for (int n2=0; n2<numOfOutputNodes; n2++)
			weightHiddenOutput[n][n2] = distribution_0(generator);
		weightHiddenHidden[n].resize(numOfHiddenNodes);//[n]
		deltaHiddenHidden[n].resize(numOfHiddenNodes);//[n]
		for (int n2=0; n2<numOfInputNodes; n2++)
			weightHiddenHidden[n][n2] = distribution_0(generator);
		
		weightHiddenCell[n].resize(numOfBlocks);//[block][cell]
		weightHiddenFOR[n].resize(numOfBlocks);//[block]
		weightHiddenIN[n].resize(numOfBlocks);//[block]
		weightHiddenOUT[n].resize(numOfBlocks);//[block]
		
		deltaHiddenCell[n].resize(numOfBlocks);//[block][cell]
		deltaHiddenFOR[n].resize(numOfBlocks);//[block]
		deltaHiddenIN[n].resize(numOfBlocks);//[block]
		deltaHiddenOUT[n].resize(numOfBlocks);//[block]
		for (int block=0; block<numOfBlocks; block++)
		{
			weightHiddenFOR[n][block] = distribution_0(generator);
			weightHiddenIN[n][block] = distribution_0(generator);
			weightHiddenOUT[n][block] = distribution_0(generator);
			
			weightHiddenCell[n][block].resize(numOfCellsInBlock);//[cell]
			deltaHiddenCell[n][block].resize(numOfCellsInBlock);//[cell]
			for (int cell=0; cell<numOfCellsInBlock; cell++)
				weightHiddenCell[n][block][cell] = distribution_0(generator);
		}
	}
	//cout << "whc[0][0] size: " << weightHiddenCell[0][0].size() <<endl;
	
	weightBiasCell.resize(numOfBlocks);//[block][cell]
	weightBiasIN.resize(numOfBlocks);//[block]
	weightBiasOUT.resize(numOfBlocks);//[block]
	weightBiasFOR.resize(numOfBlocks);//[block]
	
	deltaBiasCell.resize(numOfBlocks);//[block][cell]
	deltaBiasIN.resize(numOfBlocks);//[block]
	deltaBiasOUT.resize(numOfBlocks);//[block]
	deltaBiasFOR.resize(numOfBlocks);//[block]
	for (int block=0; block<numOfBlocks; block++)
	{
		weightBiasIN[block] = distribution_n(generator);
		weightBiasOUT[block] = distribution_n(generator);
		weightBiasFOR[block] = distribution_p(generator);
		
		weightBiasCell[block].resize(numOfCellsInBlock);//[cell]
		deltaBiasCell[block].resize(numOfCellsInBlock);//[cell]
		for (int cell=0; cell<numOfCellsInBlock; cell++)
			weightBiasCell[block][cell] = distribution_0(generator);
	}
	
	weightBiasHidden.resize(numOfHiddenNodes);//[n]
	deltaBiasHidden.resize(numOfHiddenNodes);//[n]
	for (int n=0; n<numOfHiddenNodes; n++)
		weightBiasHidden[n] = distribution_0(generator);
	weightBiasOutput.resize(numOfOutputNodes);//[n]
	deltaBiasOutput.resize(numOfOutputNodes);//[n]
	for (int n=0; n<numOfOutputNodes; n++)
		weightBiasOutput[n] = distribution_0(generator);
	
	dOutput.resize(numOfOutputNodes);//[n]
	dInput.resize(numOfOutputNodes);//[n]
	dHidden.resize(TIME_SPAN);//[t][n]
	dHidden[0].resize(numOfHiddenNodes);//[n]
	dHidden[1].resize(numOfHiddenNodes);//[n]
	dOUT.resize(numOfBlocks);//[block]

	outOutput.resize(TIME_SPAN);//[t][n]
	outHidden.resize(TIME_SPAN);//[t][n]
	outInput.resize(TIME_SPAN);//[t][n]
	outCell.resize(TIME_SPAN);//[t][block][cell]
	outFOR.resize(TIME_SPAN);//[t][block]
	outIN.resize(TIME_SPAN);//[t][block]
	outOUT.resize(TIME_SPAN);//[t][block]
	outCell.resize(TIME_SPAN);//[t][block][cell]
	
	for (int tt=0; tt<TIME_SPAN; tt++)
	{
		outOutput[tt].resize(numOfOutputNodes);//[n]
		for (int n=0; n<numOfOutputNodes; n++)
			outOutput[tt][n] = distribution_0(generator);
			
		outHidden[tt].resize(numOfHiddenNodes);//[t][n]
		for (int n=0; n<numOfHiddenNodes; n++)
			outHidden[tt][n] = distribution_0(generator);
			
		outInput[tt].resize(numOfInputNodes);//[t][n]
		for (int n=0; n<numOfInputNodes; n++)
			outInput[tt][n] = distribution_0(generator);
		
		outCell[tt].resize(numOfBlocks);//[t][block][cell]
		outFOR[tt].resize(numOfBlocks);//[t][block]
		outIN[tt].resize(numOfBlocks);//[t][block]
		outOUT[tt].resize(numOfBlocks);//[t][block]
		
		for (int block=0; block<numOfBlocks; block++)
		{
			outFOR[tt][block] = distribution_0(generator);
			outIN[tt][block] = distribution_0(generator);
			outOUT[tt][block] = distribution_0(generator);
			
			outCell[tt][block].resize(numOfCellsInBlock);
			for (int cell=0; cell<numOfCellsInBlock; cell++)
				outCell[tt][block][cell] = distribution_0(generator);
		}
	}
	
	derInputCellIn.resize(TIME_SPAN);//[t][n][block][cell]
	derCellOutCellIn.resize(TIME_SPAN);//[t][block][cell][block][cell]
	derHiddenCellIn.resize(TIME_SPAN);//[t][n][block][cell]
	derInputIN.resize(TIME_SPAN);//[t][n][block][cell]
	derHiddenIN.resize(TIME_SPAN);//[t][n][block][cell]
	derCellOutIN.resize(TIME_SPAN);//[t][block][cell][block][cell]
	derInputFOR.resize(TIME_SPAN);//[t][n][block][cell]
	derHiddenFOR.resize(TIME_SPAN);//[t][n][block][cell]
	derCellOutFOR.resize(TIME_SPAN);//[t][block][cell][block][cell]
	
	derBiasCellIn.resize(TIME_SPAN);//[t][block][cell]
	derBiasIN.resize(TIME_SPAN);//[t][block][cell]
	derBiasFOR.resize(TIME_SPAN);//[t][block][cell]
	
	stateCell.resize(TIME_SPAN);//[t][block][cell]
	
	for (int tt=0; tt<TIME_SPAN; tt++)
	{
		derInputCellIn[tt].resize(numOfInputNodes);//[t][n][block][cell]
		derInputFOR[tt].resize(numOfInputNodes);//[t][n][block][cell]
		derInputIN[tt].resize(numOfInputNodes);//[t][n][block][cell]
		for (int n=0; n<numOfInputNodes; n++)
		{
			derInputCellIn[tt][n].resize(numOfBlocks);//[t][n][block][cell]
			derInputFOR[tt][n].resize(numOfBlocks);//[t][n][block][cell]
			derInputIN[tt][n].resize(numOfBlocks);//[t][n][block][cell]
			
			for (int block=0; block<numOfBlocks; block++)
			{
				derInputCellIn[tt][n][block].resize(numOfCellsInBlock,0);//[t][n][block][cell]
				derInputFOR[tt][n][block].resize(numOfCellsInBlock,0);//[t][n][block][cell]
				derInputIN[tt][n][block].resize(numOfCellsInBlock,0);//[t][n][block][cell]
			}
		}
		
		derHiddenCellIn[tt].resize(numOfHiddenNodes);//[t][n][block][cell]
		derHiddenIN[tt].resize(numOfHiddenNodes);//[t][n][block][cell]
		derHiddenFOR[tt].resize(numOfHiddenNodes);//[t][n][block][cell]
		for (int n=0; n<numOfHiddenNodes; n++)
		{
			derHiddenCellIn[tt][n].resize(numOfBlocks);//[t][n][block][cell]
			derHiddenIN[tt][n].resize(numOfBlocks);//[t][n][block][cell]
			derHiddenFOR[tt][n].resize(numOfBlocks);//[t][n][block][cell]
			for (int block=0; block<numOfBlocks; block++)
			{
				derHiddenCellIn[tt][n][block].resize(numOfCellsInBlock,0);//[t][n][block][cell]
				derHiddenIN[tt][n][block].resize(numOfCellsInBlock,0);//[t][n][block][cell]
				derHiddenFOR[tt][n][block].resize(numOfCellsInBlock,0);//[t][n][block][cell]
			}
		}
		
		derCellOutFOR[tt].resize(numOfBlocks);//[t][block][cell][block][cell]
		derCellOutIN[tt].resize(numOfBlocks);//[t][block][cell][block][cell]
		derCellOutCellIn[tt].resize(numOfBlocks);//[t][block][cell][block][cell]
		derBiasCellIn[tt].resize(numOfBlocks);//[t][block][cell]
		
		derBiasIN[tt].resize(numOfBlocks);//[t][block][cell]
		derBiasFOR[tt].resize(numOfBlocks);//[t][block][cell]
	
		stateCell[tt].resize(numOfBlocks);//[t][block][cell]
		
		for (int block=0; block<numOfBlocks; block++)
		{
			derCellOutFOR[tt][block].resize(numOfCellsInBlock);//[t][block][cell][block][cell]
			derCellOutIN[tt][block].resize(numOfCellsInBlock);//[t][block][cell][block][cell]
			derCellOutCellIn[tt][block].resize(numOfCellsInBlock);//[t][block][cell][block][cell]
			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				derCellOutFOR[tt][block][cell].resize(numOfBlocks);//[t][block][cell][block][cell]
				derCellOutIN[tt][block][cell].resize(numOfBlocks);//[t][block][cell][block][cell]
				derCellOutCellIn[tt][block][cell].resize(numOfBlocks);//[t][block][cell][block][cell]
				for (int block2=0; block2<numOfBlocks; block2++)
				{
					derCellOutFOR[tt][block][cell][block2].resize(numOfCellsInBlock,0);//[t][block][cell][block][cell]
					derCellOutIN[tt][block][cell][block2].resize(numOfCellsInBlock,0);//[t][block][cell][block][cell]
					derCellOutCellIn[tt][block][cell][block2].resize(numOfCellsInBlock,0);//[t][block][cell][block][cell]
				}
			}
			
			derBiasCellIn[tt][block].resize(numOfCellsInBlock,0);//[t][block][cell]
		
			derBiasIN[tt][block].resize(numOfCellsInBlock,0);//[t][block][cell]
			derBiasFOR[tt][block].resize(numOfCellsInBlock,0);//[t][block][cell]
	
			stateCell[tt][block].resize(numOfCellsInBlock,0);//[t][block][cell]
		}
	}
	
	errorS.resize(numOfBlocks);//[block][cell]
	
	netCell.resize(numOfBlocks);//[block][cell]
	for (int block=0; block<numOfBlocks; block++)
	{
		errorS[block].resize(numOfCellsInBlock,0);//[block][cell]
		netCell[block].resize(numOfCellsInBlock,0);//[block][cell]
	}
}

double LSTM::errorDifSqr(const vector<double> &result, unsigned int correctNode)
{
	//cout << "Guess (" << (*result)[0] << "," << (*result)[1] << "," << (*result)[2] << ")" << endl;
	//cout << "Correct [" << correctNode << "]" << endl;
	double sum = 0;
	for (unsigned int i = 0; i < numOfOutputNodes; i++)
	{
		if (correctNode != i)
		{
			sum += pow((0 - (result)[i]),2);
		}
		else
		{
			sum += pow((1 - (result)[i]),2);
		}
	}
	//cout << "error is " << sum << endl;
	return sum;
}

void LSTM::train(const vector<vector<double>*> &instances, const vector<vector<double>*> &testInstances, int maxIter, ofstream &outfile, string saveHere)
{
	int epochSize = min((int)instances.size(),(int)10000);
	cout << "Epoch size: " << epochSize << endl;
	int runLength = 10;
	double criticalError = 0.005;
	
	int limit = 0;
	for (; limit < maxIter; limit++)
	{
		
		double error = 0;
		double weightChange=0;
		int k=2;
		for (int count = k; count < epochSize; count++)
		//for (int i = k-TRACE; i < numOfTrainingInstances; i++)
		{
		
	//cout << "2 whc[0][0] size: " << weightHiddenCell[0][0].size() <<endl;
			int i = (rand() % (instances.size() - (k+runLength))) + k;
			vector<vector<double>* > burnInstances;
			for (int j = i-(k); j < i; j++)
			{
				vector<double>* instance = instances[j];
				//int correctActivationNode = instance->back();
				burnInstances.push_back(instance);
			}
			burnInOn(burnInstances);
			
	//cout << "3 whc[0][0] size: " << weightHiddenCell[0][0].size() <<endl;
			
			for (int run=0; run<runLength; run++)
			{
				vector<double>* instance = instances[i+run];
				int correctActivationNode = instances[i+run-1]->back();
			
				//cout << "Instance " << i << "---" << endl;
				weightChange += trainOn(instance, correctActivationNode);
						
				error += errorDifSqr(outOutput[t], correctActivationNode);
				assert(!std::isnan(weightChange));
				assert(!std::isnan(error));
			}
		}
		cout << "Round " << limit << ", error: " << error << ", weight change: "<<weightChange<< endl;
		error /=epochSize;
		weightChange /=epochSize;
		outfile << "Round " << limit << ", error: " << error << ", weight change: "<<weightChange<< endl;
		
		
		if (limit%5==0)
		{
			int correctClassifications = 0;
	
			vector<vector<double>* > burnInstances;
			for (int j = k-k; j < k; j++)
			{
				vector<double>* instance = instances[j];
				//int correctActivationNode = instance->back();
				burnInstances.push_back(instance);
			}
			burnInOn(burnInstances);
	
			for (int i = k; i < instances.size(); i++)
			{
				vector<double>* instance = instances[i];
				int correctActivationNode = instance->back();
		
				runOn(instance);
				if (isRight(outOutput[t],correctActivationNode))
					correctClassifications++;
		
			}
			double acc =(double) correctClassifications / (double) instances.size();
			cout << "Round " << limit << ", Accuracy on training data: " << acc << endl;
			outfile << "\tAccuracy on training data: " << acc;
			
			if (error < (criticalError) && acc > .95)
			{
				cout << "Critical Error/Acc reached." << endl;
				outfile << "\tCritical Error/Acc reached.";
			
			
			
				break;
			}
		}
		
		if (limit%20==0)
		{
			double tacc = test(testInstances);
			cout << "Accuracy on TESTing data: " <<tacc<<endl;
			outfile << "Accuracy on TESTing data: " <<tacc<<endl;
		}
		
		
		outfile << endl;
		if (saveHere.length()>0)
			save(saveHere);
	}
	cout << "Training finished in "<< limit <<" iterations." <<  endl;
}

double LSTM::test(const vector<vector<double>*> &instances)
{
	int correct=0;
	
	vector<vector<double>* > burnInstances;
	int k=2;
	for (int j = k-k; j < k; j++)
	{
		vector<double>* instance = instances[j];
		//int correctActivationNode = instance->back();
		burnInstances.push_back(instance);
	}
	burnInOn(burnInstances);
	
	for (int i=k; i<instances.size(); i++)
	{
		vector<double>* instance = instances[i];
		runOn(instance);
		if (isRight(outOutput[t], instance->back()))
			correct++;
		
	}
	//cout << "its: " << correct << " / " << instances.size() << "-" <<k << endl;
	return ((double) correct)/((double) (instances.size()-k));
}


void LSTM::runOn(vector<double>*instance)
{
	shiftTime();
	for (int i=0; i<numOfInputNodes; i++)
	{
		outInput[t][i]=instance->at(i);
	}
	
	for (int j=0; j<numOfBlocks; j++)
	{
		double netOUT = 0;
		double netIN = 0;
		double netFOR = 0;
		
		for (int m=0; m<numOfInputNodes; m++)
		{
			netOUT += weightInputOUT[m][j]*outInput[t-1][m];
			netIN += weightInputIN[m][j]*outInput[t-1][m];
			netFOR += weightInputFOR[m][j]*outInput[t-1][m];
		}
		for (int m=0; m<numOfBlocks; m++)
		{
			for (int m_cell=0; m_cell<numOfCellsInBlock; m_cell++)
			{
				netOUT += weightCellOUT[m][m_cell][j]*outCell[t-1][m][m_cell];
				netIN += weightCellIN[m][m_cell][j]*outCell[t-1][m][m_cell];
				netFOR += weightCellFOR[m][m_cell][j]*outCell[t-1][m][m_cell];
			}
		}
		for (int m=0; m<numOfHiddenNodes; m++)
		{
			netOUT += weightHiddenOUT[m][j]*outHidden[t-1][m];
			netIN += weightHiddenIN[m][j]*outHidden[t-1][m];
			netFOR += weightHiddenFOR[m][j]*outHidden[t-1][m];
		}
		
		netOUT += weightBiasOUT[j];
		netIN += weightBiasIN[j];
		netFOR += weightBiasFOR[j];
		
		outOUT[t][j] = f(netOUT);
		outIN[t][j] = f(netIN);
		outFOR[t][j] = f(netFOR);
		
		for (int v=0; v<numOfCellsInBlock; v++)
		{
			//netCell is object saved, called in training
			for (int m=0; m<numOfInputNodes; m++)
			{
				netCell[j][v] += weightInputCell[m][j][v]*outInput[t-1][m];
			}
			for (int m=0; m<numOfBlocks; m++)
			{
				for (int m_cell=0; m_cell<numOfCellsInBlock; m_cell++)
				{
					netCell[j][v] += weightCellCell[m][m_cell][j][v]*outCell[t-1][m][m_cell];
				}
			}
			for (int m=0; m<numOfHiddenNodes; m++)
			{
				netCell[j][v] += weightHiddenCell[m][j][v]*outHidden[t-1][m];
			}
			netCell[j][v] += weightBiasCell[j][v];
		
			stateCell[t][j][v] = outFOR[t][j] * stateCell[t-1][j][v] + outIN[t][j] * g(netCell[j][v]);
			
			outCell[t][j][v] = outOUT[t][j] * h(stateCell[t][j][v]);
		}
	}
	
	for (int i=0; i<numOfHiddenNodes; i++)
	{
		double net = 0;
		for (int m=0; m<numOfInputNodes; m++)
		{
			net += weightInputHidden[m][i]*outInput[t-1][m];
		}
		for (int m=0; m<numOfBlocks; m++)
		{
			for (int m_cell=0; m_cell<numOfCellsInBlock; m_cell++)
			{
				net += weightCellHidden[m][m_cell][i]*outCell[t-1][m][m_cell];
			}
		}
		for (int m=0; m<numOfHiddenNodes; m++)
		{
			net += weightHiddenHidden[m][i]*outHidden[t-1][m];
		}
		net += weightBiasHidden[i];
		
		outHidden[t][i] = f(net);
	}
	
	for (int i=0; i<numOfOutputNodes; i++)
	{
		double net = 0;
		for (int m=0; m<numOfBlocks; m++)
		{
			for (int m_cell=0; m_cell<numOfCellsInBlock; m_cell++)
			{
				net += weightCellOutput[m][m_cell][i]*outCell[t][m][m_cell];
			}
		}
		for (int m=0; m<numOfHiddenNodes; m++)
		{
			net += weightHiddenOutput[m][i]*outHidden[t][m];
		}
		net += weightBiasOutput[i];
		
		outOutput[t][i] = f(net);
	}
}

double LSTM::trainOn(vector<double>*instance, int correctActivationNode)
{
//cout << "4 whc[0][0] size: " << weightHiddenCell[0][0].size() <<endl;
	runOn(instance);
//cout << "5 whc[0][0] size: " << weightHiddenCell[0][0].size() <<endl;
	//From hidden layer to output layer
	for (int i=0; i<numOfOutputNodes; i++)
	{
		dOutput[i] = (((i==correctActivationNode)?1:0)- outOutput[t][i]) * (fp(outOutput[t][i]));
		
		for (int m=0; m<numOfBlocks; m++)
		{
			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				deltaCellOutput[m][cell][i] = deltaCellOutput[m][cell][i]*momentumTerm + learningRate * dOutput[i] * outCell[t][m][cell];
			}
		}
		for (int m=0; m<numOfHiddenNodes; m++)
		{
			deltaHiddenOutput[m][i] = deltaHiddenOutput[m][i]*momentumTerm + learningRate * dOutput[i] * outHidden[t][m];
			
		}
		
		deltaBiasOutput[i] = deltaBiasOutput[i]*momentumTerm + learningRate * dOutput[i];
	}
	
	 
	
	//from X layer to cell input
	for (int j=0; j<numOfBlocks; j++)
	{
		for (int v=0; v<numOfCellsInBlock; v++)
		{
			double sum=0;
			//error truncated from cells
			for (int above=0; above<numOfOutputNodes; above++)
			{
				sum += weightCellOutput[j][v][above] * dOutput[above];
			}
			for (int above=0; above<numOfOutputNodes; above++)
			{
				sum += weightCellHidden[j][v][above] * dHidden[t-1][above];
			}
			errorS[j][v] = outOUT[t][j] * hpp(stateCell[t][j][v]) * sum; 
			//delta[m][j][cell] = *momentumTerm + learningRate * errorS[t][j][v] * derCellIn[t][m][j][v]
			
			//input to cell
			for (int m=0; m<numOfInputNodes; m++)
			{
				derInputCellIn[t][m][j][v] = derInputCellIn[t-1][m][j][v] * outFOR[t][j] + gpp(netCell[j][v]) * outIN[t][j] * outInput[t-1][m];
				deltaInputCell[m][j][v] = deltaInputCell[m][j][v]*momentumTerm + learningRate * errorS[j][v] * derInputCellIn[t][m][j][v];
			}
			
			//cell to cell
			for (int m=0; m<numOfBlocks; m++)
			{
				for (int m_cell=0; m_cell<numOfCellsInBlock; m_cell++)
				{
					derCellOutCellIn[t][m][m_cell][j][v] = derCellOutCellIn[t-1][m][m_cell][j][v] * outFOR[t][j] + gpp(netCell[j][v]) * outIN[t][j] * outCell[t-1][m][m_cell];
					
					deltaCellCell[m][m_cell][j][v] = deltaCellCell[m][m_cell][j][v]*momentumTerm + learningRate * errorS[j][v] * derCellOutCellIn[t][m][m_cell][j][v];
				}
			}
			
			//hidden to cell
			for (int m=0; m<numOfHiddenNodes; m++)
			{
				derHiddenCellIn[t][m][j][v] = derHiddenCellIn[t-1][m][j][v] * outFOR[t][j] + gpp(netCell[j][v]) * outIN[t][j] * outHidden[t-1][m];
				deltaHiddenCell[m][j][v] = deltaHiddenCell[m][j][v]*momentumTerm + learningRate * errorS[j][v] * derHiddenCellIn[t][m][j][v];
			}
			
			derBiasCellIn[t][j][v] = derBiasCellIn[t-1][j][v] * outFOR[t][j] + gpp(netCell[j][v]) * outIN[t][j];
			deltaBiasCell[j][v] = deltaBiasCell[j][v]*momentumTerm + learningRate * errorS[j][v];
		}
	}
	
	//from X to IN gate
	for (int j=0; j<numOfBlocks; j++)
	{
		//input to IN
		for (int m=0; m<numOfInputNodes; m++)
		{
			double sum=0;
			for (int v=0; v<numOfCellsInBlock; v++)
			{
				derInputIN[t][m][j][v] = derInputIN[t-1][m][j][v] * outFOR[t][j] + g(netCell[j][v]) * fp(outIN[t][j])*outInput[t-1][m];
				sum += errorS[j][v] * derInputIN[t][m][j][v];
			}
			deltaInputIN[m][j] = deltaInputIN[m][j]*momentumTerm + learningRate * sum;
		}
		
		//cell to IN
		for (int m=0; m<numOfBlocks; m++)
		{
			for (int m_cell=0; m_cell<numOfCellsInBlock; m_cell++)
			{
				double sum=0;
				for (int v=0; v<numOfCellsInBlock; v++)
				{
					derCellOutIN[t][m][m_cell][j][v] = derCellOutIN[t-1][m][m_cell][j][v] * outFOR[t][j] + g(netCell[j][v]) * fp(outIN[t][j])*outCell[t-1][m][m_cell];
					sum += errorS[j][v] * derCellOutIN[t][m][m_cell][j][v];
				}
				deltaCellIN[m][m_cell][j] = deltaCellIN[m][m_cell][j]*momentumTerm + learningRate * sum;
			}
		}
		
		//hidden to IN
		for (int m=0; m<numOfHiddenNodes; m++)
		{
			double sum=0;
			for (int v=0; v<numOfCellsInBlock; v++)
			{
				derHiddenIN[t][m][j][v] = derHiddenIN[t-1][m][j][v] * outFOR[t][j] + g(netCell[j][v]) * fp(outIN[t][j])*outHidden[t-1][m];
				sum += errorS[j][v] * derHiddenIN[t][m][j][v];
			}
			deltaHiddenIN[m][j] = deltaHiddenIN[m][j]*momentumTerm + learningRate * sum;
		}
		
		double sumB=0;
		for (int v=0; v<numOfCellsInBlock; v++)
		{
			derBiasIN[t][j][v] = derBiasIN[t-1][j][v] * outFOR[t][j] + g(netCell[j][v]) * fp(outIN[t][j]);
			sumB += errorS[j][v] * derBiasIN[t][j][v];
		}
		deltaBiasIN[j] = deltaBiasIN[j]*momentumTerm + learningRate * sumB;
	}
	
	//from X to FOR gate
	for (int j=0; j<numOfBlocks; j++)
	{
		//input to FOR
		for (int m=0; m<numOfInputNodes; m++)
		{
			double sum=0;
			for (int v=0; v<numOfCellsInBlock; v++)
			{
				derInputFOR[t][m][j][v] = derInputFOR[t-1][m][j][v] * outFOR[t][j] + stateCell[t-1][j][v] * fp(outFOR[t][j])*outInput[t-1][m];
				sum += errorS[j][v] * derInputFOR[t][m][j][v];
			}
			deltaInputFOR[m][j] = deltaInputFOR[m][j]*momentumTerm + learningRate * sum;
		}
		
		//cell to FOR
		for (int m=0; m<numOfBlocks; m++)
		{
			for (int m_cell=0; m_cell<numOfCellsInBlock; m_cell++)
			{
				double sum=0;
				for (int v=0; v<numOfCellsInBlock; v++)
				{
					derCellOutFOR[t][m][m_cell][j][v] = derCellOutFOR[t-1][m][m_cell][j][v] * outFOR[t][j] + stateCell[t-1][j][v] * fp(outFOR[t][j])*outCell[t-1][m][m_cell];
					sum += errorS[j][v] * derCellOutFOR[t][m][m_cell][j][v];
				}
				deltaCellFOR[m][m_cell][j] = deltaCellFOR[m][m_cell][j]*momentumTerm + learningRate * sum;
			}
		}
		
		//hidden to FOR
		for (int m=0; m<numOfHiddenNodes; m++)
		{
			double sum=0;
			for (int v=0; v<numOfCellsInBlock; v++)
			{
				derHiddenFOR[t][m][j][v] = derHiddenFOR[t-1][m][j][v] * outFOR[t][j] + stateCell[t-1][j][v] * fp(outFOR[t][j])*outHidden[t-1][m];
				sum += errorS[j][v] * derHiddenFOR[t][m][j][v];
			}
			deltaHiddenFOR[m][j] = deltaHiddenFOR[m][j]*momentumTerm + learningRate * sum;
		}
		
		double sumB=0;
		for (int v=0; v<numOfCellsInBlock; v++)
		{
			derBiasFOR[t][j][v] = derBiasFOR[t-1][j][v] * outFOR[t][j] + stateCell[t-1][j][v] * fp(outFOR[t][j]);
			sumB += errorS[j][v] * derBiasFOR[t][j][v];
		}
		deltaBiasFOR[j] = deltaBiasFOR[j]*momentumTerm + learningRate * sumB;
	}
//cout << "6 whc[0][0] size: " << weightHiddenCell[0][0].size() <<endl;	
	//from X to hidden node
	for (int i=0; i<numOfHiddenNodes; i++)
	{
		double sum=0;
		//error truncated from cells
		for (int above=0; above<numOfOutputNodes; above++)
		{
			sum += weightHiddenOutput.at(i).at(above) * dOutput.at(above);
		}
		for (int above=0; above<numOfHiddenNodes; above++)
		{
			sum += weightHiddenHidden.at(i).at(above) * dHidden[t-1].at(above);
		}
		
		dHidden.at(t).at(i) = fp(outHidden.at(t).at(i)) * sum;
		
		for (int m=0; m<numOfBlocks; m++)
		{
			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				deltaCellHidden.at(m)[cell].at(i) = deltaCellHidden.at(m)[cell].at(i)*momentumTerm + learningRate * dHidden.at(t).at(i) * outCell[t-1].at(m)[cell];
			}
		}
		for (int m=0; m<numOfHiddenNodes; m++)
		{
			deltaHiddenHidden.at(m).at(i) = deltaHiddenHidden.at(m).at(i)*momentumTerm + learningRate * dHidden.at(t).at(i) * outHidden[t-1].at(m);
			
		}
		for (int m=0; m<numOfInputNodes; m++)
		{
			deltaInputHidden.at(m).at(i) = deltaInputHidden.at(m).at(i)*momentumTerm + learningRate * dHidden.at(t).at(i) * outInput[t-1].at(m);
			
		}
		
		deltaBiasHidden.at(i) = deltaBiasHidden.at(i)*momentumTerm + learningRate * dHidden.at(t).at(i);
//cout << "6." <<i<<" whc[0][0] size: " << weightHiddenCell[0][0].size() <<endl;	
	}
//cout << "7 whc[0][0] size: " << weightHiddenCell[0][0].size() <<endl;	
	//from X to OUT gate
	for (int j=0; j<numOfBlocks; j++)
	{
		double sum=0;
		for (int v=0; v<numOfCellsInBlock; v++)
		{
			double sum2=0;
			//output
			for (int k=0; k<numOfOutputNodes; k++)
			{
				sum2 += weightCellOutput[j][v][k] * dOutput[k];
			}
			//cell
			//not needed? truncated
			
			//hidden, also not needed?
			for (int k=0; k<numOfHiddenNodes; k++)
			{
				sum2 += weightCellHidden[j][v][k] * dHidden[t][k];
			}
			
			sum += h(stateCell[t][j][v]) * sum2;
		}
	
		dOUT[j] = fp(outOUT[t][j]) * sum;
		
		for (int m=0; m<numOfBlocks; m++)
		{
			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				deltaCellOUT[m][cell][j] = deltaCellOUT[m][cell][j]*momentumTerm + learningRate * dOUT[j] * outCell[t-1][m][cell];
			}
		}
		for (int m=0; m<numOfHiddenNodes; m++)
		{
			deltaHiddenOUT[m][j] = deltaHiddenOUT[m][j]*momentumTerm + learningRate * dOUT[j] * outHidden[t-1][m];
			
		}
		for (int m=0; m<numOfInputNodes; m++)
		{
			deltaInputOUT[m][j] = deltaInputOUT[m][j]*momentumTerm + learningRate * dOUT[j] * outInput[t-1][m];
			
		}
		
		deltaBiasOUT[j] = deltaBiasOUT[j]*momentumTerm + learningRate * dOUT[j];
	}
//cout << "8 whc[0][0] size: " << weightHiddenCell[0][0].size() <<endl;	
	
	
	//Go through all deltas and actually update
	//
	checkWeightChange();
	updateWeights();
	
	return totalWeightChange();
}

void LSTM::updateWeights()
{
	for (int block=0; block<numOfBlocks; block++)
	{

		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
		{
			for (int n=0; n<numOfOutputNodes; n++)
				weightCellOutput[block][cell][n] += deltaCellOutput[block][cell][n];
				
			for (int n=0; n<numOfHiddenNodes; n++)
				weightCellHidden[block][cell][n] += deltaCellHidden[block][cell][n];
			
			for (int block2=0; block2<numOfBlocks; block2++)
			{
				weightCellIN[block][cell][block2] += deltaCellIN[block][cell][block2];
				weightCellOUT[block][cell][block2] += deltaCellOUT[block][cell][block2];
				weightCellFOR[block][cell][block2] += deltaCellFOR[block][cell][block2];
				
				for (int cell2=0; cell2<numOfCellsInBlock; cell2++)
				{
					weightCellCell[block][cell][block2][cell2] += deltaCellCell[block][cell][block2][cell2];
				}
			}
		}
	}
	
	for (int n=0; n<numOfInputNodes; n++)
	{

		for (int block=0; block<numOfBlocks; block++)
		{

			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				weightInputCell[n][block][cell] += deltaInputCell[n][block][cell];
			}
			weightInputIN[n][block] += deltaInputIN[n][block];
			weightInputFOR[n][block] += deltaInputFOR[n][block];
			weightInputOUT[n][block] += deltaInputOUT[n][block];
		}
		
		for (int n2=0; n2<numOfHiddenNodes; n2++)
		{
			weightInputHidden[n][n2] += deltaInputHidden[n][n2];
		}
	}
	
	for (int n=0; n<numOfHiddenNodes; n++)
	{
		
		for (int n2=0; n2<numOfOutputNodes; n2++)
			weightHiddenOutput[n][n2] += deltaHiddenOutput[n][n2];
		for (int n2=0; n2<numOfInputNodes; n2++)
			weightHiddenHidden[n][n2] += deltaHiddenHidden[n][n2];
		
		for (int block=0; block<numOfBlocks; block++)
		{
			weightHiddenFOR[n][block] += deltaHiddenFOR[n][block];
			weightHiddenIN[n][block] += deltaHiddenIN[n][block];
			weightHiddenOUT[n][block] += deltaHiddenOUT[n][block];
			
			for (int cell=0; cell<numOfCellsInBlock; cell++)
				weightHiddenCell[n][block][cell] += deltaHiddenCell[n][block][cell];
		}
	}
	
	for (int block=0; block<numOfBlocks; block++)
	{
		weightBiasIN[block] += deltaBiasIN[block];
		weightBiasOUT[block] += deltaBiasOUT[block];
		weightBiasFOR[block] += deltaBiasFOR[block] ;
		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
			weightBiasCell[block][cell] += deltaBiasCell[block][cell];
	}
	
	for (int n=0; n<numOfHiddenNodes; n++)
		weightBiasHidden[n] += deltaBiasHidden[n];
	for (int n=0; n<numOfOutputNodes; n++)
		weightBiasOutput[n] += deltaBiasOutput[n];
}

double LSTM::totalWeightChange()
{
	double sum=0;
	for (int block=0; block<numOfBlocks; block++)
	{

		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
		{
			for (int n=0; n<numOfOutputNodes; n++)
			{
				sum += fabs(deltaCellOutput[block][cell][n]);
			}
				
			for (int n=0; n<numOfHiddenNodes; n++)
			{
				sum += fabs(deltaCellHidden[block][cell][n]);
			}
			
			for (int block2=0; block2<numOfBlocks; block2++)
			{
				sum += fabs(deltaCellIN[block][cell][block2]);
				sum += fabs(deltaCellOUT[block][cell][block2]);
				sum+= fabs(deltaCellFOR[block][cell][block2]);
				for (int cell2=0; cell2<numOfCellsInBlock; cell2++)
				{
					sum += fabs(deltaCellCell[block][cell][block2][cell2]);
				}
			}
		}
	}
	
	for (int n=0; n<numOfInputNodes; n++)
	{

		for (int block=0; block<numOfBlocks; block++)
		{

			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				sum+= fabs(deltaInputCell[n][block][cell]);
			}
			sum += fabs(deltaInputIN[n][block]);
			sum += fabs(deltaInputFOR[n][block]);
			sum += fabs(deltaInputOUT[n][block]);
		}
		
		for (int n2=0; n2<numOfHiddenNodes; n2++)
		{
			sum+= fabs(deltaInputHidden[n][n2]);
		}
	}
	
	for (int n=0; n<numOfHiddenNodes; n++)
	{
		
		for (int n2=0; n2<numOfOutputNodes; n2++)
			sum += fabs(deltaHiddenOutput[n][n2]);
		for (int n2=0; n2<numOfInputNodes; n2++)
			sum += fabs(deltaHiddenHidden[n][n2]);
		
		for (int block=0; block<numOfBlocks; block++)
		{
			sum += fabs(deltaHiddenFOR[n][block]);
			sum += fabs(deltaHiddenIN[n][block]);
			sum += fabs(deltaHiddenOUT[n][block]);
			
			for (int cell=0; cell<numOfCellsInBlock; cell++)
				sum += fabs(deltaHiddenCell[n][block][cell]);
		}
	}
	
	for (int block=0; block<numOfBlocks; block++)
	{
		sum += fabs(deltaBiasIN[block]);
		sum+= fabs(deltaBiasOUT[block]);
		sum += fabs(deltaBiasFOR[block]);
		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
			sum += fabs(deltaBiasCell[block][cell]);
	}
	
	for (int n=0; n<numOfHiddenNodes; n++)
		sum += fabs(deltaBiasHidden[n]);
	for (int n=0; n<numOfOutputNodes; n++)
		sum += fabs(deltaBiasOutput[n]);
	
	
	return sum;
}

void LSTM::checkWeightChange()
{
	
	for (int block=0; block<numOfBlocks; block++)
	{

		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
		{
			for (int n=0; n<numOfOutputNodes; n++)
				assert(!std::isnan((double)deltaCellOutput[block][cell][n]));
				
			for (int n=0; n<numOfHiddenNodes; n++)
				assert(!std::isnan((double)deltaCellHidden[block][cell][n]));
			
			for (int block2=0; block2<numOfBlocks; block2++)
			{
				assert(!std::isnan((double)deltaCellIN[block][cell][block2]));
				assert(!std::isnan((double)deltaCellOUT[block][cell][block2]));
				assert(!std::isnan((double)deltaCellFOR[block][cell][block2]));
				
				for (int cell2=0; cell2<numOfCellsInBlock; cell2++)
				{
					assert(!std::isnan((double)deltaCellCell[block][cell][block2][cell2]));
				}
			}
		}
	}
	
	for (int n=0; n<numOfInputNodes; n++)
	{

		for (int block=0; block<numOfBlocks; block++)
		{

			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				assert(!std::isnan((double)deltaInputCell[n][block][cell]));
			}
			assert(!std::isnan((double)deltaInputIN[n][block]));
			assert(!std::isnan((double)deltaInputFOR[n][block]));
			assert(!std::isnan((double)deltaInputOUT[n][block]));
		}
		
		for (int n2=0; n2<numOfHiddenNodes; n2++)
		{
			assert(!std::isnan((double)deltaInputHidden[n][n2]));
		}
	}
	
	for (int n=0; n<numOfHiddenNodes; n++)
	{
		
		for (int n2=0; n2<numOfOutputNodes; n2++)
			assert(!std::isnan((double)deltaHiddenOutput[n][n2]));
		for (int n2=0; n2<numOfInputNodes; n2++)
			assert(!std::isnan((double)deltaHiddenHidden[n][n2]));
		
		for (int block=0; block<numOfBlocks; block++)
		{
			assert(!std::isnan((double)deltaHiddenFOR[n][block]));
			assert(!std::isnan((double)deltaHiddenIN[n][block]));
			assert(!std::isnan((double)deltaHiddenOUT[n][block]));
			
			for (int cell=0; cell<numOfCellsInBlock; cell++)
				assert(!std::isnan((double)deltaHiddenCell[n][block][cell]));
		}
	}
	
	for (int block=0; block<numOfBlocks; block++)
	{
		assert(!std::isnan((double)deltaBiasIN[block]));
		assert(!std::isnan((double)deltaBiasOUT[block]));
		assert(!std::isnan((double)deltaBiasFOR[block]));
		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
			assert(!std::isnan((double)deltaBiasCell[block][cell]));
	}
	
	for (int n=0; n<numOfHiddenNodes; n++)
		assert(!std::isnan((double)deltaBiasHidden[n]));
	for (int n=0; n<numOfOutputNodes; n++)
		assert(!std::isnan((double)deltaBiasOutput[n]));
	
}


void LSTM::save(string fileName)
{
	ofstream out;
	out.open(fileName);
	assert(out.is_open());
	
	out << learningRate << endl;
	out << momentumTerm << endl;
	out << numOfInputNodes << endl;
	out << numOfOutputNodes << endl;
	out <<  numOfBlocks << endl;
	out <<  numOfCellsInBlock << endl;
	out <<  numOfHiddenNodes << endl;
	
	for (int block=0; block<numOfBlocks; block++)
	{

		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
		{
			for (int n=0; n<numOfOutputNodes; n++)
				out << weightCellOutput[block][cell][n] << endl ;// deltaCellOutput[block][cell][n];
				
			for (int n=0; n<numOfHiddenNodes; n++)
				out << weightCellHidden[block][cell][n] << endl ;// deltaCellHidden[block][cell][n];
			
			for (int block2=0; block2<numOfBlocks; block2++)
			{
				out << weightCellIN[block][cell][block2] << endl ;// deltaCellIN[block][cell][block2];
				out << weightCellOUT[block][cell][block2] << endl ;// deltaCellOUT[block][cell][block2];
				out << weightCellFOR[block][cell][block2] << endl ;// deltaFOR[block][cell][block2];
				
				for (int cell2=0; cell2<numOfCellsInBlock; cell2++)
				{
					out << weightCellCell[block][cell][block2][cell2] << endl ;// deltaCellCell[block][cell][block2][cell2];
				}
			}
		}
	}
	
	for (int n=0; n<numOfInputNodes; n++)
	{

		for (int block=0; block<numOfBlocks; block++)
		{

			for (int cell=0; cell<numOfCellsInBlock; cell++)
			{
				out << weightInputCell[n][block][cell] << endl ;// deltaInputCell[n][block][cell];
			}
			out << weightInputIN[n][block] << endl ;// deltaInputIN[n][block];
			out << weightInputFOR[n][block] << endl ;// deltaInputFOR[n][block];
			out << weightInputOUT[n][block] << endl ;// deltaInputOUT[n][block];
		}
		
		for (int n2=0; n2<numOfHiddenNodes; n2++)
		{
			out << weightInputHidden[n][n2] << endl ;// deltaInputHidden[n][n2];
			out << deltaInputHidden[n][n2] << endl ;// deltaInputHidden[n][n2] ;
		}
	}
	
	for (int n=0; n<numOfHiddenNodes; n++)
	{
		
		for (int n2=0; n2<numOfOutputNodes; n2++)
			out << weightHiddenOutput[n][n2] << endl ;// deltaHiddenOutput[n][n2];
		for (int n2=0; n2<numOfInputNodes; n2++)
			out << weightHiddenHidden[n][n2] << endl ;// deltaHiddenHidden[n][n2];
		
		for (int block=0; block<numOfBlocks; block++)
		{
			out << weightHiddenFOR[n][block] << endl ;// deltaHiddenFOR[n][block];
			out << weightHiddenIN[n][block] << endl ;// deltaHiddenIN[n][block];
			out << weightHiddenOUT[n][block] << endl ;// deltaHiddenOUT[n][block];
			
			for (int cell=0; cell<numOfCellsInBlock; cell++)
				out << weightHiddenCell[n][block][cell] << endl ;// deltaHiddenCell[n][block][cell];
		}
	}
	
	for (int block=0; block<numOfBlocks; block++)
	{
		out << weightBiasIN[block] << endl ;// deltaBiasIN[block];
		out << weightBiasOUT[block] << endl ;// deltaBiasOUT[block];
		out << weightBiasFOR[block] << endl ;// deltaBiasFOR[block] ;
		
		for (int cell=0; cell<numOfCellsInBlock; cell++)
			out << weightBiasCell[block][cell] << endl ;// deltaBiasCell[block][cell];
	}
	
	for (int n=0; n<numOfHiddenNodes; n++)
		out << weightBiasHidden[n] << endl ;// deltaBiasHidden[n];
	for (int n=0; n<numOfOutputNodes; n++)
		out << weightBiasOutput[n] << endl;
}

bool LSTM::isRight(const vector<double> &result, unsigned int correctNode)
{
	double highest = (result)[correctNode];
	for (unsigned int i = 0; i < numOfOutputNodes; i++)
	{
		if (i!=correctNode && (result)[i] >= highest)
		{
			return false;
		}
	}
	return true;
}

void LSTM::shiftTime()
{
	swap(dHidden[0],dHidden[1]);//[t][n]	
	
	swap(outOutput[0],outOutput[1]);//[t][n]
	swap(outHidden[0],outHidden[1]);//[t][n]
	swap(outInput[0],outInput[1]);//[t][n]
	swap(outCell[0],outCell[1]);//[t][block][cell]
	swap(outFOR[0],outFOR[1]);//[t][block]
	swap(outIN[0],outIN[1]);//[t][block]
	swap(outOUT[0],outOUT[1]);//[t][block]
	
	swap(derInputCellIn[0],derInputCellIn[1]);//[t][n][block][cell]
	swap(derCellOutCellIn[0],derCellOutCellIn[1]);//[t][block][cell][block][cell]
	swap(derHiddenCellIn[0],derHiddenCellIn[1]);//[t][n][block][cell]
	swap(derInputIN[0],derInputIN[1]);//[t][n][block][cell]
	swap(derHiddenIN[0],derHiddenIN[1]);//[t][n][block][cell]
	swap(derCellOutIN[0],derCellOutIN[1]);//[t][block][cell][block][cell]
	swap(derInputFOR[0],derInputFOR[1]);//[t][n][block][cell]
	swap(derHiddenFOR[0],derHiddenFOR[1]);//[t][n][block][cell]
	swap(derCellOutFOR[0],derCellOutFOR[1]);//[t][block][cell][block][cell]
	
	swap(derBiasCellIn[0],derBiasCellIn[1]);//[t][block][cell]
	swap(derBiasIN[0],derBiasIN[1]);//[t][block][cell]
	swap(derBiasFOR[0],derBiasFOR[1]);//[t][block][cell]
	
	swap(stateCell[0],stateCell[1]);//[t][block][cell]
}
