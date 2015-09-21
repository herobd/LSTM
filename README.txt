LSTM
One hidden layer Long Short-term Memory Neural Net

Implementation by Brian Davis for CS678-Advanced Nueral Networks and Machine Learning

This was a project where we got to implement the learning algorithm of our choice.
Giving a training and testing dataset (ARFF format) it trains a LSTM network and
tests it, returning a final accuracy. You also can save the network weights.

The makefile should allow easy building.

Usage:
LSTM <options>
Options:

-o <file name>		Sets the output file (contains general output)
-test <arff file>		File containing testing data
-train <arff file>	File containing training data
-saveNN <file name>	File to save trained network in
-loadNN <file name>	Load network from file
-m <value>			Momentum term
-l <value>			Learning rate
-limit <value>		Maximum number of training epochs
-blocks <value>		Number of LSTM blocks
-cells <value>		Number of cells in each block
-hiddens <value>		Number of hidden nodes (regular NN nodes)

