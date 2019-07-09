// Framework: GraphFlow
// Author: Machine Learning Group of UChicago
// Main Contributor: Hy Truong Son
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

// To compile: g++ -std=c++11 -pthread test_SMP_beta.cpp

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <fstream>

#include "../GraphFlow/SMP_beta.h"
#include "../kaggle_utils/MoleculeBuilder.cpp"

using namespace std;

const bool useCoulomb = true;
const int max_nVertices = 30;
const int nChanels = 10;
const int nLevels = 2;
const int nFeatures = 5;
const int nDepth = 5;

const int nThreads = 2;

const float learning_rate = 0.001;
const int nEpochs = 100;

string model_fn = "SMP_beta-model.dat";

SMP_beta train_network(useCoulomb, max_nVertices, nLevels, nChanels, nFeatures, nDepth);
SMP_beta test_network(useCoulomb, max_nVertices, nLevels, nChanels, nFeatures, nDepth);

// Get the millisecond
void time_ms(long int &ms) {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

// Difference in milliseconds
long int difftime_ms(long int &end, long int &start) {
	return end - start;
}

Molecule **molecule;

int main(int argc, char **argv) {
	// Measuring time
	long int start, end;

	cout << "Molecule builder init..." << std::endl;
	MoleculeBuilder* moleculeBuilder = new MoleculeBuilder();

	molecule = moleculeBuilder->getMolecules();
	int nMolecules = moleculeBuilder->getNumberOfMolecules();

	// Starting time
	time_ms(start);

	cout << "--- Learning ------------------------------" << endl;

	DenseGraph **graphs = new DenseGraph* [nMolecules];
	
	double* targets = new double[nMolecules];
	double* predict = new double[nMolecules];

	for (int i = 0; i < nMolecules; ++i) {
		graphs[i] = molecule[i] -> graph;
		targets[i] = molecule[i] -> target;
	}

	cout << "Multi-threadings" << endl;
	train_network.init_multi_threads(nThreads);
	cout << "Initialized multi-threading.\n";


	for (int j = 0; j < nEpochs; ++j) {
		for (int batch = 0; batch < 10; ++batch){
			DenseGraph** _graphs = new DenseGraph*[100];
			double* _targets = new double[100];
			for(int ind = 0; ind < 100; ++ind){
				_graphs[ind] = graphs[batch * 100 + ind];
				_targets[ind] = targets[batch * 100 + ind];
			}
			train_network.Threaded_BatchLearn(100, _graphs, _targets, learning_rate);
			double totalLoss = train_network.getLoss(100, _graphs, _targets);
			cout << "Done epoch " << j + 1 << "/" << nEpochs << "  ||  Batch #" << batch + 1 << "\tLoss : " << totalLoss << "\n";
		}

		double totalLoss = train_network.getLoss(nMolecules, graphs, targets);
		cout << "\tDone epoch " << j + 1 << " / " << nEpochs << "->\tAverage loss per molecule : " << totalLoss / nMolecules << endl;
	}

	// Save model to file
	train_network.save_model(model_fn);

	cout << endl << "--- Predicting ----------------------------" << endl;

	// Load model from file
	test_network.load_model(model_fn);

	for (int i = 0; i < nMolecules; ++i) {
		double predict = test_network.Predict(molecule[i] -> graph);
		cout << "Molecule " << (i + 1) << "\t\ttarget : " << targets[i] << "\tprediction : " << predict <<"\n";
		ofstream outfile("predictions/molecule_" + std::to_string(i + 1), std::ios::out);
		outfile << predict;
		outfile << "\n";
		outfile.close();
	}

	// Ending time
	time_ms(end);

	cout << endl << difftime_ms(end, start) << " ms" << endl;

	return 0;
}