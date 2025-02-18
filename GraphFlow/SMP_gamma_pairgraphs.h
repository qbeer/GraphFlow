// Framework: GraphFlow
// Class: Second-order Steerable Message Passing (Gamma version + Pair graphs)
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#ifndef __SMP_GAMMA_PAIRGRAPHS_H_INCLUDED__
#define __SMP_GAMMA_PAIRGRAPHS_H_INCLUDED__

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <assert.h>

#include "DenseGraph.h"
#include "SumGradients.h"
#include "CacheParameters.h"
#include "Adam.h"
#include "GraphFlow.h"

using namespace std;

class SMP_gamma_pairgraphs {
public:
	// Each level
	struct Level {
		// Number of chanels in this level
		int nChanels;

		// For reducing the number of chanels
		Matrix *K;

		// Bias
		Vector *b;

		// For WL features
		MatMul **matmul;

		// LeakyReLU3D
		LeakyReLU3D **f;

		// Reshape (if necessary)
		Reshape3D **f_reshape;

		// MatTensorMul
		MatTensorMul ***X_mul_f;

		// TensorMatMul
		TensorMatMul ***quadratic;

		// RisiContraction_4
		RisiContraction_4 **contract;

		// Cut number of chanels 
		Reshape2D **reshape2D;
		MatMul **represent;
		Reshape3D **reshape3D;

		// Addition with bias
		VectorAddTensor **add;

		// Permutation matrices
		Matrix ***X;

		// Permutation matrices transpose
		Matrix ***X_transpose;

		// Indices for the receptive fields
		vector<int> *phi;
	};

	SMP_gamma_pairgraphs(int max_nVertices_1, int max_nVertices_2, int max_receptive_field, int nLevels, int nChanels, int nFeatures_1, int nFeatures_2) {
		this -> max_nVertices_1 = max_nVertices_1;
		this -> max_nVertices_2 = max_nVertices_2;
		this -> max_receptive_field = max_receptive_field;
		this -> nLevels = nLevels;
		this -> nChanels = nChanels;
		this -> nFeatures_1 = nFeatures_1;
		this -> nFeatures_2 = nFeatures_2;

		// Multi-threading mode
		this -> nThreads = 0;
		this -> multi_threaded = false;

		computation_graph();
		weights_initialization();
	}

	// +-------------------------+
	// | Multi-threading (Begin) |
	// +-------------------------+

	void init_multi_threads(int nThreads) {
		assert(nThreads > 1);

		this -> nThreads = nThreads;
		multi_threaded = true;

		// Initialize all instances
		instance = new SMP_gamma_pairgraphs* [this -> nThreads];
		for (int i = 0; i < nThreads; ++i) {
			instance[i] = new SMP_gamma_pairgraphs(max_nVertices_1, max_nVertices_2, max_receptive_field, nLevels, nChanels, nFeatures_1, nFeatures_2);
		}

		// Initialize multi-threaded jobs
		job = new std::thread [this -> nThreads];
	}

	// +-----------------------+
	// | Multi-threading (End) |
	// +-----------------------+

	void computation_graph_(
		int max_nVertices, int nFeatures, 
		SMP_gamma_pairgraphs::Level **level, Matrix **feature, ShrinkTensor ***shrinked, LeakyReLU ***vertex_feature, SumVectors **level_feature,
		int **adj, int **shortest_paths
	) {
		// Synthesized graph features
		for (int v = 0; v < max_nVertices; ++v) {
			feature[v] = new Matrix(nFeatures, 1);
		}

		// Adjacency matrix
		for (int i = 0; i < max_nVertices; ++i) {
			adj[i] = new int [max_nVertices];
		}

		// Shortest Paths for Floyd-Warshall Algorithms
		for (int i = 0; i < max_nVertices; ++i) {
			shortest_paths[i] = new int [max_nVertices];
		}

		// Number of chanels
		int C, prevC;

		// Maximum size of the receptive field
		int N = max_receptive_field;

		// Each level
		for (int l = 0; l <= nLevels; ++l) {
			// Create the level
			level[l] = new Level();

			// Receptive fields
			level[l] -> phi = new vector<int> [max_nVertices];

			// Level 0
			if (l == 0) {
				// Number of chanels in this level
				level[l] -> nChanels = nChanels;
				C = level[l] -> nChanels;

				level[l] -> matmul = new MatMul* [max_nVertices];
				level[l] -> f_reshape = new Reshape3D* [max_nVertices];
				level[l] -> f = new LeakyReLU3D* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> matmul[v] = new MatMul(C, 1);
					level[l] -> f_reshape[v] = new Reshape3D(1, 1, C);
					level[l] -> f[v] = new LeakyReLU3D(1, 1, C);
				}
			} 

			// Level from 1 to nLevels
			if (l > 0) {
				// Number of chanels in this level
				level[l] -> nChanels = level[l - 1] -> nChanels / 2;
				if (level[l] -> nChanels == 0) {
					level[l] -> nChanels = 1;
				}

				C = level[l] -> nChanels;

				// Number of chanels in the previous level
				prevC = level[l - 1] -> nChanels;

				// Bias
				level[l] -> b = new Vector(C);

				// For reducing the number of chanels
				level[l] -> K = new Matrix(nContractions * prevC, C);

				// For computations
				level[l] -> f = new LeakyReLU3D* [max_nVertices];
				level[l] -> contract = new RisiContraction_4* [max_nVertices];
				level[l] -> reshape2D = new Reshape2D* [max_nVertices];
				level[l] -> represent = new MatMul* [max_nVertices];
				level[l] -> reshape3D = new Reshape3D* [max_nVertices];
				level[l] -> add = new VectorAddTensor* [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> f[v] = new LeakyReLU3D(N, N, C);
					level[l] -> contract[v] = new RisiContraction_4(N, N, nContractions * prevC);
					level[l] -> reshape2D[v] = new Reshape2D(N * N, nContractions * prevC);
					level[l] -> represent[v] = new MatMul(N * N, nContractions * prevC, nContractions * prevC, C);
					level[l] -> reshape3D[v] = new Reshape3D(N, N, C);
					level[l] -> add[v] = new VectorAddTensor(N, N, C); 
				}

				level[l] -> X_mul_f = new MatTensorMul** [max_nVertices];
				level[l] -> quadratic = new TensorMatMul** [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> X_mul_f[v] = new MatTensorMul* [max_nVertices];
					level[l] -> quadratic[v] = new TensorMatMul* [max_nVertices];
					
					for (int w = 0; w < max_nVertices; ++w) {
						level[l] -> X_mul_f[v][w] = new MatTensorMul(N, N, prevC);
						level[l] -> quadratic[v][w] = new TensorMatMul(N, N, prevC);
					}
				}

				// For permutation matrices
				level[l] -> X = new Matrix** [max_nVertices];
				level[l] -> X_transpose = new Matrix** [max_nVertices];

				for (int v = 0; v < max_nVertices; ++v) {
					level[l] -> X[v] = new Matrix* [max_nVertices];
					level[l] -> X_transpose[v] = new Matrix* [max_nVertices];

					for (int w = 0; w < max_nVertices; ++w) {
						level[l] -> X[v][w] = new Matrix(N, N);
						level[l] -> X_transpose[v][w] = new Matrix(N, N);
					}
				}
			}
		}

		// On top of everything
		for (int l = 0; l <= nLevels; ++l) {
			// Number of chanels in this level
			C = level[l] -> nChanels;

			shrinked[l] = new ShrinkTensor* [max_nVertices];
			vertex_feature[l] = new LeakyReLU* [max_nVertices];
			level_feature[l] = new SumVectors(C);

			for (int v = 0; v < max_nVertices; ++v) {
				shrinked[l][v] = new ShrinkTensor(C);
				vertex_feature[l][v] = new LeakyReLU(C);
			}
		}
	}

	void computation_graph() {
		// +--------------------------+
		// | Component initialization |
		// +--------------------------+

		// Synthesized graph features
		feature_1 = new Matrix* [max_nVertices_1];
		feature_2 = new Matrix* [max_nVertices_2];

		// Mapping from the original synthesized graph features into chanels
		H_1 = new Matrix(nChanels, nFeatures_1);
		H_2 = new Matrix(nChanels, nFeatures_2);

		// Multiple levels
		level_1 = new Level* [nLevels + 1];
		level_2 = new Level* [nLevels + 1];

		// On top of everything
		shrinked_1 = new ShrinkTensor** [nLevels + 1];
		vertex_feature_1 = new LeakyReLU** [nLevels + 1];
		level_feature_1 = new SumVectors* [nLevels + 1];

		shrinked_2 = new ShrinkTensor** [nLevels + 1];
		vertex_feature_2 = new LeakyReLU** [nLevels + 1];
		level_feature_2 = new SumVectors* [nLevels + 1];

		// Adjacency matrix
		adj_1 = new int* [max_nVertices_1];
		adj_2 = new int* [max_nVertices_2];

		// Shortest Paths by Floyd-Warshall algorithm
		shortest_paths_1 = new int* [max_nVertices_1];
		shortest_paths_2 = new int* [max_nVertices_2];

		// Computation graph for the original graph
		computation_graph_(max_nVertices_1, nFeatures_1, level_1, feature_1, shrinked_1, vertex_feature_1, level_feature_1, adj_1, shortest_paths_1);

		// Computation graph for the line graph
		computation_graph_(max_nVertices_2, nFeatures_2, level_2, feature_2, shrinked_2, vertex_feature_2, level_feature_2, adj_2, shortest_paths_2);

		// Merge the two original graph and line graph
		int nTotalFeatures = 0;
		for (int l = 0; l <= nLevels; ++l) {
			nTotalFeatures += level_1[l] -> nChanels;
			nTotalFeatures += level_2[l] -> nChanels;
		}

		// Fully-connected layers
		graph_feature = new ConcatVectors(nTotalFeatures);

		int nHidden_1 = max(nTotalFeatures / 2, 10);
		int nHidden_2 = max(nHidden_1 / 2, 10);

		W1 = new Matrix(nHidden_1, nTotalFeatures);
		W2 = new Matrix(nHidden_2, nHidden_1);
		W3 = new Vector(nHidden_2);

		hidden_1 = new MatVecMul(nHidden_1);
		hidden_relu_1 = new LeakyReLU(nHidden_1);

		hidden_2 = new MatVecMul(nHidden_2);
		hidden_relu_2 = new LeakyReLU(nHidden_2);

		predict = new InnerProduct();
		sql = new SquaredLoss();

		// Target
		target = new Vector(1);

		// +-------------------+
		// | Computation graph |
		// +-------------------+
		
		graph = new GraphFlow();

		// +-----------------------------+
		// | Stochastic Gradient Descent |
		// +-----------------------------+

		sgd = new Adam();
		sgd -> add(H_1);
		sgd -> add(H_2);
		for (int l = 1; l <= nLevels; ++l) {
			// Graph 1
			sgd -> add(level_1[l] -> K);
			sgd -> add(level_1[l] -> b);

			// Graph 2
			sgd -> add(level_2[l] -> K);
			sgd -> add(level_2[l] -> b);
		}
		sgd -> add(W1);
		sgd -> add(W2);
		sgd -> add(W3);

		// Sum gradients
		sum_gradients = new SumGradients();
		for (int i = 0; i < sgd -> params.size(); ++i) {
			sum_gradients -> add(sgd -> params[i]);
		}

		// Cache parameters
		cache_parameters = new CacheParameters();
		for (int i = 0; i < sgd -> params.size(); ++i) {
			cache_parameters -> add(sgd -> params[i]);
		}
	}

	void weights_initialization() {
		for (int i = 0; i < sgd -> params.size(); ++i) {
			graph -> uniform_init(sgd -> params[i]);
		}
	}	

	void update_adjacency(DenseGraph *molecule, int **adj) {
		// Get the original graph
		for (int i = 0; i < molecule -> nVertices; ++i) {
			for (int j = 0; j < molecule -> nVertices; ++j) {
				adj[i][j] = molecule -> adj[i][j];
			}
		}
	}

	void update_feature(DenseGraph *molecule, Matrix **feature) {
		// Get the original graph feature
		for (int v = 0; v < molecule -> nVertices; ++v) {
			for (int f = 0; f < molecule -> nFeatures; ++f) {
				feature[v] -> value[feature[v] -> index(f, 0)] = molecule -> feature[v][f];
			}
		}
	}

	void floyd_warshall(DenseGraph *molecule, int **adj, int **shortest_paths) {
		for (int i = 0; i < molecule -> nVertices; ++i) {
			for (int j = 0; j < molecule -> nVertices; ++j) {
				shortest_paths[i][j] = INF;
				if (i == j) {
					shortest_paths[i][j] = 0;
				} else {
					if (adj[i][j] > 0) {
						shortest_paths[i][j] = 1;
						shortest_paths[j][i] = 1;
					}
				}
			}
		}

		for (int k = 0; k < molecule -> nVertices; ++k) {
			for (int i = 0; i < molecule -> nVertices; ++i) {
				for (int j = 0; j < molecule -> nVertices; ++j) {
					shortest_paths[i][j] = min(shortest_paths[i][j], shortest_paths[i][k] + shortest_paths[k][j]);
				}
			}
		}
	}

	void union_set(vector<int> &A, vector<int> &B) {
		for (int i = 0; i < B.size(); ++i) {
			bool found = false;
			for (int j = 0; j < A.size(); ++j) {
				if (A[j] == B[i]) {
					found = true;
					break;
				}
			}
			if (!found) {
				A.push_back(B[i]);
			}
		}
	}

	void init_permutation_matrix(Matrix *X, vector<int> &phi1, vector<int> &phi2) {
		assert(X -> nRows == phi1.size());
		assert(X -> nColumns == phi2.size());
		
		for (int i = 0; i < phi1.size(); ++i) {
			for (int j = 0; j < phi2.size(); ++j) {
				int index = X -> index(i, j);
				X -> value[index] = 0;
				if (phi1[i] == phi2[j]) {
					X -> value[index] = 1;
				}
			}
		}
	} 

	void limit_receptive_field(int v, vector<int> &A, int **shortest_paths) {
		for (int i = 0; i < A.size(); ++i) {
			for (int j = i + 1; j < A.size(); ++j) {
				if (shortest_paths[v][A[i]] > shortest_paths[v][A[j]]) {
					swap(A[i], A[j]);
				}
			}
		}

		int u, d;
		while (A.size() > max_receptive_field) {
			d = shortest_paths[v][A[A.size() - 1]];
			while (true) {
				u = A[A.size() - 1];
				if (shortest_paths[v][u] == d) {
					A.pop_back();
				} else {
					break;
				}
			}
		}
		
		assert(A.size() <= max_receptive_field);
		assert(A.size() > 0);
		assert(A[0] == v);
	}

	void init_receptive_field_permutation_matrix(DenseGraph *molecule, SMP_gamma_pairgraphs::Level **level, int **shortest_paths) {
		// Constructing the receptive fields
		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					level[l] -> phi[v].clear();
					level[l] -> phi[v].push_back(v);
				}
			} else {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					level[l] -> phi[v].clear();
					
					for (int u = 0; u < molecule -> nVertices; ++u) {
						if (shortest_paths[u][v] <= 1) {
							union_set(level[l] -> phi[v], level[l - 1] -> phi[u]);
						}
					}

					// Limit the size of the receptive field
					if (level[l] -> phi[v].size() > max_receptive_field) {
						limit_receptive_field(v, level[l] -> phi[v], shortest_paths);
					}
				}
			}
		}
 
		// Constructing the permutation matrices
		for (int l = 1; l <= nLevels; ++l) {
			for (int v = 0; v < molecule -> nVertices; ++v) {
				for (int i = 0; i < level[l] -> phi[v].size(); ++i) {
					int w = level[l] -> phi[v][i];
						
					level[l] -> X[v][w] -> setParameter(level[l] -> phi[v].size(), level[l - 1] -> phi[w].size());
					init_permutation_matrix(level[l] -> X[v][w], level[l] -> phi[v], level[l - 1] -> phi[w]);

					level[l] -> X_transpose[v][w] -> setParameter(level[l - 1] -> phi[w].size(), level[l] -> phi[v].size());
					init_permutation_matrix(level[l] -> X_transpose[v][w], level[l - 1] -> phi[w], level[l] -> phi[v]);
				}
			}
		}
	}

	void complete_computation_graph_(DenseGraph *molecule, SMP_gamma_pairgraphs::Level **level, Matrix *H, Matrix **feature, int **shortest_paths, 
		ShrinkTensor ***shrinked, LeakyReLU ***vertex_feature, SumVectors **level_feature) {

		// Number of chanels
		int C, prevC;

		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				// Number of chanels in this level
				C = level[l] -> nChanels;

				for (int v = 0; v < molecule -> nVertices; ++v) {
					level[l] -> matmul[v] -> setParameter(H, feature[v]);
					graph -> add(level[l] -> matmul[v], MATMUL);

					level[l] -> f_reshape[v] -> setParameter(level[l] -> matmul[v], 1, 1, C);
					graph -> add(level[l] -> f_reshape[v], RESHAPE3D);

					level[l] -> f[v] -> setParameter(level[l] -> f_reshape[v]);
					graph -> add(level[l] -> f[v], LEAKYRELU3D);
				}
			}

			if (l > 0) {
				for (int v = 0; v < molecule -> nVertices; ++v) {
					// Number of chanels in this level
					C = level[l] -> nChanels;

					// Number of chanels in the previous level
					prevC = level[l - 1] -> nChanels;

					// Size of the receptive field
					int size = level[l] -> phi[v].size();

					// Contraction
					level[l] -> contract[v] -> setParameter(size, prevC);
					level[l] -> contract[v] -> clear();

					for (int i = 0; i < size; ++i) {
						int w = level[l] -> phi[v][i];

						level[l] -> X_mul_f[v][w] -> setParameter(level[l] -> X[v][w], level[l - 1] -> f[w]);
						graph -> add(level[l] -> X_mul_f[v][w], MATTENSORMUL);

						level[l] -> quadratic[v][w] -> setParameter(level[l] -> X_mul_f[v][w], level[l] -> X_transpose[v][w]);
						graph -> add(level[l] -> quadratic[v][w], TENSORMATMUL);

						level[l] -> contract[v] -> add_tensor(level[l] -> quadratic[v][w]);
					}

					graph -> add(level[l] -> contract[v], RISICONTRACTION_4);

					// Cut the number of chanels
					level[l] -> reshape2D[v] -> setParameter(level[l] -> contract[v], size * size, nContractions * prevC);
					graph -> add(level[l] -> reshape2D[v], RESHAPE2D);

					level[l] -> represent[v] -> setParameter(level[l] -> reshape2D[v], level[l] -> K);
					graph -> add(level[l] -> represent[v], MATMUL);

					level[l] -> reshape3D[v] -> setParameter(level[l] -> represent[v], size, size, C);
					graph -> add(level[l] -> reshape3D[v], RESHAPE3D);

					// Add the bias
					level[l] -> add[v] -> setParameter(level[l] -> b, level[l] -> reshape3D[v]);
					graph -> add(level[l] -> add[v], VECTORADDTENSOR);

					// Non-linearity
					level[l] -> f[v] -> setParameter(level[l] -> add[v]);
					graph -> add(level[l] -> f[v], LEAKYRELU3D);
				}
			}
		}

		// On top of everything
		for (int l = 0; l <= nLevels; ++l) {
			level_feature[l] -> clear();

			for (int v = 0; v < molecule -> nVertices; ++v) {
				shrinked[l][v] -> setParameter(level[l] -> f[v]);
				graph -> add(shrinked[l][v], SHRINKTENSOR);

				vertex_feature[l][v] -> setParameter(shrinked[l][v]);
				graph -> add(vertex_feature[l][v], LEAKYRELU);

				level_feature[l] -> add_vector(vertex_feature[l][v]);
			}

			graph -> add(level_feature[l], SUMVECTORS);
		}
	}

	void complete_computation_graph(DenseGraph *molecule_1, DenseGraph *molecule_2) {
		assert(molecule_1 -> nFeatures == nFeatures_1);
		assert(molecule_1 -> nVertices <= max_nVertices_1);
		assert(molecule_2 -> nFeatures == nFeatures_2);
		assert(molecule_2 -> nVertices <= max_nVertices_2);

		// Update the adjacency matrix
		update_adjacency(molecule_1, adj_1);
		update_adjacency(molecule_2, adj_2);

		// Update the feature (with dummy nodes)
		update_feature(molecule_1, feature_1);
		update_feature(molecule_2, feature_2);

		// Finding the shortest-paths by Floyd-Warshall algorithm
		floyd_warshall(molecule_1, adj_1, shortest_paths_1);
		floyd_warshall(molecule_2, adj_2, shortest_paths_2);

		// Initialize the receptive fields and permutation matrices
		init_receptive_field_permutation_matrix(molecule_1, level_1, shortest_paths_1);
		init_receptive_field_permutation_matrix(molecule_2, level_2, shortest_paths_2);

		// Constructing the dynamic computation graph
		graph -> clear();
		graph -> add(H_1, MATRIX);
		graph -> add(H_2, MATRIX);
		for (int l = 1; l <= nLevels; ++l) {
			// Graph 1
			graph -> add(level_1[l] -> K, MATRIX);
			graph -> add(level_1[l] -> b, VECTOR);

			// Graph 2
			graph -> add(level_2[l] -> K, MATRIX);
			graph -> add(level_2[l] -> b, VECTOR);
		}
		graph -> add(W1, MATRIX);
		graph -> add(W2, MATRIX);
		graph -> add(W3, VECTOR);

		// Complete computation graph - original graph
		complete_computation_graph_(molecule_1, level_1, H_1, feature_1, shortest_paths_1, shrinked_1, vertex_feature_1, level_feature_1);
		
		// Complete computation graph - original graph
		complete_computation_graph_(molecule_2, level_2, H_2, feature_2, shortest_paths_2, shrinked_2, vertex_feature_2, level_feature_2);

		// Concatenate every level feature
		graph_feature -> clear();

		for (int l = 0; l <= nLevels; ++l) {
			graph_feature -> add_vector(level_feature_1[l]);
			graph_feature -> add_vector(level_feature_2[l]);
		}

		graph -> add(graph_feature, CONCATVECTORS);
		
		// Fully-connected layers
		hidden_1 -> setParameter(W1, graph_feature);
		graph -> add(hidden_1, MATVECMUL);

		hidden_relu_1 -> setParameter(hidden_1);
		graph -> add(hidden_relu_1, LEAKYRELU);

		hidden_2 -> setParameter(W2, hidden_relu_1);
		graph -> add(hidden_2, MATVECMUL);

		hidden_relu_2 -> setParameter(hidden_2);
		graph -> add(hidden_relu_2, LEAKYRELU);

		predict -> setParameter(hidden_relu_2, W3);
		graph -> add(predict, INNERPRODUCT);

		sql -> setParameter(predict, target);
		graph -> add(sql, SQUAREDLOSS);
	}

	double getLoss(int nBatch, DenseGraph **molecule_1, DenseGraph **molecule_2, double *target) {
		double total_loss = 0.0;
		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule_1[i], molecule_2[i]);
			this -> target -> value[0] = target[i];
			graph -> forward();
			total_loss += sql -> getLoss();
		}
		return total_loss;
	}

	// +-------------------------+
	// | Multi-threading (Begin) |
	// +-------------------------+

	void copy_value(Adam *from, Adam *to) {
		assert(from -> params.size() == to -> params.size());

		for (int i = 0; i < from -> params.size(); ++i) {
			assert(from -> params[i] -> size == to -> params[i] -> size);

			for (int v = 0; v < from -> params[i] -> size; ++v) {
				to -> params[i] -> value[v] = from -> params[i] -> value[v];
			}
		}
	}

	void clean_gradient(Adam *sgd) {
		for (int i = 0; i < sgd -> params.size(); ++i) {
			for (int v = 0; v < sgd -> params[i] -> size; ++v) {
				sgd -> params[i] -> gradient[v] = 0.0;
			}
		}
	}

	void add_gradient(Adam *from, Adam *to) {
		assert(from -> params.size() == to -> params.size());

		for (int i = 0; i < from -> params.size(); ++i) {
			assert(from -> params[i] -> size == to -> params[i] -> size);

			for (int v = 0; v < from -> params[i] -> size; ++v) {
				to -> params[i] -> gradient[v] += from -> params[i] -> gradient[v];
			}
		}
	}

	static void compute_gradient_job(SMP_gamma_pairgraphs *instance, 
		int sample, DenseGraph **molecule_1, DenseGraph **molecule_2, double *target, double *loss) {
		
		// Complete the computation graph
		instance -> complete_computation_graph(molecule_1[sample], molecule_2[sample]);
		
		// Learning target
		instance -> target -> value[0] = target[sample];

		// Forward pass
		instance -> graph -> forward();

		// Save the loss value
		if (loss != NULL) {
			loss[sample] = instance -> sql -> getLoss();
		}

		// Backward pass
		instance -> graph -> backward();
	}

	void Threaded_ComputeGradient(int nBatch, DenseGraph **molecule_1, DenseGraph **molecule_2, double *target, double *loss) {
		assert(multi_threaded == true);
		assert(nThreads > 1);

		assert(nBatch > 0);
		for (int i = 0; i < nBatch; ++i) {
			assert(molecule_1[i] -> nVertices <= max_nVertices_1);
			assert(molecule_1[i] -> nFeatures == nFeatures_1);

			assert(molecule_2[i] -> nVertices <= max_nVertices_2);
			assert(molecule_2[i] -> nFeatures == nFeatures_2);
		}

		clean_gradient(sgd);

		int start = 0;
		while (start < nBatch) {
			int finish = start + nThreads - 1;
			if (finish >= nBatch) {
				finish = nBatch - 1;
			}

			int nRuns = finish - start + 1;

			for (int t = 0; t < nRuns; ++t) {
				copy_value(sgd, instance[t] -> sgd);
			}

			for (int i = start; i <= finish; ++i) {
				int t = i - start;
				job[t] = std::thread(compute_gradient_job, instance[t], i, molecule_1, molecule_2, target, loss);
			}

			for (int t = 0; t < nRuns; ++t) {
				job[t].join();
			}

			for (int t = 0; t < nRuns; ++t) {
				add_gradient(instance[t] -> sgd, sgd);
			}

			start = finish + 1;
		}
	}

	void Threaded_BatchLearn(int nBatch, DenseGraph **molecule_1, DenseGraph **molecule_2, double *target, double learning_rate) {
		// Gradient computation
		Threaded_ComputeGradient(nBatch, molecule_1, molecule_2, target, NULL);

		// Update the learning parameters
		sgd -> Learn(learning_rate, nBatch);
	}

	void Threaded_BatchLearn(int nBatch, DenseGraph **molecule_1, DenseGraph **molecule_2, double *target, double *loss, double learning_rate) {
		// Gradient computation
		Threaded_ComputeGradient(nBatch, molecule_1, molecule_2, target, loss);

		// Update the learning parameters
		sgd -> Learn(learning_rate, nBatch);
	}

	// +-----------------------+
	// | Multi-threading (End) |
	// +-----------------------+

	pair < double, double > BatchLearn(int nBatch, DenseGraph **molecule_1, DenseGraph **molecule_2, double *target, double learning_rate) {
		assert(nBatch > 0);
		for (int i = 0; i < nBatch; ++i) {
			assert(molecule_1[i] -> nVertices <= max_nVertices_1);
			assert(molecule_1[i] -> nFeatures == nFeatures_1);

			assert(molecule_2[i] -> nVertices <= max_nVertices_2);
			assert(molecule_2[i] -> nFeatures == nFeatures_2);
		}

		pair < double, double > ret;
		ret.first = getLoss(nBatch, molecule_1, molecule_2, target);

		sum_gradients -> reset_sum_gradients();

		for (int i = 0; i < nBatch; ++i) {
			complete_computation_graph(molecule_1[i], molecule_2[i]);
			this -> target -> value[0] = target[i];

			graph -> forward();
			graph -> backward();

			sum_gradients -> cache_gradients();
		}

		sum_gradients -> get_sum_gradients();
		sgd -> Learn(learning_rate, nBatch);

		ret.second = getLoss(nBatch, molecule_1, molecule_2, target);
		return ret;
	}

	pair < double, double > BatchLearn(int nBatch, DenseGraph **molecule_1, DenseGraph **molecule_2, double *target, int nIterations, double learning_rate, double epsilon) {
		assert(nBatch > 0);
		for (int i = 0; i < nBatch; ++i) {
			assert(molecule_1[i] -> nVertices <= max_nVertices_1);
			assert(molecule_1[i] -> nFeatures == nFeatures_1);

			assert(molecule_2[i] -> nVertices <= max_nVertices_2);
			assert(molecule_2[i] -> nFeatures == nFeatures_2);
		}

		cache_parameters -> cache_parameters();

		pair<double, double> ret;
		ret.first = getLoss(nBatch, molecule_1, molecule_2, target);
		ret.second = ret.first;

		double decay_lr = 0.5;
		double min_lr = 1e-6;

		for (int iter = 0; iter < nIterations; ++iter) {
			sum_gradients -> reset_sum_gradients();

			for (int i = 0; i < nBatch; ++i) {
				complete_computation_graph(molecule_1[i], molecule_2[i]);
				this -> target -> value[0] = target[i];

				graph -> forward();
				graph -> backward();

				sum_gradients -> cache_gradients();
			}

			sum_gradients -> get_sum_gradients();
			sgd -> Learn(learning_rate, nBatch);

			double loss = getLoss(nBatch, molecule_1, molecule_2, target);

			if (loss > ret.second) {
				cache_parameters -> restore_parameters();
				learning_rate *= decay_lr;
				if (learning_rate < min_lr) {
					break;
				}
			} else {
				ret.second = loss;
				cache_parameters -> cache_parameters();
			}
		}

		return ret;
	}

	pair < double, double > Learn(DenseGraph *molecule_1, DenseGraph *molecule_2, double target, int nIterations, double learning_rate, double epsilon) {
		assert(molecule_1 -> nVertices <= max_nVertices_1);
		assert(molecule_1 -> nFeatures == nFeatures_1);
		assert(molecule_2 -> nVertices <= max_nVertices_2);
		assert(molecule_2 -> nFeatures == nFeatures_2);

		complete_computation_graph(molecule_1, molecule_2);
		this -> target -> value[0] = target;

		cache_parameters -> cache_parameters();

		graph -> forward();
		double best_error = sql -> getLoss();

		if (best_error < epsilon) {
			return make_pair(best_error, best_error);
		}

		pair<double, double> ret;
		ret.first = best_error;

		double decay_lr = 0.5;
		double min_lr = 1e-6;

		for (int iter = 0; iter < nIterations; ++iter) {
			graph -> forward();
			graph -> backward();
			sgd -> Learn(learning_rate);

			graph -> forward();
			double error = sql -> getLoss();

			if (error < epsilon) {
				break;
			}

			if (error >= best_error) {
				cache_parameters -> restore_parameters();
				learning_rate *= decay_lr;
				learning_rate = max(learning_rate, min_lr);
			} else {
				best_error = error;
				cache_parameters -> cache_parameters();
			}
		}

		ret.second = best_error;
		return ret;
	}

	// +------------------------+
	// | Multi-threaded (Begin) |
	// +------------------------+

	static void predict_job(SMP_gamma_pairgraphs *instance, DenseGraph *molecule_1, DenseGraph *molecule_2, double *predict, int position) {
		instance -> complete_computation_graph(molecule_1, molecule_2);
		instance -> graph -> forward();
		predict[position] = instance -> predict -> value[0];
	}

	void Threaded_Predict(int nBatch, DenseGraph **molecule_1, DenseGraph **molecule_2, double *predict) {
		assert(multi_threaded == true);
		assert(nThreads > 1);

		assert(nBatch > 0);
		for (int i = 0; i < nBatch; ++i) {
			assert(molecule_1[i] -> nVertices <= max_nVertices_1);
			assert(molecule_1[i] -> nFeatures == nFeatures_1);
			assert(molecule_2[i] -> nVertices <= max_nVertices_2);
			assert(molecule_2[i] -> nFeatures == nFeatures_2);
		}

		for (int t = 0; t < nThreads; ++t) {
			copy_value(sgd, instance[t] -> sgd);
		}

		int start = 0;
		while (start < nBatch) {
			int finish = start + nThreads - 1;
			if (finish >= nBatch) {
				finish = nBatch - 1;
			}

			int nRuns = finish - start + 1;

			for (int i = start; i <= finish; ++i) {
				int t = i - start;
				job[t] = std::thread(predict_job, instance[t], molecule_1[i], molecule_2[i], predict, i);
			}

			for (int t = 0; t < nRuns; ++t) {
				job[t].join();
			}

			start = finish + 1;
		}
	}

	// +----------------------+
	// | Multi-threaded (End) |
	// +----------------------+

	double Predict(DenseGraph *molecule_1, DenseGraph *molecule_2) {
		assert(molecule_1 -> nVertices <= max_nVertices_1);
		assert(molecule_1 -> nFeatures == nFeatures_1);
		assert(molecule_2 -> nVertices <= max_nVertices_2);
		assert(molecule_2 -> nFeatures == nFeatures_2);

		complete_computation_graph(molecule_1, molecule_2);

		graph -> forward();

		return predict -> value[0];
	}

	vector<double> Feature(DenseGraph *molecule_1, DenseGraph *molecule_2) {
		assert(molecule_1 -> nVertices <= max_nVertices_1);
		assert(molecule_1 -> nFeatures == nFeatures_1);
		assert(molecule_2 -> nVertices <= max_nVertices_2);
		assert(molecule_2 -> nFeatures == nFeatures_2);

		complete_computation_graph(molecule_1, molecule_2);

		graph -> forward();

		vector<double> vect;
		vect.clear();
		for (int i = 0; i < graph_feature -> size; ++i) {
			vect.push_back(graph_feature -> value[i]);
		}
		return vect;
	}

	void save_model(string filename) {
		ofstream file(filename.c_str(), ios::out);

		for (int i = 0; i < sgd -> params.size(); ++i) {
			for (int j = 0; j < sgd -> params[i] -> size; ++j) {
				file << sgd -> params[i] -> value[j] << " ";
			}
		}

		file.close();
	}

	void load_model(string filename) {
		ifstream file(filename.c_str(), ios::in);
		
		for (int i = 0; i < sgd -> params.size(); ++i) {
			for (int j = 0; j < sgd -> params[i] -> size; ++j) {
				file >> sgd -> params[i] -> value[j];
			}
		}

		file.close();
	}

	// Dynamic computation graph
	GraphFlow *graph;

	// Number of contractions in this class
	static const int nContractions = RisiContraction_4::nContractions;

	// Infinity
	static const int INF = 1e9;

	// Number of threads
	int nThreads;

	// Multi-threading mode
	bool multi_threaded;

	// Instances for multi-threads
	SMP_gamma_pairgraphs **instance;

	// Multi-threaded jobs
	std::thread *job;

	// Original graph features
	Matrix **feature_1;
	Matrix **feature_2;

	// Mapping from the original graph features into chanels
	Matrix *H_1;
	Matrix *H_2;

	// Multiple levels
	Level **level_1;
	Level **level_2;	

	// Shrink from tensors to vectors
	ShrinkTensor ***shrinked_1;
	ShrinkTensor ***shrinked_2;

	// Vertex features
	LeakyReLU ***vertex_feature_1;
	LeakyReLU ***vertex_feature_2;

	// Level feature
	SumVectors **level_feature_1;
	SumVectors **level_feature_2;

	// Graph feature
	ConcatVectors *graph_feature;

	// Fully-connected layer
	Matrix *W1;
	Matrix *W2;
	Vector *W3;

	MatVecMul *hidden_1;
	LeakyReLU *hidden_relu_1;

	MatVecMul *hidden_2;
	LeakyReLU *hidden_relu_2;

	// Prediction
	InnerProduct *predict;

	// Target
	Vector *target;

	// Squared loss
	SquaredLoss *sql;

	// Stochastic Gradient Descent
	Adam *sgd;

	// Sum gradients
	SumGradients *sum_gradients;

	// Cache parameters
	CacheParameters *cache_parameters;

	// Maximum number of vertices
	int max_nVertices_1;
	int max_nVertices_2;

	// Maximum size of the receptive field
	int max_receptive_field;

	// Number of levels
	int nLevels;

	// Number of chanels
	int nChanels;

	// Number of original vertex features
	int nFeatures_1;
	int nFeatures_2;

	// Adjacency matrix
	int **adj_1;
	int **adj_2;

	// Floyd-Warshall algorithm
	int **shortest_paths_1;
	int **shortest_paths_2;

	~SMP_gamma_pairgraphs() {
		delete[] feature_1;
		delete[] feature_2;
		for (int l = 0; l <= nLevels; ++l) {
			if (l == 0) {
				delete[] level_1[l] -> matmul;
				delete[] level_1[l] -> f_reshape;
				delete[] level_1[l] -> f;

				delete[] level_2[l] -> matmul;
				delete[] level_2[l] -> f_reshape;
				delete[] level_2[l] -> f;
			} else {
				delete[] level_1[l] -> f;
				delete[] level_1[l] -> contract;
				delete[] level_1[l] -> reshape2D;
				delete[] level_1[l] -> represent;
				delete[] level_1[l] -> reshape3D;
				delete[] level_1[l] -> add;
				delete[] level_1[l] -> phi;
				delete[] level_1[l] -> quadratic;
				delete[] level_1[l] -> X_mul_f;
				delete[] level_1[l] -> X;
				delete[] level_1[l] -> X_transpose;

				delete[] level_2[l] -> f;
				delete[] level_2[l] -> contract;
				delete[] level_2[l] -> reshape2D;
				delete[] level_2[l] -> represent;
				delete[] level_2[l] -> reshape3D;
				delete[] level_2[l] -> add;
				delete[] level_2[l] -> phi;
				delete[] level_2[l] -> quadratic;
				delete[] level_2[l] -> X_mul_f;
				delete[] level_2[l] -> X;
				delete[] level_2[l] -> X_transpose;
			}
			delete[] shrinked_1[l];
			delete[] vertex_feature_1[l];

			delete[] shrinked_2[l];
			delete[] vertex_feature_2[l];
		}
		delete graph_feature;
		delete W1;
		delete W2;
		delete W3;
		delete hidden_1;
		delete hidden_relu_1;
		delete hidden_2;
		delete hidden_relu_2;
		delete predict;
		delete target;
		delete sql;
		delete sgd;
		delete sum_gradients;
		delete cache_parameters;
		delete[] adj_1;
		delete[] shortest_paths_1;
		delete[] adj_2;
		delete[] shortest_paths_2;
	}
};

#endif