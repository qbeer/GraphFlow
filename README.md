# GraphFlow modification for CHAMPS challenge

![](https://travis-ci.com/qbeer/GraphFlow.svg?branch=feature/gpu)

- Travis-CI
- Molecular structure

To do learning with a second order model do:

```bash
git clone https://github.com/qbeer/GraphFlow
cd GraphFlow
cd tests/ && g++ -std=c++11 -pthread test_SMP_theta_physics.cpp
./a.out
```

To use the GPU the model available currently can be compiled as:

```bash
git clone https://github.com/qbeer/GraphFlow
cd GraphFlow
cd tests/ && nvcc -std=c++11 -o executable test_SMP_beta_gpu_multistreams.cu
./executable
```

**ATTENTION** : the model above expects the size of adjecency matrices in directory `kaggle_utils/molecules/bonds/`
and `kaggle_utils/molecules/labels/` to be divisible by 10!

The predictions for the training set are outputted to directore `tests/predictions/`.
