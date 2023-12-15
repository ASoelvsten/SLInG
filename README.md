# Sparse Likelihodd-free Inference using Gibbs, SLInG

If you use SLInG in a publication, please cite our paper: [Bayesian model discovery for reverse-engineering biochemical networks from data](https://www.biorxiv.org/content/10.1101/2023.09.15.557764v1).

To see how to use and run SLInG, see **example_01.py**.

The other scripts are:

**GIBBS.py**: This is SLInG: The Python3 script contains all functions connected to the GIBBS sampler.

**Adaption.py**: This scripts contains models related to adaptive genetic networks.

**simple\_network.py**: This Python3 file calls the Julia scipts for creating genetic networks for the reverse engineering taks. NOTE: This file requires PyJulia and is thus currently not called by GIBBS.py (the corresponding import statement is commented out).

To apply the sampling schemes to other data sets, you need to extend the function **observations** in GIBBS.py. You can define new models by adding these to **modellib.py** or adding new scripts and calling these in GIBBS.py. The function **ParamPrior** invokes a uniform prior. To change the width of the uniform prior in GIBBS.py, change the array "truncation". It's one of the inputs of the main call. The array "truncation" contains the lower and upper limit for each parameter. If you want to go beyond uniform priors, adapt ParamPrior.
