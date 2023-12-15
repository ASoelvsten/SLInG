import GIBBS
import numpy as np

# This file gives an overview of how to run SLInG.
# Running SLinG can be done with one line of code. However, for a better overview, we define the individual settings one by one.

# 1) What data model and data set are we using? You need to define these in the scripts modellib.py and GIBBS.py.

name =  "Diabetes"

# 2) Where do you want to stor your output?

direc = "./EXAMPLE_01/"

# 3) What distance measure do you want to use?

disttype = "RMSD"

# 4) How many samples do you want to include in your final production run?

sam = 10000

# Note that SLInG allows you to make a few explorations of the parameter space to iteratively set dabc_min to the value you request below.
# You can easily do so when you have to pass an initial guess for the stepsize. For instance, if you pass an array with 3 values, SLInG will run three iterations:
# stepsize = [1.,1.,1.]
# Let's just skip the adjustment of dabc_min here:

stepsize = [1.]

# 5) Define your uniform priors that define the space that you want to explore:

truncation = [-100.*np.ones(10),100.*np.ones(10)]

# 6) Decide which of the parameters also need a sparsity prior. 

sparsity_prior = np.ones(10,dtype=bool) # Here, all get a sparsity prior.

# What type of sparsity prior?

ptype = "Laplace"

# 7) What value of dabc_min would you like to try?

dabc_min = 0.02

# 8) If you were to know the ground truth you could define it here for comparison in the automated plots. If you don't know the groundtruth, just set a dummy array: 

theta_obs =  np.zeros(10)

# CALL SLInG

GIBBS.chain(name, theta_obs, truncation, direc=direc, sparsity_prior=sparsity_prior, dabc_min=dabc_min, spls = [], dabc_scale=35, ptype=ptype, disttype = disttype, burnin = 100, samples = sam, samples_scouts = 500, stepsize = stepsize)
