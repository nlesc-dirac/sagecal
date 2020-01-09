User manual
===========

sagecal
^^^^^^^
| Usage:
|     sagecal -d MS -s sky.txt -c cluster.txt
|     or
|     sagecal -f MSlist -s sky.txt -c cluster.txt
|     
|     Stochastic calibration:
|     sagecal -d MS -s sky.txt -c cluster.txt -N epochs -M minibatches
| 
|     Show all options below:
|     sagecal -h
|     
|     sagecal here denotes the sagecal executable, compiled either for CPU or GPU.

- **-d MS name**. This is the observation of the target, which has to be in the format of a `Measurement Set`_.
- **-f MSlist**. Text file with names of Measurement Sets.
- **-s sky.txt**. Sky model file
- **-c cluster.txt**. Cluster file
- **-p solutions.txt**. If given, save solution in this file, or read the solutions if doing simulations
- **-F sky model format**. 0: LSM, 1: LSM with 3 order spectra. Default: 1.
- **-I input column** in the Measurement Set (DATA/CORRECTED_DATA/...). Default: DATA.
- **-O ouput column** in the Measurement Set (DATA/CORRECTED_DATA/...). Default: CORRECTED_DATA.
- **-e maximum number of expectation maximization iterations**.  Default: 3.
- **-g maximum number of iterations within a single expectation maximization step**. Default 2.
- **-l maximum number of iterations for the LBFGS algorithm**. Default: 10.
- **-m LBFGS memory size**, see the `Wikipedia page on LBFGS`_ where the same symbol m is used for the past m updates of the position and gradient vectors.  Default: 7 updates kept in history.
- **-n number of worker threads** on the CPU. Default: 6.
- **-t tile size**. This means the number of time samples than constitute a solution interval. Default: 120, so that would be two minutes for visibility sampling at one second.
- **-a 0,1,2,3**. These are the simulation options, so just converting a sky model to its corresponding visibilities at the (u, v, w) triples of the observation without calibrating. if 1, only simulate, if 2, simulate and add to input, if 3, simulate and subtract from input (For a>0, multiplied by solutions if solutions file is also given). Default: 0, which means no simulation. 
- **-z ignore_clusters**. If only doing a simulation, ignore the cluster ids listed in this file.
- **-b 0,1**. If 1, solve for each channel. Default: 0.
- **-B 0,1**. If 1, predict array beam, which means that SAGECal will compute and take account of the sensitivity profile of the telescope for this particular observation, i.e. the array beam. Default: 0.
- **-E 0,1**. If 1, use GPU for model computing, i.e. for converting a sky model to its corresponding visibilities at the (u, v, w) triples of the observation. Default: 0.
- **-x exclude baselines length (lambda) lower than** this in calibration. Default: 0.
- **-y exclude baselines length (lambda) higher than** this in calibration. Default: 1e+08.
 
Advanced options:

- **-k cluster_id**. Correct residuals with solution of this cluster. Default: -99999.
- **-o rho**. For robust matrix inversion during correction. Rho is a small value to make sure that the inverse of J does not blow up when J is singular (inverse J+rho*I). Default: 1e-09. 
- **-J 0,1**. If >0, use phase only correction. Default: 0.
- | **-j 0,1,2...**. 0 : OSaccel, 1 no OSaccel, 2: OSRLM, 3: RLM, 4: RTR, 5: RRTR, 6:NSD. Default: 5.
  | *** OSaccel**. Ordered Subsets acceleration.
  | *** OSRLM**. Ordered Subsets accelerated Robust Levenberg Marquardt.
  | *** RLM**. Robust Levenberg Marquardt.
  | *** RTR**. Riemannian Trust Region.
  | *** RRTR**. Robust Riemannian Trust Region.
  | *** NSD**. Nesterov's Steepest Descent.

- **-L Lower bound for nu**, a parameter in the robust noise model. Default: 2. 
- **-H Upper bound for nu**, a parameter in the robust noise model. Default: 30.
- **-W pre-whiten data**. Default: 0. This option has been deprecated.
- **-R randomize iterations**. Default: 1. This option can be used, for instance, when you want to randomize the order of the calibration directions.
- **-S GPU heap size (MB)**. Default: 32.
- **-D 0,1,2**. If >0, enable diagnostics (Jacobian Leverage) 1 replace Jacobian Leverage as output, 2 only fractional noise/leverage is printed. Default: 0.
- **-q solutions.txt**. If given, initialize solutions by reading this file (need to have the same format as a solution file, only solutions for 1 timeslot needed).
 
  | Stochastic mode:
  | **-N epochs**. If >0, use stochastic calibration. Default: 0.
  | **-M minibatches**. Must be >0, split data to this many minibatches. Default: 1.
  | **-w mini-bands**. Must be >0, split channels to this many mini-bands for bandpass calibration. Default: 1.
  |
  | Stochastic mode with consensus:
  | **-A ADMM iterations**. Default: 1.
  | **-P consensus polynomial terms**. Default: 2.
  | **-Q consensus polynomial type** (0,1,2,3). Default: 2.
  | **-r regularization factor**. Default: 5.
  | **-u regularization factor**. Must be >0, regularization in federated averaging between global and local value. Default: 0.1.

| Note: 
| - In stochastic mode, no hybrid solutions are allowed.
| - All clusters should have 1 in the second column of cluster file.
| - Report bugs at https://github.com/nlesc-dirac/sagecal/issues.

.. _`Measurement Set`: https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/measurement-set
.. _`Wikipedia page on LBFGS`: https://en.wikipedia.org/wiki/Limited-memory_BFGS

sagecal-mpi
^^^^^^^^^^^

restore
^^^^^^^

buildsky
^^^^^^^^
