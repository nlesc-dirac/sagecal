User manual
===========

[hspreeuw@node503 bin]$ ./sagecal_gpu -h
SAGECal 0.7.0 (C) 2011-2020 Sarod Yatawatta
Usage:
sagecal -d MS -s sky.txt -c cluster.txt
or
sagecal -f MSlist -s sky.txt -c cluster.txt

Stochastic calibration:
sagecal -d MS -s sky.txt -c cluster.txt -C epochs -M minibatches

- -d MS name. This is the observation of the target, which has to be in the format of a `Measurement Set`_.
- -f MSlist: text file with names of Measurement Sets.
- -s sky.txt: sky model file
- -c cluster.txt: cluster file
- -p solutions.txt: if given, save solution in this file, or read the solutions if doing simulations
- -F sky model format: 0: LSM, 1: LSM with 3 order spectra : default 1
- -I input column in the Measurement Set (DATA/CORRECTED_DATA/...) : default DATA
- -O ouput column in the Measurement Set (DATA/CORRECTED_DATA/...) : default CORRECTED_DATA
- -e maximum number of expectation maximization iterations : default 3
- -g maximum number of iterations within a single expectation maximization step : default 2
- -l maximum number of iterations for the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS) algorithm : default 10
- -m LBFGS memory size, see the `Wikipedia page on LBFGS`_ where the same symbol m is used for the past m updates of the position and gradient vectors: default 7 updates kept in history.
- -n number of worker threads on the CPU: default 6
- -t tile size, this means the number of time samples than constitute a solution interval : default 120, so that would be two minutes for visibility sampling at one second.
- -a 0,1,2,3 : These are the simulation options, so just converting a sky model to its corresponding visibilities at the (u, v, w) triples of the observation without calibrating. if 1, only simulate, if 2, simulate and add to input, if 3, simulate and subtract from input (For a>0, multiplied by solutions if solutions file is also given): default 0, which means no simulation. 
- -z ignore_clusters: if only doing a simulation, ignore the cluster ids listed in this file
- -b 0,1 : if 1, solve for each channel: default 0
- -B 0,1 : if 1, predict array beam, which means that SAGECal will compute and take account of the sensitivity profile of the telescope for this particular observation, i.e. the array beam: default 0
- -E 0,1 : if 1, use GPU for model computing, i.e. for converting a sky model to its corresponding visibilities at the (u, v, w) triples of the observation: default 0
- -x exclude baselines length (lambda) lower than this in calibration : default 0
- -y exclude baselines length (lambda) higher than this in calibration : default 1e+08
 
Advanced options:

- -k cluster_id : correct residuals with solution of this cluster : default -99999
- -o rho, for robust matrix inversion during correction: default 1e-09. Rho is a small value to make sure that the inverse of J does not blow up when J is singular (inverse J+rho*I).
- -J 0,1 : if >0, use phase only correction: default 0
- | -j 0,1,2... 0 : OSaccel, 1 no OSaccel, 2: OSRLM, 3: RLM, 4: RTR, 5: RRTR, 6:NSD : default 5
  | * OSaccel: Ordered Subsets acceleration
  | * OSRLM: Ordered Subsets accelerated Robust Levenberg Marquardt
  | * RLM: Robust Levenberg Marquardt
  | * RTR: Riemannian Trust Region
  | * RRTR: Robust Riemannian Trust Region
  | * NSD: Nesterov's Steepest Descent

- -L Lower bound for nu, a parameter in the robust noise model: default 2. 
- -H Upper bound for nu: a parameter in the robust noise model: default 30
- -W pre-whiten data: default 0
- -R randomize iterations: default 1
- -S GPU heap size (MB): default 32
- -D 0,1,2 : if >0, enable diagnostics (Jacobian Leverage) 1 replace Jacobian Leverage as output, 2 only fractional noise/leverage is printed: default 0
- -q solutions.txt: if given, initialize solutions by reading this file (need to have the same format as a solution file, only solutions for 1 timeslot needed)
 
  | Stochastic mode:
  | -C epochs, if >0, use stochastic calibration: default 0
  | -M minibatches, must be >0, split data to this many minibatches: default 1
  | -w mini-bands, must be >0, split channels to this many mini-bands for bandpass calibration: default 1
  |
  | Stochastic mode with consensus:
  | -A ADMM iterations: default 1
  | -P consensus polynomial terms: default 2
  | -Q consensus polynomial type (0,1,2,3): default 2
  | -r regularization factor: default 5

| Note: 
| - In stochastic mode, no hybrid solutions are allowed.
| - All clusters should have 1 in the second column of cluster file.
| - Report bugs at https://github.com/nlesc-dirac/sagecal/issues.

.. _`Measurement Set`: https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/measurement-set
.. _`Wikipedia page on LBFGS`: https://en.wikipedia.org/wiki/Limited-memory_BFGS
