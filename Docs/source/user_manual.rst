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

-d MS name
-f MSlist: text file with MS names
-s sky.txt: sky model file
-c cluster.txt: cluster file
-p solutions.txt: if given, save solution in this file, or read the solutions if doing simulations
-F sky model format: 0: LSM, 1: LSM with 3 order spectra : default 1
-I input column (DATA/CORRECTED_DATA/...) : default DATA
-O ouput column (DATA/CORRECTED_DATA/...) : default CORRECTED_DATA
-e max EM iterations : default 3
-g max iterations  (within single EM) : default 2
-l max LBFGS iterations : default 10
-m LBFGS memory size : default 7
-n no of worker threads : default 6
-t tile size : default 120
-a 0,1,2,3 : if 1, only simulate, if 2, simulate and add to input, if 3, simulate and subtract from input (For a>0, multiplied by solutions if solutions file is also given): default 0
-z ignore_clusters: if only doing a simulation, ignore the cluster ids listed in this file
-b 0,1 : if 1, solve for each channel: default 0
-B 0,1 : if 1, predict array beam: default 0
-E 0,1 : if 1, use GPU for model computing: default 0
-x exclude baselines length (lambda) lower than this in calibration : default 0
-y exclude baselines length (lambda) higher than this in calibration : default 1e+08

Advanced options:
-k cluster_id : correct residuals with solution of this cluster : default -99999
-o robust rho, robust matrix inversion during correction: default 1e-09
-J 0,1 : if >0, use phase only correction: default 0
-j 0,1,2... 0 : OSaccel, 1 no OSaccel, 2: OSRLM, 3: RLM, 4: RTR, 5: RRTR, 6:NSD : default 5
-L robust nu, lower bound: default 2
-H robust nu, upper bound: default 30
-W pre-whiten data: default 0
-R randomize iterations: default 1
-S GPU heap size (MB): default 32
-D 0,1,2 : if >0, enable diagnostics (Jacobian Leverage) 1 replace Jacobian Leverage as output, 2 only fractional noise/leverage is printed: default 0
-q solutions.txt: if given, initialize solutions by reading this file (need to have the same format as a solution file, only solutions for 1 timeslot needed)

Stochastic mode:
-C epochs, if >0, use stochastic calibration: default 0
-M minibatches, must be >0, split data to this many minibatches: default 1
-w mini-bands, must be >0, split channels to this many mini-bands for bandpass calibration: default 1

Stochastic mode with consensus:
-A ADMM iterations: default 1
-P consensus polynomial terms: default 2
-Q consensus polynomial type (0,1,2,3): default 2
-r regularization factor: default 5
Note: In stochastic mode, no hybrid solutions are allowed.
All clusters should have 1 in the second column of cluster file.
Report bugs to <sarod@users.sf.net>

