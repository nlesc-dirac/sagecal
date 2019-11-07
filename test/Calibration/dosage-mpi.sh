# Before running this, make sure you have at least 2 data files with names matching the pattern '*.ms'

# This command assumes sagecal-mpi binary is at ../../build/dist/bin/sagecal-mpi
# or sagecal-mpi_gpu if GPU acceleration is enabled
# 'regularization_factors.txt' should exactly match the cluster info given by '3c196.sky.txt.cluster'
mpirun -np 3 ../../build/dist/bin/sagecal-mpi -f '*.ms' -A 10 -P 2 -Q 2 -G regularization_factors.txt -s 3c196.sky.txt -c 3c196.sky.txt.cluster -p zsol -n 2 -t 10 -e 4 -g 2 -l 10 -m 7 -x 30 -F 1 -j 5 -B 1 -k 1 -K 3 -W 0 -V > sagecal-mpi.ms.output
