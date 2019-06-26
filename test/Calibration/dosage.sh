# Before running this, untar sm.ms.tar and build sagecal
# this command assumes sagecal binary is at ../src/MS/sagecal
../src/MS/sagecal -d sm.ms -s 3c196.sky.txt -c 3c196.sky.txt.cluster -n 4 -t 10 -p sm.ms.solutions -e 4 -g 2 -l 10 -m 7 -x 30 -F 1 -j 2  -k -1 -B 1 -W 0 > sm.ms.output
