# Before running this, untar sm.ms.tar and build sagecal
# this command assumes sagecal binary is at ../src/MS/sagecal
../src/MS/sagecal -d sm.ms -s extended_source_list.txt -c extended_source_list.txt.cluster -n 4 -t 10 -p sm.ms.solutions -e 4 -g 2 -l 10 -m 7 -x 30 -F 1 -j 2  -k -1 -B 1 -W 0 > sm.ms.output
