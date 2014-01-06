LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mic/filesystem/base/lib64/
export LD_LIBRARY_PATH

source /opt/intel/composerxe/bin/compilervars.sh intel64

export MIC_ENV_PREFIX=MIC
export MIC_PREFIX=MIC
export MIC_OMP_NUM_THREADS=240
# 
export MKL_NUM_THREADS=240
export OMP_NUM_THREADS=1
export OFFLOAD_REPORT=1
# either 0 or 1
export OFFLOAD_DEVICES="0,1"

export SINK_LD_LIBRARY_PATH=/opt/intel/composerxe/lib/mic:/usr/linux-k1om-4.7/x86_64-k1om-linux/lib64

PATH=$PATH:/opt/intel/mic/bin:/usr/linux-k1om-4.7/bin/

# CC x86_64-k1om-linux-gcc

export MIC_KMP_AFFINITY='disabled'
export KMP_AFFINITY='disabled'

export MIC_USE_2MB_BUFFERS=32K

# OMP version not better than pthread version
