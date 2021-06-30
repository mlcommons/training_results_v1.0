#!ignore
HOSTS=10.129.96.118,10.129.96.122 #only two hosts are enough but confirmed working with 8 separate as well
VIPU_SERVER_HOST=10.129.96.118

export POPLAR_ENGINE_OPTIONS='{"target.gatewayMode": "true", "target.syncReplicasIndependently": "true", "target.hostSyncTimeout": "600"}' 

rm -f session_cache/* 

source  /home/custeng-scratch/shengf/sdk/poplar/enable.sh 
source  /home/custeng-scratch/shengf/sdk/popart/enable.sh
source venv/bin/activate

poprun -vv --num-instances 2 --num-replicas 2 --num-ilds=1 --ipus-per-replica 4 --numa-aware=yes --host 10.129.96.118 --vipu-server-host=10.129.96.118 --vipu-server-timeout=600 --vipu-partition=gcl128 --vipu-cluster=2xPOD64 --reset-partition=no --update-partition=yes --remove-partition=0  --mpi-global-args="--tag-output  --allow-run-as-root  --mca oob_tcp_if_include 10.129.96.118/24 --mca btl_tcp_if_include 10.129.96.118/24" --mpi-local-args="-x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x OMPI_CPPFLAGS -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=INFO -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x TMP=/localdata/$USER/tmp -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS -x GCL_LOG_LEVEL=INFO" python3 ./bert.py --config configs/mk2/mlperf_mini_poprun.json

poprun -vv --num-instances 2 --num-replicas 32 --num-ilds=2 --ipus-per-replica 4 --numa-aware=yes --host 10.129.96.118,10.129.96.122 --vipu-server-host=10.129.96.118 --vipu-server-timeout=600 --vipu-partition=gcl128 --vipu-cluster=2xPOD64 --reset-partition=yes --update-partition=yes --remove-partition=1  --mpi-global-args="--tag-output  --allow-run-as-root  --mca oob_tcp_if_include 10.129.96.118/24 --mca btl_tcp_if_include 10.129.96.118/24" --mpi-local-args="-x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x OMPI_CPPFLAGS -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=OFF -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x TMP=/localdata/$USER/tmp -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS -x GCL_LOG_LEVEL=OFF" python3 ./bert.py --config configs/mk2/pod128.json >& pod128.run
