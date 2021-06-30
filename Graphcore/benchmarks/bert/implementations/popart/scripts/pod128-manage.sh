$(which mpirun) --tag-output --host 10.129.96.118,10.129.96.122 hostname

poprun -vv --host 10.1.67.150 --num-instances=1 --num-replicas=4 --vipu-server-host=10.1.67.150 --vipu-partition=lr67_64ipu_reconfig --num-ilds=1 hostname

AGENTS1="ag.13.01,ag.13.02,ag.13.03,ag.13.04,ag.13.05,ag.13.06,ag.13.07,ag.13.08,ag.13.09,ag.13.10,ag.13.11,ag.13.12,ag.13.13,ag.13.14,ag.13.15,ag.13.16"
AGENTS2="ag.14.01,ag.14.02,ag.14.03,ag.14.04,ag.14.05,ag.14.06,ag.14.07,ag.14.08,ag.14.09,ag.14.14,ag.14.11,ag.14.12,ag.14.13,ag.14.14,ag.14.15,ag.14.16"
AGENTS=$AGENTS1,$AGENTS2
vipu-admin -H 10.129.96.118 create cluster 2xPOD64 --num-ipulinkdomains 2 --topology torus --cluster-topology looped --agents "${AGENTS}"

vipu-admin -H 10.129.96.118 create partition gcl128 --size 128 --num-gcds 32 --gcd-sync-replicas 32 --routing ringswnc --cluster 2xPOD64

# reset ipum
pod_utils/scripts/power_control/ipum-reset-parallel.sh 13 14 1 16

# copy-ssh-keys-multipod.ssh checks if the number of parameters is 2, it should be changed to 4
pod_utils/scripts/virm_vipu/copy-ssh-keys-multipod.ssh 13 14 1 16

parallel --tag -kM scp gwlinks itadmin@10.2.{1}.{2}:. ::: {13..14} ::: {1..16}

parallel --tag -kM ssh itadmin@10.2.{1}.{2} 'sudo ./gwlinks link_enable' ::: {13..14} ::: {1..16}

vipu-admin -H 10.129.96.118 create partition gcl64-pod14 --cluster 2xPOD64 --size=64 --reconfigurable

vipu-admin -H 10.129.96.118 get partition gcl64-pod14  --gcd 0 --ipuof-configs > partition-pod14.conf 

export IPUOF_CONFIG_PATH="~/ipu-config/partition-pod14.conf"