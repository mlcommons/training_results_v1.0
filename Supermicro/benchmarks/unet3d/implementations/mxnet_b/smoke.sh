set -e

echo "DALI 8 GPU normal"
horovodrun -np 8 python main.py --amp --data_dir /data/kits19_ncdhw --epochs 1 -v --benchmark -sgs 1 --eval_every 1 --start_eval_at 0 -ibm 2

echo "DALI 8 GPU normal static warmup"
horovodrun -np 8 python main.py --static_cast -sls 32768 --data_dir /data/kits19_ncdhw --epochs 1 -v --benchmark -sgs 1 --eval_every 1 --start_eval_at 0 -ibm 2 --batch_size 7 --warmup

echo "DALI 8 GPU spatial SL"
horovodrun -np 8 python main.py --amp --data_dir /data/kits19_ncdhw --epochs 1 -v --benchmark -sgs 2 --eval_every 1 --start_eval_at 0 -ucl -ibm 2

echo "DALI 8 GPU spatial static warmup"
horovodrun -np 8 python main.py --static_cast -sls 32768 --data_dir /data/kits19_ncdhw --epochs 1 -v --benchmark -sgs 2 --eval_every 1 --start_eval_at 0 -ibm 2 --warmup

echo "DALI 8 GPU normal eval cache"
horovodrun -np 8 python main.py --amp --data_dir /data/kits19_ncdhw -v -sgs 1 --val_batch_size 4 --exec_mode evaluate --load_ckpt_path /results/checkpoints/normal -ced
echo "Expected L1 dice 0.9199529; L2 dice 0.69763553; mean_dice 0.8087942600250244"

echo "DALI 8 GPU spatial eval cache"
horovodrun -np 8 python main.py --amp --data_dir /data/kits19_ncdhw -v -sgs 2 --exec_mode evaluate --load_ckpt_path /results/checkpoints/spatial -ucl -ibm 2 -ced
echo "Expected L1 dice 0.9455913; L2 dice 0.82252705; mean_dice 0.8840591907501221"

