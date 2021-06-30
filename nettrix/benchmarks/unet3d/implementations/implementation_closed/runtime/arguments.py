import argparse

PARSER = argparse.ArgumentParser(description="UNet-3D")

# Training params
PARSER.add_argument('--exec_mode', dest='exec_mode', choices=['train', 'evaluate'], default='train')
PARSER.add_argument('--benchmark', dest='benchmark', action='store_true', default=False)
PARSER.add_argument('--amp', dest='amp', action='store_true', default=False)
PARSER.add_argument('--static_cast', dest='static_cast', action='store_true', default=False)
PARSER.add_argument('--static_loss_scale', '-sls', dest='static_loss_scale', type=float, default=1.0)
PARSER.add_argument('--loss_scale_inc_cycles', nargs='+', type=int, default=[])
PARSER.add_argument('--grad_predivide_factor', '-gpf', dest='grad_predivide_factor', type=int, default=1)
PARSER.add_argument('--epochs', dest='epochs', type=int, default=1)
PARSER.add_argument('--ga_steps', dest='ga_steps', type=int, default=1)
PARSER.add_argument('--batch_size', dest='batch_size', type=int, default=1)
PARSER.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=1)
PARSER.add_argument('--warmup', dest='warmup', action='store_true', default=False)
PARSER.add_argument('--log_dir', dest='log_dir', type=str)
PARSER.add_argument('--save_ckpt_path', dest='save_ckpt_path', type=str, default="")
PARSER.add_argument('--load_ckpt_path', dest='load_ckpt_path', type=str, default="")
PARSER.add_argument('--verbose', '-v', dest='verbose', action='store_true', default=False)
PARSER.add_argument('--seed', dest='seed', default=-1, type=int)
PARSER.add_argument('--num_workers', dest='num_workers', type=int, default=4)
PARSER.add_argument('--spatial_group_size', '-sgs', dest='spatial_group_size', type=int, default=1)

# Evaluation params
PARSER.add_argument('--evaluate_every', '--eval_every', dest='evaluate_every', type=int, default=20)
PARSER.add_argument('--start_eval_at', dest='start_eval_at', type=int, default=1000)
PARSER.add_argument('--quality_threshold', dest='quality_threshold', type=float, default=0.908)
PARSER.add_argument('--val_batch_size', dest='val_batch_size', type=int, default=1)
PARSER.add_argument('--nodes_for_eval', dest='nodes_for_eval', type=int, default=0)
PARSER.add_argument('--cache_eval_dataset', '-ced', dest='cache_eval_dataset', action='store_true', default=False)

# Optimizer params
PARSER.add_argument('--optimizer', dest='optimizer', type=str, default="nag", choices=["adam", "sgd", "nag", "nadam"])
PARSER.add_argument('--init_learning_rate', dest='init_learning_rate', type=float, default=1e-4)
PARSER.add_argument('--lr_warmup_epochs', dest='lr_warmup_epochs', type=int, default=1000)
PARSER.add_argument('--lr_decay_epochs', nargs='+', type=int, default=[])
PARSER.add_argument('--lr_decay_factor', dest='lr_decay_factor', type=float, default=1.0)
PARSER.add_argument('--learning_rate', dest='learning_rate', type=float, default=1.0)
PARSER.add_argument('--momentum', dest='momentum', type=float, default=0.9)
PARSER.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0)
PARSER.add_argument('--warmup_iters', dest='warmup_iters', type=int, default=20)

# Model params
PARSER.add_argument('--layout', dest='layout', type=str, choices=['NCDHW', 'NDHWC'], default='NDHWC')
PARSER.add_argument('--normalization', dest='normalization', type=str,
                    choices=['instancenorm', 'batchnorm'], default='instancenorm')
PARSER.add_argument('--activation', dest='activation', type=str, choices=['relu', 'leaky_relu'], default='relu')
PARSER.add_argument('--model_dir', dest='model_dir', type=str, default='/workspace/unet3d/model_dir')

# Dataset params
PARSER.add_argument('--data_dir', dest='data_dir', default="/tmp")
PARSER.add_argument('--loader', dest='loader', choices=['dali', 'synthetic', 'mxnet'], default="dali", type=str)
PARSER.add_argument('--input_shape', nargs='+', type=int, default=[128, 128, 128])
PARSER.add_argument('--val_input_shape', nargs='+', type=int, default=[128, 128, 128])


# REMAINING
PARSER.add_argument('--lamb_betas', nargs='+', type=int, default=[0.9, 0.999])
PARSER.add_argument('--pad_mode', dest='pad_mode', default="constant", choices=['constant', 'edge', 'reflect',
                                                                                'median', 'symmetric'], type=str)
PARSER.add_argument('--oversampling', dest='oversampling', type=float, default=0.4)
PARSER.add_argument('--overlap', dest='overlap', type=float, default=0.5)
PARSER.add_argument('--include_background', dest='include_background', action='store_true', default=False)
PARSER.add_argument('--gpu_per_node', dest='gpu_per_node', default=8, type=int)

# DALI
PARSER.add_argument('--prefetch_queue_depth', '-pqd', dest='prefetch_queue_depth', default=1, type=int)
PARSER.add_argument('--dont_use_mmap', '-mmap', dest='dont_use_mmap', action='store_true', default=False)
PARSER.add_argument('--input_batch_multiplier', '-ibm', dest='input_batch_multiplier', default=1, type=int)
PARSER.add_argument('--use_cached_loader', '-ucl', dest='use_cached_loader', action='store_true', default=False)
PARSER.add_argument('--stick_to_shard', '-sts', dest='stick_to_shard', action='store_true', default=False)
