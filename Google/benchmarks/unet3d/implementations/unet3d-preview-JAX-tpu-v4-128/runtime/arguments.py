"""Reference implementation for arguments.

https://github.com/mmarcinkiewicz/training/blob/Add_unet3d/image_segmentation/pytorch/runtime/arguments.py
"""

from absl import flags

flags.DEFINE_string(
    'data_dir', help='data_dir',
    default='/REDACTED/nm-d/home/tpu-perf-team/unet3d/kits19/numpy_data')
flags.DEFINE_string('log_output_dir', help='log_output_dir', default='/tmp')
flags.DEFINE_string('save_ckpt_path', help='save_ckpt_path', default='')
flags.DEFINE_string('load_ckpt_path', help='load_ckpt_path', default='')
flags.DEFINE_float('quality_threshold', help='quality_threshold', default=0.908)

flags.DEFINE_integer('epochs', help='epochs', default=4000)
flags.DEFINE_integer('ga_steps', help='ga_steps', default=1)

flags.DEFINE_integer('warmup_steps', help='warmup_steps', default=4)
flags.DEFINE_integer('batch_size', help='batch_size', default=2)
flags.DEFINE_enum('layout', 'NDHWC', ['NCDHW', 'NDHWC'], 'layout')
flags.DEFINE_multi_integer(
    'input_shape', help='train_input_shape', default=[128, 128, 128, 1])
flags.DEFINE_multi_integer(
    'val_input_shape', help='value_input_shape', default=[128, 128, 128, 1])
flags.DEFINE_integer('seed', help='seed', default=-1)
flags.DEFINE_enum('exec_mode', 'train', ['train', 'evaluate'], 'exec_mode')
flags.DEFINE_bool('benchmark', help='benchmark', default=False)
flags.DEFINE_bool('use_bfloat16', help='use bfloat16 or not', default=False)
flags.DEFINE_enum('optimizer', 'sgd', ['sgd', 'adam', 'lamb'], 'optimizer')
flags.DEFINE_float('learning_rate', help='learning_rate', default=1.0)
flags.DEFINE_float(
    'init_learning_rate', help='init_learning_rate', default=1e-4)
flags.DEFINE_integer('lr_warmup_epochs', help='lr_warmup_epochs', default=-1)
flags.DEFINE_multi_integer(
    'lr_decay_epochs', default=[], help='lr_decay_epochs')
flags.DEFINE_float('lr_decay_factor', help='lr_decay_factor', default=1.0)
flags.DEFINE_multi_float('lamb_betas', help='lamb_betas', default=[0.9, 0.999])
flags.DEFINE_float('momentum', help='momentum', default=0.9)
flags.DEFINE_float('weight_decay', help='weight_decay', default=0.0)
flags.DEFINE_integer('start_eval_at', help='epoch_to_start_eval', default=1000)
flags.DEFINE_integer('evaluate_every', help='eval_every', default=20)
flags.DEFINE_bool('verbose', help='verbose', default=False)
flags.DEFINE_enum('normalization', 'instancenorm',
                  ['instancenorm', 'batchnorm', 'syncbatchnorm'],
                  'normalization')
flags.DEFINE_enum('activation', 'relu', ['relu', 'leaky_relu'], 'activation')
flags.DEFINE_enum('pad_mode', 'reflect',
                  ['constant', 'edge', 'reflect', 'median', 'symmetric'],
                  'pad_mode')
flags.DEFINE_float('oversampling', help='oversampling', default=0.4)
flags.DEFINE_float('overlap', help='overlap', default=0.5)
flags.DEFINE_bool(
    'include_background', help='include_background', default=False)
flags.DEFINE_bool(
    'use_fake_data', help='Use fake or real data', default=False)

flags.DEFINE_integer('num_train_images', help='num train images', default=168)
flags.DEFINE_integer('num_eval_images', help='num_eval_images', default=42)
flags.DEFINE_integer(
    'eval_batch_size', default=2, help=('Batch size to use for eval'))
flags.DEFINE_integer(
    'eval_score_fn_bs', default=2, help=('Batch size to use for eval'))
flags.DEFINE_bool(
    'use_eval_device_loop', help='Use device loop for eval', default=True)
flags.DEFINE_integer(
    'num_eval_passes',
    help='If memory is not big enough to hold all eval images do eval in '
    'multiple passes', default=1)
flags.DEFINE_bool(
    'use_train_device_loop', help='Use device loop for training', default=True)

flags.DEFINE_multi_string(
    'eval_image_indices',
    help='Eval image indices',
    default=[
        '00000', '00003', '00005', '00006', '00012', '00024', '00034', '00041',
        '00044', '00049', '00052', '00056', '00061', '00065', '00066', '00070',
        '00076', '00078', '00080', '00084', '00086', '00087', '00092', '00111',
        '00112', '00125', '00128', '00138', '00157', '00160', '00161', '00162',
        '00169', '00171', '00176', '00185', '00187', '00189', '00198', '00203',
        '00206', '00207'
    ])
