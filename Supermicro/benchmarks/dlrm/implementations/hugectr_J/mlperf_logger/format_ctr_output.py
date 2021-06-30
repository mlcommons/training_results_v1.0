import json
import argparse
import os

from . import utils as mllogger
import mlperf_logging.mllog as mllog

# map keys traditionally used in hugectr configs
# to more descriptive ones more suitable for MLPerf
hugectr_to_mlperf_layer_name = {
    'sparse_embedding1': 'embeddings',
    'fc1': 'bottom_mlp_dense1',
    'fc2': 'bottom_mlp_dense2',
    'fc3': 'bottom_mlp_dense3',
    'fc4': 'top_mlp_dense1',
    'fc5': 'top_mlp_dense2',
    'fc6': 'top_mlp_dense3',
    'fc7': 'top_mlp_dense4',
    'fc8': 'top_mlp_dense5'
}

def log_hparams(config):
    mllogger.log_event(key='eval_samples',
                       value=config['layers'][0]['eval_num_samples'])
    mllogger.log_event(key='global_batch_size',
                       value=config['solver']['batchsize'])
    mllogger.log_event(key='opt_base_learning_rate',
                       value=config['optimizer']['sgd_hparam']['learning_rate'])
    mllogger.log_event(key='sgd_opt_base_learning_rate',
                       value=config['optimizer']['sgd_hparam']['learning_rate'])
    mllogger.log_event(key='sgd_opt_learning_rate_decay_poly_power',
                       value=config['optimizer']['sgd_hparam'].get('decay_power'))
    mllogger.log_event(key='opt_learning_rate_warmup_steps',
                       value=config['optimizer']['sgd_hparam']['warmup_steps'])
    mllogger.log_event(key='opt_learning_rate_warmup_factor',
                       value=0.0)  # not configurable
    mllogger.log_event(key='lr_decay_start_steps',
                       value=config['optimizer']['sgd_hparam'].get('decay_start'))
    mllogger.log_event(key='sgd_opt_learning_rate_decay_steps',
                       value=config['optimizer']['sgd_hparam'].get('decay_steps'))
    mllogger.log_event(key='gradient_accumulation_steps',
                       value=1)  # not configurable


def log_config(config):
    # print hparams and submission info on the first node only
    if 'SLURM_NODEID' not in os.environ or os.environ['SLURM_NODEID'] == '0':
        mllogger.mlperf_submission_log('dlrm')
        log_hparams(config)

    for layer in config['layers']:
        hugectr_name = layer['name']
        if hugectr_name not in hugectr_to_mlperf_layer_name:
            # layer has no parameters, nothing to be done
            continue

        mlperf_name = hugectr_to_mlperf_layer_name[hugectr_name]
        mllogger.log_event(mllog.constants.WEIGHTS_INITIALIZATION,
                           metadata={'tensor': mlperf_name})


class LogConverter:
    def __init__(self, steps_per_epoch, start_timestamp):
        self.start_time = start_timestamp
        self.steps_per_epoch = steps_per_epoch

    def _get_log_foo(self, key):
        if '_start' in key:
            return mllogger.log_start
        if '_end' in key or '_stop' in key:
            return mllogger.log_end
        else:
            return mllogger.log_event

    def _get_value(self, data):
        if data[0] == 'eval_accuracy':
            return float(data[1])
        if data[0] == 'train_samples':
            return int(data[1])

    def _get_metadata(self, data):
        if data[0] == 'eval_accuracy':
            self._last_eval_accuracy = float(data[1])
            return { 'epoch_num': float(data[2]) + 1 }
        if 'eval' in data[0]:
            return { 'epoch_num': float(data[1]) + 1 }
        if 'epoch' in data[0]:
            return { 'epoch_num': int(data[1]) + 1 }
        if data[0] == 'run_stop':
            return { 'status': 'success' if self._last_eval_accuracy > 0.8025 else 'aborted' }

    def _get_kvm(self, data):
        key = data[0]
        if data[0] == 'init_end':
            key = 'init_stop'
        if data[0] == 'train_epoch_start':
            key = 'epoch_start'
        if data[0] == 'train_epoch_end':
            key = 'epoch_stop'

        value = self._get_value(data)
        metadata = self._get_metadata(data)

        return key, value, metadata

    def _get_time_ms(self, ms):
        return self.start_time + int(float(ms))

    def validate_event(self, event):
        try:
            float(event[0])

            if not event[1].isidentifier():
                return False

            for x in event[2:]:
                float(x)
            return True
        except:
            return False

    def log_throughput(self, line):
        """Read throughput from log file of recommendation

        Example:
            From Below line
            "Hit target accuracy AUC 0.8025 at epoch 0.95 in 22718s. Average speed 175427.7 records/s."
            Get the value -> 175427.7
        """
        lastmatch = None
        if "Hit target accuracy" in line:
            lastmatch = line
        elif "Average speed" in line:
            lastmatch = line
        throughput = 0

        if lastmatch is not None:
            throughput = float(lastmatch.split(' ')[-2])
            epoch = float(lastmatch.split(' ')[7])

        if throughput > 0:
            mllogger.log_event(key="tracked_stats",
                               value={'throughput': throughput},
                               metadata={'step': epoch})

    def log_event(self, event_log):
        if self.validate_event(event_log):
            log_foo = self._get_log_foo(event_log[1])
            key, value, metadata = self._get_kvm(event_log[1:])
            time_ms = self._get_time_ms(event_log[0])

            log_foo(key=key, value=value, metadata=metadata, time_ms=time_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str,
                        help='Path to the logs to be translated')

    parser.add_argument('--config_file', type=str,
                        help='HugeCTR input config file in JSON format')

    parser.add_argument('--start_timestamp', type=int,
                        help='Seconds since 1970-01-01 00:00:00 UTC at the time of training start')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)
    log_config(config)

    # Convert to ms to be consistent with the MLPerf logging API
    start_timestamp_ms = args.start_timestamp * 1000
    converter = LogConverter(
        steps_per_epoch=(config['layers'][0]['num_samples'] / config['solver']['batchsize']),
        start_timestamp=start_timestamp_ms,
    )

    with open(args.log_path) as f:
        log_lines = f.readlines()

    for line in log_lines:
        event_log = [x.strip() for x in line.strip().strip('][\x08 ,').split(',')]
        converter.log_event(event_log)
        converter.log_throughput(line)


if __name__ == '__main__':
    main()
