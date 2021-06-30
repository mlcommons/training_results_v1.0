# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import enum
import popart
import logging

import numpy as np

logger = logging.getLogger(__name__)


class ScheduleMode(enum.Enum):
    CONSTANT = 0
    STEP = 1
    EPOCH = 2


class BaseOptimizerFactory():
    def __init__(self, args, iteration, opt_type="SGD", tensors=None):
        self.args = args
        self.opt_type = opt_type

        if self.opt_type == "SGD":
            self.option_values = {
                "defaultLearningRate": args.learning_rate,
                "defaultMomentum": args.momentum,
                "defaultDampening": args.dampening or args.momentum,
                "defaultVelocityScaling": args.velocity_scaling,
                "lossScaling": args.loss_scaling,
            }
            self._non_const_options = set()
        else:
            self.option_values = {
                "defaultLearningRate": args.learning_rate,
                "defaultBeta1":  args.beta1 if args.beta1 is not None else args.beta1_schedule["init"],
                "defaultBeta2": args.beta2,
                "lossScaling": args.loss_scaling,
                "maxWeightNorm": args.max_weight_norm if args.max_weight_norm is not None else np.finfo(np.float16).max
            }
            self._non_const_options = set(["defaultLearningRate", "defaultBeta1", "defaultBeta2"])

        self._options_created = False
        self.iteration = iteration

        self.squad_lr_scaling = args.task == "SQUAD"
        self.squad_lr_scale = args.squad_lr_scale

        self.lr_scaling = args.pipeline_lr_scaling
        self.weight_decay = args.weight_decay
        self.momentum_scaling = args.pipeline_momentum_scaling
        self.execution_mode = args.execution_mode
        self.tensors = tensors if tensors is not None else {}
        self.pipeline_stage_lr_scaling = []

        # If pipelining is enabled, we want to scale the parameters for different
        # pipeline stages. If not, don't perform scaling.
        # Note: This calculates the scale factors, not the absolute values
        if self.execution_mode == "PIPELINE" and (self.lr_scaling or self.momentum_scaling):
            if self.lr_scaling:
                offset = args.pipeline_lr_scaling_offset
                self.pipeline_stage_lr_scaling = self._pipeline_stage_parameter_scaling(offset, tensors)
            if self.momentum_scaling:
                offset = args.pipeline_momentum_scaling_offset
                self.pipeline_stage_momentum_scaling = self._pipeline_stage_parameter_scaling(offset, tensors)
                if args.pipeline_dampening_scaling_offset is not None:
                    offset = args.pipeline_dampening_scaling_offset
                self.pipeline_stage_dampening_scaling = self._pipeline_stage_parameter_scaling(offset, tensors)

    @property
    def optimizer_options(self):
        self._options_created = True
        # By default, options are const. They only become variable when they're scheduled in some way, at
        # which point their key should be appended to _non_const_options
        return {k: (v, k not in self._non_const_options) for k, v in self.option_values.items()}

    @property
    def learning_rate(self):
        return self.option_values["defaultLearningRate"]

    @property
    def weight_decay_tensor_list(self):
        if getattr(self, "_weight_decay_tensor_list", "uninitialized") is "uninitialized":
            self._weight_decay_tensor_list = []
            if self.execution_mode == "PIPELINE":
                for stage in self.tensors:
                    for tensor_id in self.tensors[stage]:
                        if self.include_for_weight_decay(tensor_id):
                            self._weight_decay_tensor_list.append(tensor_id)
            else:
                for tensor_id in self.tensors[0]:
                    if self.include_for_weight_decay(tensor_id):
                        self._weight_decay_tensor_list.append(tensor_id)
        return self._weight_decay_tensor_list

    @property
    def validation_optimizer(self):
        # Define the validation optimzier based on the training optimizer
        if getattr(self, "_validation_optimizer", "not_initialized") == "not_initialized":
            # An optimizer with one that zeroes out all training
            validation_optimizer = {
                "defaultLearningRate": (0, False),
                "defaultBeta1": (1, False),
                "defaultBeta2": (1, False),
                "lossScaling": (self.args.loss_scaling, True),
                "maxWeightNorm": (np.finfo(np.float16).max, True)
            }
            self._validation_optimizer = popart.Adam(validation_optimizer, mode=popart.AdamMode.LambNoBias)

            # Also zero out weight decay on parameters to which it applies
            for tensor_id in self.weight_decay_tensor_list:
                self._validation_optimizer.insertSpecific(tensor_id, {"weightDecay": (0, False)})

        return self._validation_optimizer

    def include_for_weight_decay(self, tensor_id):
        """ Do not include bias and norms for weight decay."""
        return self.weight_decay > 0 and\
            not tensor_id.endswith('B') and\
            not tensor_id.endswith('bias') and\
            not tensor_id.endswith('Beta') and\
            not tensor_id.endswith('Gamma')

    def update_and_create(self, iteration):
        self.update(iteration)
        return self.create()

    def create(self):
        self.iteration.learning_rate = self.optimizer_options["defaultLearningRate"][0]
        self.iteration.beta1 = self.optimizer_options["defaultBeta1"][0]
        dtype = popart.DataType.FLOAT16
        if self.opt_type == "SGD":
            optimizer = popart.SGD(self.optimizer_options)
        elif self.opt_type == "ADAM":
            optimizer = popart.Adam(self.optimizer_options)
        elif self.opt_type == "ADAM_NO_BIAS":
            optimizer = popart.Adam(self.optimizer_options,
                                    mode=popart.AdamMode.AdamNoBias)
        elif self.opt_type == "LAMB":
            optimizer = popart.Adam(self.optimizer_options,
                                    mode=popart.AdamMode.Lamb)
        elif self.opt_type == "LAMB_NO_BIAS":
            accl1_type = popart.DataType.FLOAT
            if self.args.lamb_m_dtype == "FLOAT16":
                accl1_type = popart.DataType.FLOAT16
            optimizer = popart.Adam(self.optimizer_options,
                                    mode=popart.AdamMode.LambNoBias,
                                    accl1_type=accl1_type)

        # Add weight decay to the specified tensors
        for tensor_id in self.weight_decay_tensor_list:
            specific_parameters = {"weightDecay": (self.weight_decay, False)}
            optimizer.insertSpecific(tensor_id, specific_parameters)
            logger.debug(f" Weight decay of {self.weight_decay} applied to: {tensor_id}")
        self.training_optimizer = optimizer
        return optimizer

    def should_update(self, iteration):
        raise NotImplementedError("This method should be overridden and not called directly")

    def update(self, iteration):
        raise NotImplementedError("This method should be overridden and not called directly")

    def _pipeline_stage_parameter_scaling(self, offset, tensors):
        if len(tensors) == 1:
            return {tensors.keys()[0]: 1.0}

        stages = tensors.keys()
        scale_factor = (1 - offset)/max(stages)
        return {stage: abs(scale_factor * stage + offset) for stage in stages}


class Schedule(object):
    def __init__(self, mode, schedule, param, default_value):
        self.mode = mode
        self.schedule = schedule
        self.param = param

        self._initial_value = self.schedule[0] if 0 in self.schedule else default_value
        self.current_value = self.initial_value
        self.current_critereon = 0

    def should_update(self, iteration):
        """If using constant mode, we should never update the learning rate.
        If a shedule has been provided, check whether it's the right mode (i.e.
        due to a step or epoch change), and if so, whether it's the right time."""
        if self.mode == ScheduleMode.CONSTANT:
            return False

        # Check if the relevant critereon has changed (needed because we check epochs and steps)
        criterion = self._read_schedule_criterion(iteration)
        if criterion == self.current_critereon:
            return False

        self.current_critereon = criterion
        return criterion in self.schedule.keys()

    def update(self, iteration):
        criterion = self._read_schedule_criterion(iteration)

        # Sanity check that the learning rate is in the schedule, if not return the current LR
        if criterion is not None:
            self.current_value = self.schedule[criterion]
        return self.current_value

    @property
    def initial_value(self):
        return self._initial_value

    def _read_schedule_criterion(self, iteration):
        if self.mode == ScheduleMode.STEP:
            return iteration.count
        elif self.mode == ScheduleMode.EPOCH:
            return iteration.epoch
        return None

    def fast_forward(self, iteration):
        target_criterion = self._read_schedule_criterion(iteration)

        diffs = {(target_criterion - k): k for k in self.schedule.keys() if k <= target_criterion}
        closest_key = diffs[min(diffs)]

        self.current_value = self.schedule[closest_key]
        return self.current_value

    @staticmethod
    def from_args(param, default_value, schedule_arg_epoch, schedule_arg_steps, lr_bert_schedule):
        # Epoch and step arguments are in a mutually exclusive group in argparse
        if schedule_arg_epoch is not None:
            mode = ScheduleMode.EPOCH
            schedule = Schedule.parse(param, schedule_arg_epoch)
        elif schedule_arg_steps is not None:
            mode = ScheduleMode.STEP
            schedule = Schedule.parse(param, schedule_arg_steps)
        elif lr_bert_schedule is not None:
            mode = ScheduleMode.STEP

            def lr(global_step):
                if global_step < lr_bert_schedule["num_warmup_steps"]:
                    linear_lr = (global_step)/(lr_bert_schedule["num_warmup_steps"])*lr_bert_schedule["init_lr"]
                else:
                    # Polynomial power is hardcoded to 1, as usual
                    global_step = global_step - lr_bert_schedule["num_warmup_steps"]
                    linear_lr = lr_bert_schedule["init_lr"]*(lr_bert_schedule["num_training_steps"] - global_step)/lr_bert_schedule["num_training_steps"]
                return max(linear_lr, 1e-7)

            raw_schedule = {step: lr(step) for step in range(lr_bert_schedule["num_training_steps"] + lr_bert_schedule["num_warmup_steps"])}
            schedule = Schedule.parse(param, raw_schedule)
        else:
            # If no schedule is provided, set the learning rate mode to constant
            # and initialise it at the provided learning rate.
            mode = ScheduleMode.CONSTANT
            schedule = {0: default_value}
        return Schedule(mode, schedule, param, default_value)

    @staticmethod
    def parse(param, raw_schedule):
        try:
            return {int(k): float(raw_schedule[k]) for k in raw_schedule}
        except ValueError as ex:
            logger.warning(f"Invalid Schedule provided for parameter [{param}]. "
                           "It should be a set of int:float pairs.")
            raise ex


class ScheduledOptimizerFactory(BaseOptimizerFactory):
    def __init__(self, args, iteration, opt_type="SGD", tensors=None):
        super().__init__(args, iteration, opt_type, tensors)

        self._schedules = {}
        self.awaiting_update = []
        self.current_critereon = 0

        self._create_schedules(args)

        # Since the step count is set > 0 if we start from a given epoch,
        # this will catch either step or epoch start states
        if iteration.count > 0:
            self._fast_forward()

    def should_update(self, iteration):
        self.awaiting_update = [p for p, s in self._schedules.items() if s.should_update(iteration)]
        return len(self.awaiting_update) > 0

    def update(self, iteration):
        for param_name in self.awaiting_update:
            self.option_values[param_name] = self._schedules[param_name].update(iteration)

    def add_schedule(self, schedule):
        # This is required since if we specify any option as const, it cannot then change.
        if self._options_created:
            raise RuntimeError(
                "Cannot add new schedules once options have been created.")
        self._non_const_options.add(schedule.param)
        self._schedules[schedule.param] = schedule
        self.option_values[schedule.param] = schedule.initial_value

    def _create_schedules(self, args):
        if any([a is not None for a in [args.lr_schedule_by_epoch, args.lr_schedule_by_step, args.lr_bert_schedule]]):
            self.add_schedule(Schedule.from_args("defaultLearningRate",
                                                 args.learning_rate,
                                                 args.lr_schedule_by_epoch,
                                                 args.lr_schedule_by_step,
                                                 args.lr_bert_schedule))
        if args.ls_schedule_by_epoch is not None or args.ls_schedule_by_step is not None:
            self.add_schedule(Schedule.from_args("lossScaling",
                                                 args.loss_scaling,
                                                 args.ls_schedule_by_epoch,
                                                 args.ls_schedule_by_step))

        # Schedule for beta1
        # Example: "beta1_schedule": {"init": 0.75, "final": 0.4, "num_transition_steps": 1000},
        if args.beta1_schedule is not None:
            init = args.beta1_schedule["init"]
            final = args.beta1_schedule["final"]
            num_transition_steps = args.beta1_schedule["num_transition_steps"]

            def f(step):
                return init + (final-init)*step/num_transition_steps

            transition_steps = list(range(num_transition_steps))
            beta1_values = list(map(f, transition_steps))
            schedule = {int(k): float(v) for k, v in zip(transition_steps, beta1_values)}
            mode = ScheduleMode.STEP
            param = "defaultBeta1"
            self.add_schedule(Schedule(mode, schedule, param, init))

        logger.debug("Created schedules...")
        for schedule in self._schedules.values():
            logger.debug(f"Schedule[{schedule.param} | {str(schedule.mode)}]")
            for key, value in schedule.schedule.items():
                logger.debug(f"\t{key:>6}: {value}")

    def _fast_forward(self):
        for param_name in self._schedules.keys():
            self.option_values[param_name] = self._schedules[param_name].fast_forward(self.iteration)
