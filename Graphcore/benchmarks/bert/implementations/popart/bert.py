# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import time
import os
import sys
import math
import ctypes
import random
import datetime
from functools import reduce, partial
from collections import deque
from collections import defaultdict
from itertools import chain
import logging
import socket

import popart
import popdist
import popdist.popart
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from bert_model import ExecutionMode, get_model, BertConfig
from bert_data import get_pretraining_dataset, get_squad_dataset, get_packed_pretraining_dataset
from bert_tf_loader import load_initializers_from_tf
from bert_optimizer import ScheduledOptimizerFactory, BaseOptimizerFactory
from phased_execution.weight_mapping import get_phased_initializers_from_default
import utils
import utils.popvision as popvision
from utils.device import acquire_device, device_is_replicated
from utils.distributed import popdist_root, distributed_barrier, average_distributed_deques, sum_distributed_data
from utils.inference import (create_callback_stepio,
                             realtime_scheduling,
                             compute_latency_from_durations,
                             compute_latency_from_callbacks)
from mlperf_logging import mllog

mlm_accuracy_target = 0.720

logger = logging.getLogger('BERT')

so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "custom_ops.so")
if os.path.exists(so_path):
    ctypes.cdll.LoadLibrary(so_path)
else:
    logger.warning("Could not find custom_ops.so. Execute `make` before running this script.")


def set_library_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


def bert_config_from_args(args):
    return BertConfig(**{k: getattr(args, k)
                         for k in BertConfig._fields if hasattr(args, k)})


def bert_add_embedding_inputs(args, model, sequence_info):
    if args.host_embedding == "NONE":
        indices = model.builder.addInputTensor(sequence_info, "indices")
        positions = model.builder.addInputTensor(sequence_info, "positions")
    else:   # "ALL", "WORD", "MERGE"
        expanded_sequence_info = popart.TensorInfo(
            "FLOAT16", [args.batch_size * args.sequence_length, args.hidden_size])
        indices = model.builder.addInputTensor(
            expanded_sequence_info, "indices_expanded")
        if args.host_embedding == "ALL":
            positions = model.builder.addInputTensor(
                expanded_sequence_info, "pos_expanded")
        else:
            positions = model.builder.addInputTensor(sequence_info, "positions")

    return indices, positions


def bert_add_inputs(args, model):
    # make inputs directly accessible to the model
    model.inputs = {}
    model.masks = []
    model.labels = {}

    if not args.inference:
        mode_info = popart.TensorInfo("UINT32", [1])
        model.inputs["is_training"] = model.builder.addInputTensor(mode_info, "is_training")

    sequence_info = popart.TensorInfo("UINT32", [args.batch_size * args.sequence_length])
    indices, positions = bert_add_embedding_inputs(args, model, sequence_info)
    segments = model.builder.addInputTensor(sequence_info, "segment_ids")
    model.inputs["input_ids"] = indices
    model.inputs["position_ids"] = positions
    model.inputs["segment_ids"] = segments

    if args.task == "PRETRAINING":
        mask_info = popart.TensorInfo("UINT32", [args.batch_size, args.sequence_length])
        model.masks.append(model.builder.addInputTensor(mask_info, "input_mask"))

        # Labels for MLM also include positions
        mlm_info = popart.TensorInfo("UINT32", [args.batch_size, args.mask_tokens + args.max_sequences_per_pack])
        model.inputs["masked_lm_positions"] = model.builder.addInputTensor(mlm_info, "masked_lm_positions")
        model.labels["masked_lm_ids"] = model.builder.addInputTensor(mlm_info, "masked_lm_ids")
        model.inputs["masked_lm_weights"] = model.builder.addInputTensor(mlm_info, "masked_lm_weights")

        # NSP as usual
        if not args.reduce_nsp_overhead:
            nsp_info = popart.TensorInfo("UINT32", [args.batch_size, args.max_sequences_per_pack])
            model.labels["nsp_labels"] = model.builder.addInputTensor(nsp_info, "nsp_labels")
            model.labels["nsp_weights"] = model.builder.addInputTensor(nsp_info, "nsp_weights")
            model.inputs["nsp_positions"] = model.builder.addInputTensor(nsp_info, "nsp_positions")

    elif args.task == "SQUAD":
        mask_info = popart.TensorInfo("UINT32", [args.batch_size, 1])
        model.masks.append(model.builder.addInputTensor(mask_info, "seq_pad_idx"))
        if not args.inference:
            labels_info = popart.TensorInfo("UINT32", [args.batch_size])
            model.labels["start_labels"] = model.builder.addInputTensor(labels_info, "start_labels")
            model.labels["end_labels"] = model.builder.addInputTensor(labels_info, "end_labels")

    return


def bert_logits_graph(model, mode):
    if mode == ExecutionMode.PHASED:
        logits = model(model.inputs["input_ids"], model.inputs["position_ids"],
                       model.inputs["segment_ids"], model.masks if len(model.masks) > 0 else None)
    else:
        logits = model.build_graph()
    return logits


def bert_infer_graph(model, logits, include_probs=True):
    # NOTE: include_probs added as we don't need to calculate the softmax if we only care about accuracy and not
    # about loss
    probs = None
    if model.config.task == "SQUAD":
        scope = model.squad_scope
        if model.config.execution_mode == ExecutionMode.PHASED:
            scope = model.scope_provider(model.builder, scope)
        with scope:
            predictions = list(
                model.builder.aiOnnx.argmax([logit],
                                            axis=-1,
                                            keepdims=0,
                                            debugContext=f"{logit}/ArgMax")
                for logit in logits)
            if include_probs:
                probs = list(
                    model.builder.aiOnnx.softmax(
                        [logit], axis=-1, debugContext=f"{logit}/Softmax")
                    for logit in logits)

                for prob in probs:
                    model.builder.setInplacePreferences(
                        prob, {"SoftmaxInplace": -1})
    elif model.config.task == "PRETRAINING":
        nsp_scope = model.nsp_scope
        mlm_scope = model.mlm_scope
        if model.config.execution_mode == ExecutionMode.PHASED:
            nsp_scope = model.scope_provider(model.builder, nsp_scope)
            mlm_scope = model.scope_provider(model.builder, mlm_scope)
        if not model.config.reduce_nsp_overhead:
            with nsp_scope:
                nsp_predictions = model.builder.aiOnnx.argmax([logits[1]], axis=-1, keepdims=0, debugContext="ArgMax")
                if include_probs:
                    nsp_probs = model.builder.aiOnnx.softmax([logits[1]], axis=-1, debugContext="Softmax")

        with mlm_scope:
            mlm_predictions = model.builder.aiOnnx.argmax([logits[0]], axis=-1, keepdims=0, debugContext="ArgMax")
            if include_probs:
                mlm_probs = model.builder.aiOnnx.softmax([logits[0]], axis=-1, debugContext="Softmax")

        if not model.config.reduce_nsp_overhead:
            predictions = [mlm_predictions, nsp_predictions]
        else:
            predictions = [mlm_predictions]
        if include_probs:
            if not model.config.reduce_nsp_overhead:
                probs = [mlm_probs, nsp_probs]
            else:
                probs = [mlm_probs]
    return predictions, probs


def get_ignore_index(label):
    if 'mask' in label:
        return 0
    return None


def get_loss_scope(model, label):
    if model.config.task == "SQUAD":
        scope = model.squad_scope
    elif 'nsp' in label:
        scope = model.nsp_scope
    else:
        scope = model.mlm_scope
    if model.config.execution_mode == ExecutionMode.PHASED:
        scope = model.scope_provider(model.builder, scope)
    return scope


def packed_bert_loss_and_accuracy(args, model, probs, logits, predictions):

    # Which outputs should be streamed back to host
    outputs_to_anchor = {}

    # MLM
    with model.mlm_scope:
        mlm_probs = probs[0]
        mlm_predictions = predictions[0]
        mlm_labels = model.labels["masked_lm_ids"]
        mlm_labels = model.builder.aiOnnx.cast([mlm_labels], "INT32")
        mlm_seq_ind = model.inputs["masked_lm_weights"]
        mlm_seq_ind = model.builder.reshape_const(model.builder.aiOnnx, [mlm_seq_ind], [model.config.batch_size, 1, -1])

        # Sequences per pack
        masked_lm_weights = model.builder.aiOnnx.cast([model.inputs["masked_lm_weights"]], "FLOAT")
        sequences_in_sample = model.builder.aiOnnx.reducemax([masked_lm_weights], axes=[-1], keepdims=True)
        sequences_in_sample = model.builder.checkpointOutput([sequences_in_sample])[0]
        sequences_in_sample_flattened = model.builder.aiOnnx.reducesum([sequences_in_sample], keepdims=False)

        # First compute the per-token nll and do not perform any reduction
        nll_per_token = model.builder.aiGraphcore.nllloss([mlm_probs, mlm_labels], ignoreIndex=0,
                                                          reduction=popart.ReductionType.NoReduction, debugContext=f"MLM/loss")
        nll_per_token = model.builder.checkpointOutput([nll_per_token])[0]
        nll_per_token = model.builder.aiOnnx.cast([nll_per_token], "FLOAT")
        nll_per_token = model.builder.reshape_const(model.builder.aiOnnx, [nll_per_token], [model.config.batch_size, 1, -1])

        # Calculate the per token accuracy (hit or miss) and do not perform any reduction
        mlm_accuracy_per_token = model.builder.aiOnnx.equal([mlm_predictions, mlm_labels])
        mlm_accuracy_per_token = model.detach(mlm_accuracy_per_token)
        mlm_accuracy_per_token = model.builder.aiOnnx.cast([mlm_accuracy_per_token], "FLOAT")
        mlm_accuracy_per_token = model.builder.reshape_const(model.builder.aiOnnx, [mlm_accuracy_per_token], [model.config.batch_size, 1, -1])

        # Now expand the per-token loss an a per-sequence basis [B, T] -> [B, max_sequences_per_pack, T]
        # The sequence selection tensor select the different sequences in a pack by masking
        sequence_selection = np.array(range(1, model.config.max_sequences_per_pack + 1)).reshape([1, -1, 1])
        sequence_selection = model.builder.aiOnnx.equal([mlm_seq_ind, model.constant_tensor(sequence_selection, dtype=np.uint32)])
        sequence_selection = model.detach(sequence_selection)
        sequence_selection_i = model.builder.aiOnnx.cast([sequence_selection], "INT32")
        sequence_selection_f = model.builder.aiOnnx.cast([sequence_selection], "FLOAT")
        sequence_selection_f = model.builder.checkpointOutput([sequence_selection_f])[0]

        # Calculate the per-sequence normalization constants (if 0, increase to 1 to avoid NaNs)
        attempted = model.builder.aiOnnx.reducesum([sequence_selection_i], axes=[-1], keepdims=True)

        # Mask stating which sequences were not attempted
        not_attempted_mask = model.detach(model.builder.aiOnnx.equal([attempted, model.constant_tensor([0], dtype=np.int32)]))  # prevent nans
        not_attempted_mask_f = model.builder.aiOnnx.cast([not_attempted_mask], "FLOAT")

        # Number of tokens in each of the attempted sequences
        attempted = model.builder.aiOnnx.cast([attempted], "FLOAT")
        attempted = model.builder.aiOnnx.add([attempted, not_attempted_mask_f])
        attempted = model.detach(attempted)
        attempted = model.builder.checkpointOutput([attempted])[0]

        # Calculate per sequence loss
        nll_per_sequence = model.builder.aiOnnx.mul([nll_per_token, sequence_selection_f])
        nll_per_sequence = model.builder.aiOnnx.div([nll_per_sequence, attempted])

        # Divide total loss by number of sequences to get average loss
        mlm_loss = model.builder.aiOnnx.reducesum([nll_per_sequence], keepdims=False)
        mlm_loss = model.builder.aiOnnx.div([mlm_loss, sequences_in_sample_flattened])
        outputs_to_anchor[mlm_loss] = popart.AnchorReturnType("SUM")

        # Now compute the MLM accuracy
        mlm_accuracy_per_sequence = model.builder.aiOnnx.mul([mlm_accuracy_per_token, sequence_selection_f])
        mlm_accuracy_per_sequence = model.builder.aiOnnx.div([mlm_accuracy_per_sequence, attempted])
        mlm_accuracy = model.builder.aiOnnx.reducesum([mlm_accuracy_per_sequence], axes=[-1], keepdims=False)
        # For accuracy we need to return all values since each batch has a different number of sequences
        outputs_to_anchor[mlm_accuracy] = popart.AnchorReturnType("ALL")

    # NSP accuracy and loss computed per-pack
    if not model.config.reduce_nsp_overhead:
        with model.nsp_scope:
            nsp_predictions = predictions[1]
            nsp_probs = probs[1]
            nsp_labels = model.builder.aiOnnx.cast([model.labels["nsp_labels"]], "INT32")

            # Loss and accuracy per token
            nsp_nll_per_token = model.builder.aiGraphcore.nllloss([nsp_probs, model.labels["nsp_labels"]], ignoreIndex=None,
                                                                  reduction=popart.ReductionType.NoReduction, debugContext=f"NSP/loss")
            nsp_nll_per_token = model.builder.checkpointOutput([nsp_nll_per_token])[0]
            nsp_nll_per_token = model.builder.aiOnnx.cast([nsp_nll_per_token], "FLOAT")
            nsp_accuracy_per_token = model.builder.aiOnnx.equal([nsp_labels, nsp_predictions])
            nsp_accuracy_per_token = model.builder.aiOnnx.cast([nsp_accuracy_per_token], "FLOAT")

            # Attempted tokens
            nsp_weights = model.builder.aiOnnx.cast([model.labels["nsp_weights"]], "INT32")
            nsp_weights_f = model.builder.aiOnnx.cast([nsp_weights], "FLOAT")  # 1 or 0 mask
            attempted = model.builder.aiOnnx.reducesum([nsp_weights_f], axes=[-1], keepdims=True)  # always > 0

            # NSP loss
            nsp_loss = model.builder.aiOnnx.mul([nsp_nll_per_token, nsp_weights_f])
            nsp_loss = model.builder.aiOnnx.reducesum([nsp_loss], axes=[-1], keepdims=False)
            nsp_loss = model.builder.aiOnnx.div([nsp_loss, attempted])
            nsp_loss = model.builder.aiOnnx.reducemean([nsp_loss], keepdims=False)
            outputs_to_anchor[nsp_loss] = popart.AnchorReturnType("SUM")

            # NSP accuracy
            nsp_accuracy = model.builder.aiOnnx.mul([nsp_accuracy_per_token, nsp_weights_f])
            nsp_accuracy = model.builder.aiOnnx.div([nsp_accuracy, attempted])
            nsp_accuracy = model.builder.aiOnnx.reducesum([nsp_accuracy], axes=[-1], keepdims=False)
            nsp_accuracy = model.builder.aiOnnx.reducemean([nsp_accuracy], keepdims=False)
            outputs_to_anchor[nsp_accuracy] = popart.AnchorReturnType("SUM")

    # MLM + NSP is final loss
    with model.final_loss_scope:
        if not model.config.reduce_nsp_overhead:
            final_loss = model.builder.aiOnnx.add([mlm_loss, nsp_loss], "FinalLoss")
        else:
            final_loss = mlm_loss

    for out in outputs_to_anchor.keys():
        model.builder.addOutputTensor(out)
    if not model.config.reduce_nsp_overhead:
        return [mlm_loss, nsp_loss], [mlm_accuracy, nsp_accuracy], final_loss, outputs_to_anchor
    else:
        return [mlm_loss], [mlm_accuracy], final_loss, outputs_to_anchor


def bert_perplexity_graph(args, model, logits, labels):
    with model.mlm_scope:
        mlm_probs = model.builder.aiOnnx.softmax(
            [logits[0]], axis=2, debugContext="Softmax")

    losses, final_loss = bert_loss_graph(args, model, [mlm_probs], [labels[0]])

    losses.append(None)

    return losses, final_loss


def bert_accuracy_calculation(builder, prediction, label, ignore_index=None, mask=None):
    # Prediction will be the output of an ArgMax -> INT32
    with builder.nameScope("Accuracy"):
        label = builder.aiOnnx.cast([label], "INT32")
        results = builder.aiOnnx.equal([prediction, label])
        results = builder.aiOnnx.cast([results], "INT32")
        if ignore_index is not None:
            _ii = builder.aiOnnx.constant(np.array(ignore_index).astype(np.int32), f"{label}_ignore_index")
            mask = builder.aiOnnx.equal([label, _ii], "Mask")
            mask = builder.aiOnnx.logical_not([mask], "~Mask")
            mask = builder.aiOnnx.cast([mask], "INT32")
            results = builder.aiOnnx.mul([results, mask], "MaskApply")
            total_attempted = builder.aiOnnx.reducesum([mask],
                                                       axes=range(len(builder.getTensorShape(mask))),
                                                       keepdims=0,
                                                       debugContext="TotalAttempted")
        else:
            mask = builder.aiOnnx.cast([mask], "INT32")
            results = builder.aiOnnx.mul([results, mask], "MaskApply")
            total_attempted = builder.aiOnnx.reducesum([mask], keepdims=0, debugContext="TotalAttempted")

        total_correct = builder.aiOnnx.reducesum([results],
                                                 axes=range(len(builder.getTensorShape(label))),
                                                 keepdims=0,
                                                 debugContext="TotalCorrect")
        total_correct = builder.aiOnnx.cast([total_correct], "FLOAT")
        total_attempted = builder.aiOnnx.cast([total_attempted], "FLOAT")
        accuracy = builder.aiOnnx.div([total_correct, total_attempted])
    return accuracy, total_attempted


def bert_add_validation_outputs(args, model, predictions, labels, losses):
    outputs = {}
    accuracies = []
    avg_losses = []
    for pred, label, loss in zip(predictions, labels, losses):
        with get_loss_scope(model, label):
            accuracy, num_attempted = bert_accuracy_calculation(model.builder, pred, label, get_ignore_index(label), mask=labels[2])
            accuracies.append(accuracy)
            outputs[accuracy] = popart.AnchorReturnType("SUM")

            if loss is not None:
                loss = model.builder.aiOnnx.cast([loss], "FLOAT")
                if args.gradient_reduction_type == "Sum":
                    loss = model.builder.aiOnnx.div([loss, num_attempted])
                avg_losses.append(loss)
                outputs[loss] = popart.AnchorReturnType("SUM")
    for out in outputs.keys():
        model.builder.addOutputTensor(out)
    return outputs, accuracies, avg_losses


def bert_add_logit_outputs(model, logits):
    outputs = {}
    for logit in logits:
        outputs[logit] = popart.AnchorReturnType("ALL")
    for out in outputs.keys():
        model.builder.addOutputTensor(out)
    return outputs


def bert_optimizer_location_settings(args):
    storage = popart.TensorStorage.OnChip
    if args.optimizer_state_offchip:
        storage = popart.TensorStorage.OffChip
    rts = popart.ReplicatedTensorSharding.Off
    if args.replicated_tensor_sharding:
        rts = popart.ReplicatedTensorSharding.On

    return popart.TensorLocationSettings(popart.TensorLocation(storage, rts))


def bert_session_options(args, model):
    engine_options = {}
    options = popart.SessionOptions()
    options.virtualGraphMode = popart.VirtualGraphMode.Manual
    options.enableFloatingPointChecks = args.floating_point_exceptions
    options.enableStochasticRounding = args.stochastic_rounding
    options.enableGroupedMatmuls = False
    options.enablePrefetchDatastreams = not args.minimum_latency_inference
    options.enableOutlining = not args.no_outlining
    options.subgraphCopyingStrategy = popart.SubgraphCopyingStrategy.JustInTime
    partials_type = "half" if args.enable_half_partials else "float"
    options.partialsTypeMatMuls = partials_type
    options.convolutionOptions = {'partialsType': partials_type}
    if args.replication_factor > 1:
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = args.replication_factor
        engine_options["target.syncReplicasIndependently"] = "true"
    else:
        # only one replicator, can not use RTS
        args.replicated_tensor_sharding = False

    if args.use_popdist:
        popdist.popart.configureSessionOptions(options)

    # Increasing the outlineThreshold prevents creating subgraphs of cheap Ops
    # such as add or reshapeInplace.
    # Instead only reusing ops with a highSubgraphValue such as matmul or normalisation.
    options.outlineThreshold = 10.0
    if args.execution_mode == "PIPELINE":
        options.enablePipelining = True
        options.autoRecomputation = popart.RecomputationType.Pipeline

    elif args.execution_mode == "PHASED":
        set_phased_options(options, engine_options, model, args)

    options.optimizerStateTensorLocationSettings = bert_optimizer_location_settings(args)

    if args.gradient_accumulation_factor > 1:
        options.enableGradientAccumulation = True
        options.accumulationFactor = args.gradient_accumulation_factor
        if args.gradient_reduction_type == "Mean":
            options.accumulationAndReplicationReductionType = popart.ReductionType.Mean

        # When not replicated SyncPattern.SinglePipeline will provide better overlap
        # than this option.
        if args.optimizer_state_offchip and device_is_replicated(args):
            options.accumulateOuterFragmentSettings = popart.AccumulateOuterFragmentSettings(
                popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized, [0])
    if args.engine_cache is not None:
        options.enableEngineCaching = True
        options.cachePath = args.engine_cache
    if args.profile:
        options.enableEngineCaching = False
    options.instrumentWithHardwareCycleCounter = args.report_hw_cycle_count
    options.disableGradAccumulationTensorStreams = True
    if args.max_copy_merge_size == -1:
        logger.debug("No copy merge size limit applied")
    else:
        logger.warning(
            f"Copy merge size limit set to {args.max_copy_merge_size}")
        engine_options["opt.maxCopyMergeSize"] = str(args.max_copy_merge_size)

    # Adding {"fullyConnectedPass", "TRAINING_BWD"} to some matmuls causes large
    # transposes before operations.
    if args.disable_fully_connected_pass:
        if args.task == "SQUAD" and args.sequence_length == 384:
            logger.warning(
                "Fully connected pass has been disabled. This may cause SQuAD 384 12-layer to go OOM.")
        options.enableFullyConnectedPass = False

    if args.inference and args.engine_cache is not None and not args.variable_weights_inference:
        logger.warning("Using engine cache with constant weights. Checkpoint weights will be ignored. "
                       "Use the `--variable-weights-inference` flag if checkpoint weights should be used.")

    if args.variable_weights_inference:
        options.constantWeights = False

    if args.group_host_syncs:
        options.groupHostSync = True

    if args.internal_exchange_optimisation_target is not None:
        engine_options["opt.internalExchangeOptimisationTarget"] = str(args.internal_exchange_optimisation_target)

    engine_options["debug.allowOutOfMemory"] = "true"
    engine_options["opt.useAutoloader"] = "true"

    options.engineOptions = engine_options

    # Set synthetic data mode (if active)
    if args.synthetic_data:
        if args.synthetic_data_initializer == "zeros":
            options.syntheticDataMode = popart.SyntheticDataMode.Zeros
        else:
            options.syntheticDataMode = popart.SyntheticDataMode.RandomNormal
        logger.info(
            f"Running with Synthetic Data Type '{options.syntheticDataMode}'")
    return options


def bert_session_patterns(args):
    patterns = popart.Patterns()
    if args.task != "SQUAD":
        patterns.enablePattern("DisableAttnDropoutBwdPattern", False)

    if args.execution_mode == ExecutionMode.PHASED:
        patterns.enablePattern("TiedGatherPattern", False)
        patterns.enablePattern("SparseAccumulatePattern", False)

    if args.execution_mode == ExecutionMode.PIPELINE and args.recompute_checkpoint_every_layer and any(map(lambda l: l > 1, args.layers_per_ipu)):
        patterns.enablePattern("AccumulatePriorityPattern", True)

    if args.task == "PRETRAINING" and args.execution_mode != ExecutionMode.PHASED and args.gradient_accumulation_factor <= 1 and not args.inference:
        patterns.enablePattern("TiedGatherPattern", False)
        patterns.enablePattern("SparseAccumulatePattern", False)
        logger.warning("Running Pretraining without Gradient Accumulation will disable optimisations "
                       "for the Word Embedding weight. This will increase memory usage. "
                       "Consider enabling Gradient Accumulation.")

    if args.optimizer == "SGD" and args.optimizer_state_offchip and args.execution_mode != ExecutionMode.PHASED:
        patterns.enablePattern("TiedGatherPattern", False)
        logger.warning("Remote Optimizer State with SGD/SGD+M is not a recommended configuration")

    return patterns


def compile_graph_checked(args, session):
    start_time = time.time()
    session.prepareDevice()
    end_time = time.time()
    compile_time = end_time - start_time
    logger.info(f"Compiled. Duration {compile_time} seconds")
    if args.profile:
        popvision.save_app_info({"compile_time": compile_time})


def bert_training_session(model, args, feed, loss, device,
                          optimizer_factory):
    options = bert_session_options(args, model)

    patterns = bert_session_patterns(args)

    proto = model.builder.getModelProto()

    optimizer = optimizer_factory.create()

    logger.info("Creating Session")
    session_kwargs = dict(fnModel=proto,
                          loss=loss,
                          deviceInfo=device,
                          optimizer=optimizer,
                          dataFlow=feed,
                          patterns=patterns,
                          userOptions=options)
    if args.use_popdist:
        import horovod.popart as hvd
        session = hvd.DistributedTrainingSession(**session_kwargs)
    else:
        session = popart.TrainingSession(**session_kwargs)

    logger.info("Compiling Training Graph")
    compile_graph_checked(args, session)

    if (args.use_popdist and popdist_root(args)) or not args.use_popdist:
        logger.info("Loading optimizer state")
        parameters = model.builder.getTrainableTensorIds()

        stream = {}
        init_copy = model.initializers.copy()

        for p in parameters:
            target_shape = tuple(model.builder.getTensorShape(p))
            if p != "Embedding/Embedding_Dict" or model.config.embedding_serialization_vocab_steps == 1:
                loading_scales = [args.scale_loaded_optimizer_state, args.scale_loaded_optimizer_state**2]
                for i, prefix in enumerate((popart.reservedAccl1Prefix(), popart.reservedAccl2Prefix())):
                    target = prefix + p
                    if target in model.initializers:
                        value = model.initializers[target]
                        init_copy.pop(target,  'No Key found')
                        init_copy.pop(p, 'No Key found')
                        # Handle the case where the checkpoint holds transposed values
                        if value.T.shape == target_shape:
                            value = value.T
                        stream[target] = value.copy()/loading_scales[i]
                    else:
                        logger.warning(f"Tensor {target} was not in checkpoint")
            # Embedding dict with serialization is a special case
            else:
                num_splits = model.config.embedding_serialization_vocab_steps
                accl1_value = model.initializers.get(popart.reservedAccl1Prefix() + p, None)
                accl2_value = model.initializers.get(popart.reservedAccl2Prefix() + p, None)
                init_copy.pop(p, 'No Key found')
                init_copy.pop(popart.reservedAccl1Prefix() + p, 'No Key found')
                init_copy.pop(popart.reservedAccl2Prefix() + p, 'No Key found')
                for split_ind in range(num_splits):
                    slice_size = target_shape[1]//num_splits
                    slice_start = split_ind * slice_size
                    slice_end = (split_ind + 1) * slice_size
                    name1 = f"Accl1___MLM/MatMul:MatMulRhsGradOp_Slice:0/{split_ind}".replace("0/0", "0")
                    name2 = f"Accl2___MLM/MatMul:MatMulRhsGradOp_Slice:0/{split_ind}".replace("0/0", "0")
                    if accl1_value is not None:
                        stream[name1] = accl1_value.T[:, slice_start:slice_end].copy()/args.scale_loaded_optimizer_state
                    if accl2_value is not None:
                        stream[name2] = accl2_value.T[:, slice_start:slice_end].copy()/args.scale_loaded_optimizer_state**2

        if len(init_copy) != 0:
            logger.warning(f"The following data {init_copy.keys()} in the initializer have NOT been used. "
                           "The model might be different from the one that generated the checkpoint.")

        weightsIo = popart.PyWeightsIO(stream)
        session.writeWeights(weightsIo)

    if args.use_popdist:
        logger.info("Broadcasting weights to all instances")
        hvd.broadcast_weights(session)

    session.weightsFromHost()
    session.setRandomSeed(args.seed)

    anchors = session.initAnchorArrays()

    return session, anchors


def bert_inference_session(model, args, feed, device):
    options = bert_session_options(args, model)

    patterns = bert_session_patterns(args)

    proto = model.builder.getModelProto()

    logger.info("Creating Session")
    session = popart.InferenceSession(fnModel=proto,
                                      deviceInfo=device,
                                      dataFlow=feed,
                                      patterns=patterns,
                                      userOptions=options)

    logger.info("Compiling Inference Graph")
    compile_graph_checked(args, session)

    session.weightsFromHost()
    session.setRandomSeed(args.seed)

    anchors = session.initAnchorArrays()

    return session, anchors


def bert_writer(args):
    writer = None
    if args.enable_tensorboard and args.log_dir is not None and popdist_root(args):
        log_name = f"{os.path.basename(args.checkpoint_dir)}."\
            f"{datetime.datetime.now().isoformat()}"
        log_dir = os.path.join(
            args.log_dir, log_name)
        writer = SummaryWriter(log_dir=log_dir)
    return writer


def get_online_evaluation_dataset(model, args):
    config = model.config
    shapeOf = model.builder.getTensorShape
    inputs = [model.inputs["input_ids"], model.masks[0], model.inputs["segment_ids"], model.inputs["position_ids"]]
    inputs += [model.inputs["masked_lm_positions"], model.labels["masked_lm_ids"], model.inputs["masked_lm_weights"]]
    if not config.reduce_nsp_overhead:
        inputs += [model.inputs["nsp_positions"], model.labels["nsp_labels"], model.labels["nsp_weights"]]
    tensor_shapes = [(tensorId, shapeOf(tensorId)) for tensorId in inputs]

    dataset = get_packed_pretraining_dataset(tensor_shapes,
                                             input_files=args.on_the_spot_validation_files,
                                             seed=0,
                                             sequence_length=config.sequence_length,
                                             mask_tokens=config.mask_tokens,
                                             max_sequences_per_pack=args.max_sequences_per_pack,
                                             vocab_length=config.vocab_length,
                                             batch_size=config.batch_size,
                                             batches_per_step=args.batches_per_step,
                                             accumulation_factor=args.gradient_accumulation_factor,
                                             replication_factor=args.replication_factor,
                                             duplication_factor=1,
                                             shuffle=False,
                                             generated_data=False,
                                             epochs_to_cache=1,
                                             drop_remainder=False)
    return dataset


def get_bert_dataset(model, args, embedding_dict=None, positional_dict=None, merge_both_embeddings=False):
    config = model.config
    shapeOf = model.builder.getTensorShape

    if config.task == "PRETRAINING" and not args.use_prepacked_pretraining_dataset:
        inputs = [model.inputs["input_ids"], model.masks[0], model.inputs["segment_ids"]]
        inputs += [model.inputs["masked_lm_positions"], model.labels["masked_lm_ids"]]
        if not config.reduce_nsp_overhead:
            inputs += [model.labels["nsp_labels"]]
        tensor_shapes = [(tensorId, shapeOf(tensorId)) for tensorId in inputs]

        return get_pretraining_dataset(
            tensor_shapes,
            input_files=args.input_files,
            seed=args.seed,
            sequence_length=config.sequence_length,
            mask_tokens=config.mask_tokens,
            vocab_length=config.vocab_length,
            batch_size=config.batch_size,
            batches_per_step=args.batches_per_step,
            accumulation_factor=args.gradient_accumulation_factor,
            replication_factor=args.replication_factor,
            duplication_factor=args.duplication_factor,
            shuffle=args.shuffle,
            generated_data=args.generated_data or args.synthetic_data,
            epochs_to_cache=args.epochs_to_cache,
            continue_training_from_epoch=args.continue_training_from_epoch,
            use_popdist=args.use_popdist,
            popdist_size=args.popdist_size,
            popdist_rank=args.popdisk_rank)

    if config.task == "PRETRAINING" and args.use_prepacked_pretraining_dataset:
        inputs = [model.inputs["input_ids"], model.masks[0], model.inputs["segment_ids"], model.inputs["position_ids"]]
        inputs += [model.inputs["masked_lm_positions"], model.labels["masked_lm_ids"], model.inputs["masked_lm_weights"]]
        if not config.reduce_nsp_overhead:
            inputs += [model.inputs["nsp_positions"], model.labels["nsp_labels"], model.labels["nsp_weights"]]
        tensor_shapes = [(tensorId, shapeOf(tensorId)) for tensorId in inputs]

        return get_packed_pretraining_dataset(
            tensor_shapes,
            input_files=args.input_files,
            seed=args.seed,
            sequence_length=config.sequence_length,
            mask_tokens=config.mask_tokens,
            max_sequences_per_pack=args.max_sequences_per_pack,
            vocab_length=config.vocab_length,
            batch_size=config.batch_size,
            batches_per_step=args.batches_per_step,
            accumulation_factor=args.gradient_accumulation_factor,
            replication_factor=args.replication_factor,
            duplication_factor=args.duplication_factor,
            shuffle=args.shuffle,
            generated_data=args.generated_data or args.synthetic_data,
            epochs_to_cache=args.epochs_to_cache,
            drop_remainder=not args.inference,
            continue_training_from_epoch=args.continue_training_from_epoch,
            use_popdist=args.use_popdist,
            popdist_size=args.popdist_size,
            popdist_rank=args.popdist_rank)

    if config.task == "SQUAD":
        inputs = reduce(chain, [model.masks, list(model.labels.values()), list(model.inputs.values())[:3]])
        tensor_shapes = [(tensorId, shapeOf(tensorId)) for tensorId in inputs]

        # squad dataset does not support poprun yet
        ds = get_squad_dataset(
            tensor_shapes,
            input_file=args.input_files[0],
            output_dir=args.squad_results_dir,
            sequence_length=config.sequence_length,
            vocab_file=args.vocab_file,
            vocab_length=config.vocab_length,
            batch_size=config.batch_size,
            batches_per_step=args.batches_per_step,
            embedding_dict=embedding_dict,
            positional_dict=positional_dict,
            merge_both_embeddings=merge_both_embeddings,
            accumulation_factor=args.gradient_accumulation_factor,
            replication_factor=args.replication_factor,
            shuffle=args.shuffle,
            is_training=not args.inference,
            overwrite_cache=args.overwrite_cache,
            no_drop_remainder=args.no_drop_remainder,
            evaluate_script=args.squad_evaluate_script,
            generated_data=args.generated_data or args.synthetic_data,
            do_lower_case=args.do_lower_case,
            max_pipeline_stage=model.total_pipeline_stages if args.execution_mode == "PIPELINE" else 1,
            seed=args.seed,
            mpi_size=args.mpi_size,
            mpi_rank=args.mpi_rank,
            is_distributed=args.mpi_size > 1)
        return ds


def bert_average_metric(args, anchors, metrics):
    accumulated_stats = args.gradient_accumulation_factor * args.batches_per_step
    if len(metrics) > 1:
        metric = np.add(*[anchors[metric] for metric in metrics])
    else:
        metric = anchors[metrics[0]]
    return np.mean(metric / accumulated_stats)


def bert_output_stats(args, anchors, losses, accuracies):
    return (bert_average_metric(args, anchors, losses),
            bert_average_metric(args, anchors, accuracies))


def bert_pretraining_stats(args, anchors, losses, accuracies):
    losses = map(lambda loss: bert_average_metric(args, anchors, [loss]), losses)
    accuracies = map(lambda acc: bert_average_metric(args, anchors, [acc]), accuracies)
    return tuple(losses), tuple(accuracies)


def bert_pretraining_inference_stats(args, anchors, losses, accuracies):
    if args.inference_lm_perplexity:
        loss = bert_average_metric(args, anchors, losses[0])
    else:
        loss = None
    accuracies = map(lambda acc: bert_average_metric(args, anchors, [acc]), accuracies)
    return loss, tuple(accuracies)


def save_model_and_stats(args, session, writer, step, epoch=None, step_in_filename=False):
    if not args.no_model_save and popdist_root(args):
        save_file = "model"
        if epoch is not None:
            save_file += f"_{epoch}"
        if step_in_filename:
            save_file += f":{step}"
        save_file += '.onnx'
        save_path = os.path.join(args.checkpoint_dir, save_file)
        logger.info(f"Saving model to: {save_path}")
        session.modelToHost(save_path)
        if args.enable_tensorboard:
            utils.save_model_statistics(save_path, writer, step)


class Iteration:
    def __init__(self, args, batches_per_step, steps_per_epoch, writer, model, recording_steps=None):
        self.epoch = args.continue_training_from_epoch
        self.count = self.epoch * steps_per_epoch
        self.args = args
        self.suggested_loss_scaling = args.loss_scaling
        self.minimum_loss_scaling = 1
        self.epochs = args.epochs
        self.epochs_per_save = args.epochs_per_save
        self.steps_per_log = args.steps_per_log
        self.batches_per_step = batches_per_step
        self.total_sequences_so_far = 0
        self.sequences_per_step = deque(maxlen=recording_steps)
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = self.steps_per_epoch * self.epochs
        self.writer = writer

        # Compute the global batch size
        gbs = args.batch_size*args.gradient_accumulation_factor*args.replication_factor
        if args.use_popdist:
            gbs *= args.popdist_size

        # Account for packing (multiply by the average number of sequences in a pack)
        gbs *= args.avg_seq_per_pack
        self.global_batch_size = gbs

        if len(args.on_the_spot_validation_triggers) == 0:
            # When running the benchmark, the frequency of evaluation depends on the global batch size
            # Implement rule
            multiple = 25000
            eval_interval_samples = int((0.05 * (230.23 * gbs + 3000000))//multiple)*multiple
            eval_start = eval_interval_samples
            triggers = list(range(eval_start, args.max_training_sequences + eval_interval_samples, eval_interval_samples))
            self.on_the_spot_validation_triggers = triggers
            status_str = f"\nVALIDATION CONFIG:"
            status_str = f"\n\t Target accuracy: {mlm_accuracy_target}"
            status_str += f"\n\t Global batch size: {gbs}"
            status_str += f"\n\t Validation interval: {eval_interval_samples}"
            status_str += f"\n\t Validation triggers: {triggers}"
            logger.info(status_str)
        else:
            self.on_the_spot_validation_triggers = sorted(args.on_the_spot_validation_triggers)
            logger.info(f"Using pred-defined evaluation sample counts: {self.on_the_spot_validation_triggers}")
        self.task = args.task
        self.calculate_perplexity = args.inference_lm_perplexity

        # This should get overridden but will ensure we can always write a scalar to TB.
        self.learning_rate = 0
        if recording_steps is None:
            recording_steps = self.steps_per_epoch
        self.durations = deque(maxlen=recording_steps)
        self.cycles = deque(maxlen=recording_steps)
        if self.task == "PRETRAINING":
            self.mlm_losses = deque(maxlen=recording_steps)
            self.nsp_losses = deque(maxlen=recording_steps)
            self.mlm_accuracies = deque(maxlen=recording_steps)
            self.nsp_accuracies = deque(maxlen=recording_steps)
            if args.inference:
                self.stats_fn = bert_pretraining_inference_stats
            else:
                self.stats_fn = bert_pretraining_stats
        else:
            self.losses = deque(maxlen=recording_steps)
            self.accuracies = deque(maxlen=recording_steps)
            self.stats_fn = bert_output_stats

        if args.use_popdist:
            self.distributed = True
            self.steps_per_distributed_reduce = 1
        else:
            self.distributed = False

    def add_stats(self, duration, data, hw_cycles, *args):
        self.durations.append(duration)

        sequences_per_pack = data["input_mask"].max(-1, keepdims=True)
        sequences_in_sample = int(sequences_per_pack.sum())
        self.sequences_per_step.append(sequences_in_sample)

        if self.distributed:
            self.total_sequences_so_far += sum_distributed_data(sequences_in_sample)
        else:
            self.total_sequences_so_far += sequences_in_sample

        if hw_cycles:
            self.cycles.append(hw_cycles)
        loss, accuracy = self.stats_fn(*args)
        if self.writer is not None:
            self.writer.add_scalar("optimization/learning_rate",
                                   self.learning_rate,
                                   self.total_sequences_so_far)
            self.writer.add_scalar("optimization/beta1",
                                   self.beta1,
                                   self.total_sequences_so_far)
            self.writer.add_scalar("optimization/throughput",
                                   np.average(self.throughput),
                                   self.total_sequences_so_far)
            self.writer.add_scalar("optimization/number_of_parameter_updates",
                                   self.count*self.batches_per_step,
                                   self.total_sequences_so_far)
        if self.task == "PRETRAINING":
            self.mlm_losses.append(loss[0])
            if not self.args.reduce_nsp_overhead:
                self.nsp_losses.append(loss[1])
                self.nsp_accuracies.append(accuracy[1])
            if self.writer is not None:
                self.writer.add_scalar("loss/MLM",
                                       np.average(self.mlm_losses),
                                       self.total_sequences_so_far)
                if not self.args.reduce_nsp_overhead:
                    self.writer.add_scalar("loss/NSP",
                                           np.average(self.nsp_losses),
                                           self.total_sequences_so_far)

            # Calculate the sequence averaged MLM accuracies
            _, anchors, *_, accuracies = args
            mlm_accuracies = anchors[accuracies[0]]
            mlm_accuracy = mlm_accuracies.sum()/sequences_in_sample
            self.mlm_accuracies.append(mlm_accuracy)

            # report distributed stats
            if self.distributed and (self.count % self.steps_per_distributed_reduce) == 0:
                replica_avg = partial(average_distributed_deques, N=self.steps_per_distributed_reduce)
                self.durations = replica_avg(self.durations)
                if self.cycles:
                    self.cycles = replica_avg(self.cycles)
                self.mlm_losses = replica_avg(self.mlm_losses)
                self.mlm_accuracies = replica_avg(self.mlm_accuracies)
                if not self.args.reduce_nsp_overhead:
                    self.nsp_losses = replica_avg(self.nsp_losses)
                    self.nsp_accuracies = replica_avg(self.nsp_accuracies)

            if self.writer is not None:
                self.writer.add_scalar("accuracy/MLM",
                                       np.average(self.mlm_accuracies),
                                       self.total_sequences_so_far)
                if not self.args.reduce_nsp_overhead:
                    self.writer.add_scalar("accuracy/NSP",
                                           np.average(self.nsp_accuracies),
                                           self.total_sequences_so_far)

        else:
            self.losses.append(loss)
            self.accuracies.append(accuracy)
            if self.writer is not None:
                self.writer.add_scalar("loss",
                                       np.average(self.losses),
                                       self.total_sequences_so_far)
                self.writer.add_scalar("accuracy",
                                       np.average(self.accuracies),
                                       self.total_sequences_so_far)

    def add_inference_stats(self, duration, data, hw_cycles, *args):
        sequences_per_pack = data["input_mask"].max(-1, keepdims=True)
        sequences_in_sample = int(sequences_per_pack.sum())
        self.sequences_per_step.append(sequences_in_sample)
        self.total_sequences_so_far += sequences_in_sample
        self.durations.append(duration)
        if hw_cycles:
            self.cycles.append(hw_cycles)

        if self.task == "PRETRAINING":
            loss, accuracy = self.stats_fn(*args)

            # Calculate the sequence averaged MLM
            _, anchors, *_, accuracies = args
            mlm_accuracies = anchors[accuracies[0]]
            mlm_accuracy = mlm_accuracies.sum()/sequences_in_sample

            self.mlm_accuracies.append(mlm_accuracy)
            if not self.args.reduce_nsp_overhead:
                self.nsp_accuracies.append(accuracy[1])

            if loss is not None:
                self.mlm_losses.append(loss)

    @property
    def throughput(self):
        return np.divide(self.sequences_per_step, self.durations)

    def report_evaluation_accuracy(self, mlm_eval_accuracy):
        if self.writer is not None:
            self.writer.add_scalar("accuracy/MLM/eval",
                                   mlm_eval_accuracy,
                                   self.total_sequences_so_far)

    def report_stats(self):
        avg = np.average
        status_string = \
            f"Iteration: {self.count:6} " \
            f"Epoch: {self.count/self.steps_per_epoch:6.2f}/{self.epochs} "
        if self.task == "PRETRAINING":
            if not self.args.reduce_nsp_overhead:
                status_string += \
                    f"Loss (MLM NSP): {avg(self.mlm_losses):5.3f} {avg(self.nsp_losses):5.3f} " \
                    f"Accuracy (MLM NSP): {avg(self.mlm_accuracies):5.3f} {avg(self.nsp_accuracies):5.3f} "
            else:
                status_string += \
                    f"Loss (MLM): {avg(self.mlm_losses):5.3f}  " \
                    f"Accuracy (MLM): {avg(self.mlm_accuracies):5.3f} "
        else:
            status_string += \
                f"Loss: {avg(self.losses):5.3f} " \
                f"Accuracy: {avg(self.accuracies):5.3f} "
        status_string += \
            f"Learning Rate: {self.learning_rate:.5f} "
        status_string += \
            f"Duration: {avg(self.durations):6.4f} s " \
            f"Throughput: {avg(self.throughput):6.1f} samples/s"
        if self.cycles:
            status_string += f" Cycles: {int(avg(self.cycles))}"
        logger.info(status_string)

    def report_inference_stats(self, mean_latency, min_latency, max_latency, hw_cycles):
        avg = np.average
        status_string = \
            f"Iteration: {self.count:6} " \
            f"Duration: {avg(self.durations):6.4f} s " \
            f"Throughput: {avg(self.throughput):6.1f} samples/s"

        if self.task == "PRETRAINING":
            if not self.args.reduce_nsp_overhead:
                status_string += \
                    f" Accuracy (MLM NSP): {avg(self.mlm_accuracies):5.3f} {avg(self.nsp_accuracies):5.3f}"
            else:
                status_string += \
                    f" Accuracy (MLM NSP): {avg(self.mlm_accuracies):5.3f} "

            if self.calculate_perplexity:
                status_string += \
                    f" LM Perplexity: {np.exp(avg(self.mlm_losses)):5.3f}"

        if mean_latency is not None:
            status_string += f" Per-sample Latency: {mean_latency} {min_latency} {max_latency} seconds (mean min max)"
        if hw_cycles is not None:
            status_string += f" Cycles: {hw_cycles}"
        logger.info(status_string)


def bert_process_data(args,
                      session,
                      data,
                      anchors,
                      losses,
                      accuracies,
                      iteration: Iteration,
                      optimizer_factory: BaseOptimizerFactory):
    data["is_training"] = np.ones([args.replication_factor*args.batches_per_step*args.gradient_accumulation_factor], dtype=np.uint32)
    stepio = popart.PyStepIO(data, anchors)
    start = time.time()
    session.run(stepio)
    duration = time.time() - start
    hw_cycles = session.getCycleCount() if args.report_hw_cycle_count else None

    iteration.add_stats(duration,
                        data,
                        hw_cycles,
                        args,
                        anchors,
                        losses,
                        accuracies)

    if (iteration.count % iteration.steps_per_log) == 0:
        iteration.report_stats()

    if args.profile:
        sys.exit(0)

    # The following will only be true if:
    #   Learning rate mode is STEP and the current total step counter is in the schedule
    #   Learning rate mode is EPOCH and the current epoch has just changed to one in the schedule
    if optimizer_factory.should_update(iteration):
        optimizer = optimizer_factory.update_and_create(iteration)
        session.updateOptimizerFromHost(optimizer)

    iteration.count += 1


def compute_latency(args,
                    start_times,
                    end_times,
                    durations):
    if args.low_latency_inference:
        if not start_times or not end_times:
            logger.warning("No stepio callback times recorded. Using durations for fallback calculation.")
        else:
            return compute_latency_from_callbacks(start_times, end_times, args.batches_per_step)
    return compute_latency_from_durations(durations)


def bert_process_infer_data(args,
                            session,
                            data,
                            anchors,
                            logits,
                            iteration: Iteration,
                            start_times=None,
                            end_times=None,
                            stepio=None,
                            accuracies=None,
                            losses=None):
    if stepio is None:
        stepio = popart.PyStepIO(data, anchors)

    start = time.perf_counter()
    session.run(stepio)
    duration = time.perf_counter() - start
    hw_cycles = session.getCycleCount() if args.report_hw_cycle_count else None

    iteration.add_inference_stats(
        duration, data, hw_cycles, args, anchors, losses, accuracies)

    mean_latency, min_latency, max_latency = compute_latency(
        args, start_times, end_times, iteration.durations)

    if (iteration.count % iteration.steps_per_log) == 0 or \
       iteration.total_sequences_so_far >= mlperf_inference_samples:
        iteration.report_inference_stats(mean_latency, min_latency, max_latency, hw_cycles)

    if args.profile:
        sys.exit(0)

    iteration.count += 1

    if args.task == "PRETRAINING":
        return None

    return [anchors[logit] for logit in logits]


def bert_train_loop(args,
                    session,
                    writer,
                    dataset,
                    accuracies,
                    losses,
                    anchors,
                    iteration,
                    optimizer_factory: BaseOptimizerFactory):
    losses = [loss for loss in losses]
    mllogger = iteration.mllogger

    # During validation the optimizer should prevent all training
    start_epoch = iteration.epoch
    mlm_eval_accuracy = 0
    first_epoch_num = 0
    mllogger.start(mllog.constants.BLOCK_START, None, metadata={"epoch_count": 1, "first_epoch_num": first_epoch_num})
    for iteration.epoch in range(start_epoch, iteration.epochs):
        # Begin training block (it is unknown exactly how many samples there are in an epoch)

        for data in dataset:
            bert_process_data(args, session, data, anchors,
                              losses, accuracies, iteration, optimizer_factory)

            # Run evaluation on the same graph by zeroing out the optimizer
            if len(iteration.on_the_spot_validation_triggers) > 0:
                next_trigger = min(iteration.on_the_spot_validation_triggers)
                if iteration.total_sequences_so_far > next_trigger:
                    # Exit training block and enter validation block
                    mllogger.end(mllog.constants.BLOCK_STOP, None, metadata={"first_epoch_num": first_epoch_num})
                    first_epoch_num += 1

                    iteration.on_the_spot_validation_triggers.remove(next_trigger)
                    start = time.time()

                    # Stream the validation optimizer to device (this optimizer zeros out all learning)
                    validation_optimizer = optimizer_factory.validation_optimizer
                    session.updateOptimizerFromHost(validation_optimizer)

                    # Loop through the evaluation dataset to determine the total accuracy
                    # The evaluation dataset always uses 1 seq/sample for transparency
                    mlm_accuracy_data = []
                    indexing_data = []
                    for i, eval_data in enumerate(iteration.evaluation_dataset):
                        eval_data["is_training"] = np.zeros([args.replication_factor*args.batches_per_step*args.gradient_accumulation_factor], dtype=np.uint32)
                        stepio = popart.PyStepIO(eval_data, anchors)
                        session.run(stepio)
                        # Collect the accuracies from the anchors, and copy them over
                        mlm_accuracy_data.append(np.reshape(anchors[accuracies[0]].copy(), [-1, args.max_sequences_per_pack]))

                        # Pick out the sequences which contain data
                        tmp = np.arange(args.max_sequences_per_pack).reshape(1, args.max_sequences_per_pack)
                        indexing_data.append(eval_data["input_mask"].copy().reshape([-1, args.sequence_length]).max(-1, keepdims=True) > tmp)

                    # Concatenate the results from all the steps
                    mlm_accuracy_data = np.concatenate(mlm_accuracy_data)
                    indexing_data = np.concatenate(indexing_data)

                    # The dataset is padded up to batch size, such that the remainder is not dropped
                    # this padding should now be removed before determining accuracy
                    num_padding_samples = iteration.evaluation_dataset.loader.dataloader.num_padding_samples
                    mlm_accuracy_data = mlm_accuracy_data[:-num_padding_samples, :]
                    indexing_data = indexing_data[:-num_padding_samples, :]

                    # Use the indexing information to slice out non-zero sequences from the packed results
                    mlm_accuracy_data = mlm_accuracy_data[indexing_data]
                    eval_sample_count = len(mlm_accuracy_data)

                    # Average the accuracies
                    mlm_eval_accuracy = mlm_accuracy_data.mean()
                    iteration.report_evaluation_accuracy(mlm_eval_accuracy)

                    # Go back to using the training optimizer
                    training_optimizer = optimizer_factory.training_optimizer
                    session.updateOptimizerFromHost(training_optimizer)

                    # Log the accuracy
                    mllogger.event(key=mllog.constants.EVAL_ACCURACY, value=mlm_eval_accuracy,
                                   metadata={'epoch_num': iteration.total_sequences_so_far}, clear_line=True)

                    # If accuracy not at target, and there are still evaluation points left, cotinue training
                    if mlm_eval_accuracy < mlm_accuracy_target and len(iteration.on_the_spot_validation_triggers) > 0:
                        mllogger.start(mllog.constants.BLOCK_START, None, metadata={"epoch_count": 1, "first_epoch_num": first_epoch_num})
                    logger.info(f"Eval accuracy: {mlm_eval_accuracy:5.3f}")
                    logger.info(f"Evaluation took: {time.time() - start:4.3f} seconds")

            if mlm_eval_accuracy >= mlm_accuracy_target or len(iteration.on_the_spot_validation_triggers) == 0:
                iteration.report_stats()
                return mlm_eval_accuracy, eval_sample_count


def bert_infer_loop(args,
                    model,
                    session,
                    dataset,
                    logits,
                    anchors,
                    accuracies,
                    losses,
                    iteration: Iteration):
    save_results = args.task == "SQUAD" and not (args.synthetic_data or args.generated_data)

    if not losses:
        losses = None

    # Create the stepio once outside of the inference loop:
    static_data = {}
    start_times = defaultdict(list)
    end_times = defaultdict(list)

    stepio = None
    if args.low_latency_inference and args.task == "SQUAD":
        stepio = create_callback_stepio(static_data, anchors, start_times, end_times,
                                        dataset.batches_per_step)

    with realtime_scheduling(args.realtime_scheduler):
        for iteration.epoch in range(args.epochs_inference):
            mlm_accuracy_data = []
            for data in dataset:
                static_data.update({t: data[t] for t in model.inputs.values()})
                static_data.update({t: data[t] for t in model.masks})
                static_data.update({t: data[t] for t in model.labels})
                result = bert_process_infer_data(args, session, static_data, anchors,
                                                 logits, iteration,
                                                 start_times, end_times, stepio,
                                                 accuracies, losses)
                # Append the mlm accuracy outputs from the last step (populates anchors)
                mlm_accuracy_data.append(anchors[accuracies[0]][..., 0].copy())

                if result is not None and save_results and iteration.epoch == args.epochs_inference - 1:
                    dataset.add_results(data, result)
                start_times.clear()
                end_times.clear()

            # Combine all computed accuracies to determine the no_drop_remainder adjusted accuracy
            mlm_accuracy_data = np.concatenate(mlm_accuracy_data)
            num_padding_samples = dataset.loader.dataloader.num_padding_samples
            mlm_eval_accuracy = mlm_accuracy_data.flatten()[:-num_padding_samples]
            num_samples = len(mlm_eval_accuracy)
            mlm_eval_accuracy = mlm_eval_accuracy.mean()
            logger.info(f"Inference accuracy {mlm_eval_accuracy:3.4f} computed on {num_samples} samples")

    # If SQuAD save the predictions and run the evaulation script
    if save_results:
        dataset.write_predictions()


def bert_required_ipus(args, model):
    if args.execution_mode == "PHASED":
        if args.phased_execution_type == "DUAL":
            num_ipus = 2
        else:
            num_ipus = 1
    else:
        num_ipus = model.total_ipus
    num_ipus *= args.replication_factor
    return num_ipus


def bert_pretrained_initialisers(config, args):

    if args.synthetic_data:
        logger.info("Initialising from synthetic_data")
        return None

    if args.generated_data:
        logger.info("Initialising from generated_data")
        return None

    # The initialised weights will be broadcast after the session has been created
    if not popdist_root(args):
        return None

    init = None
    if args.onnx_checkpoint:
        logger.info(f"Initialising from ONNX checkpoint: {args.onnx_checkpoint}")
        init = utils.load_initializers_from_onnx(args.onnx_checkpoint)

    if args.tf_checkpoint:
        logger.info(f"Initialising from TF checkpoint: {args.tf_checkpoint}")
        init = load_initializers_from_tf(args.tf_checkpoint, True, config, args.task)

    if init is not None and args.execution_mode == "PHASED":
        init.update(**get_phased_initializers_from_default(args, init))

    return init


def main(args):
    set_library_seeds(args.seed)

    config = bert_config_from_args(args)

    # Log the model configuration
    mllogger = mllog.get_mllogger()
    filename = f"results/bert/result_{args.submission_run_index}.txt"
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/bert"):
        os.mkdir("results/bert")
    if os.path.exists(filename):
        os.remove(filename)

    mllog.config(filename=filename)
    mllogger.event(key=mllog.constants.SUBMISSION_ORG, value="Graphcore")
    mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value='bert')
    if args.submission_division == "closed":
        mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value=mllog.constants.CLOSED)
    else:
        mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value=mllog.constants.OPEN)
    mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value=mllog.constants.ONPREM)
    mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value=args.submission_platform)

    # Initialize and construct the model
    mllogger.start(mllog.constants.INIT_START, None)
    initializers = bert_pretrained_initialisers(config, args)

    logger.info("Building Model")
    # Specifying ai.onnx opset9 for the slice syntax
    model = get_model(config,
                      mode=args.execution_mode,
                      initializers=initializers,
                      block=None)

    # Define inputs, build graph, and construct the loss and accuracy ops
    bert_add_inputs(args, model)
    logits = bert_logits_graph(model, args.execution_mode)
    predictions, probs = bert_infer_graph(model, logits)
    losses, accuracies, final_loss, outputs_to_anchor = packed_bert_loss_and_accuracy(args, model, probs, logits, predictions)
    data_flow = popart.DataFlow(args.batches_per_step, outputs_to_anchor)
    writer = bert_writer(args)

    # Acquire accelerators
    device = acquire_device(args, bert_required_ipus(args, model))

    iteration = Iteration(
        args,
        # batches_per_step and steps_per_epoch will be set later when dataset is loaded
        batches_per_step=0,
        steps_per_epoch=0,
        writer=writer,
        model=model,
        recording_steps=args.aggregate_metrics_over_steps)
    iteration.mllogger = mllogger

    # Report the required characteristics and hyperparameters of the run
    mllogger.event(mllog.constants.SEED, value=args.seed)
    mllogger.event(mllog.constants.MAX_SEQUENCE_LENGTH, value=args.sequence_length)
    mllogger.event(mllog.constants.GLOBAL_BATCH_SIZE, value=iteration.global_batch_size)
    mllogger.event(mllog.constants.GRADIENT_ACCUMULATION_STEPS, args.gradient_accumulation_factor)
    mllogger.event("opt_base_learning_rate", args.lr_bert_schedule["init_lr"])
    mllogger.event("opt_lamb_weight_decay_rate", args.weight_decay)
    mllogger.event(mllog.constants.OPT_LAMB_BETA_1, args.beta1 if args.beta1_schedule is None else args.beta1_schedule["init"])
    mllogger.event(mllog.constants.OPT_LAMB_BETA_2, args.beta2)
    mllogger.event(mllog.constants.OPT_LR_WARMUP_STEPS, args.lr_bert_schedule["num_warmup_steps"])
    mllogger.event("num_warmup_steps", args.lr_bert_schedule["num_warmup_steps"])
    mllogger.event("start_warmup_step", 0)
    mllogger.event("opt_learning_rate_training_steps", args.lr_bert_schedule["num_training_steps"])
    mllogger.event("opt_epsilon", 1e-6)  # default
    mllogger.event(mllog.constants.OPT_LAMB_LR_DECAY_POLY_POWER, 1.0)

    if args.inference:
        session, anchors = bert_inference_session(
            model, args, data_flow, device)
    else:
        optimizer_factory = ScheduledOptimizerFactory(args,
                                                      iteration,
                                                      args.optimizer,
                                                      model.tensors)

        session, anchors = bert_training_session(model,
                                                 args,
                                                 data_flow,
                                                 final_loss,
                                                 device,
                                                 optimizer_factory)

    # Start the benchmark run
    logger.info("Benchmark timer started")
    benchmark_start = time.perf_counter()
    mllogger.event(mllog.constants.CACHE_CLEAR, True)
    mllogger.start(key=mllog.constants.INIT_STOP)
    mllogger.start(key=mllog.constants.RUN_START)

    # Load the training dataset
    embedding_dict, positional_dict = model.get_model_embeddings()
    dataset = get_bert_dataset(model,
                               args,
                               embedding_dict,
                               positional_dict,
                               config.host_embedding == "MERGE")

    # Load the evaluation dataset (if provided)
    evaluation_dataset = None
    if len(args.on_the_spot_validation_files) > 0:
        evaluation_dataset = get_online_evaluation_dataset(model, args)
    iteration.evaluation_dataset = evaluation_dataset

    iteration.batches_per_step = dataset.batches_per_step
    iteration.steps_per_epoch = dataset.steps_per_epoch
    logger.info(f"Dataset length: {len(dataset)}")

    if args.inference:
        logger.info("Inference Started")
        bert_infer_loop(args, model, session,
                        dataset, logits, anchors,
                        accuracies, losses, iteration)
        logger.info("Inference Finished")
    else:
        logger.info("Training Started")
        mlm_eval_accuracy, eval_sample_count = bert_train_loop(args, session, writer,
                                                               dataset, accuracies, losses, anchors,
                                                               iteration, optimizer_factory)


        # Report the number of sequences in the evaluation set
        benchmark_duration = time.perf_counter() - benchmark_start
        logger.info("Training Finished")
        logger.info(f"Benchmark timer ended, time to train = {benchmark_duration}")

        mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=eval_sample_count)
        mllogger.event(key=mllog.constants.TRAIN_SAMPLES, value=iteration.total_sequences_so_far)

        # Determine the status of the run
        status = mllog.constants.SUCCESS if mlm_eval_accuracy >= mlm_accuracy_target else mllog.constants.ABORTED
        mllogger.start(key=mllog.constants.RUN_STOP, metadata={"status": status, "TTT": benchmark_duration})

        # Evaluation has already finished by this point as well, saving checkpoint for debugging only
        save_model_and_stats(args, session, writer, iteration.count)

    device.detach()

    return session, iteration


def setup_logger(log_level, handler=None):

    # Define a root config with a format which is simpler for console use
    root = logging.getLogger()
    root.setLevel(log_level)
    root_handler = logging.StreamHandler(sys.stdout)
    root_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s',
                                       '%Y-%m-%d %H:%M:%S')
    root_handler.setFormatter(root_formatter)
    root.handlers = [root_handler]
    if handler is not None:
        root.handlers += [handler]

    # Define a specific Handler for this file that removes the root name.
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if handler is not None:
        logger.addHandler(handler)
    logger.propagate = False


if __name__ == "__main__":

    args = utils.parse_bert_args()

    if args.profile:
        popvision.set_profiling_vars(args.profile_dir, args.profile_instrument)
        popvision.set_logging_vars()
        args_dict = vars(args)
        args_dict["hostname"] = socket.gethostname()
        args_dict["command"] = ' '.join(sys.argv)
        popvision.save_app_info(args_dict)
        logging_handler = popvision.get_profile_logging_handler()
    else:
        logging_handler = None

    setup_logger(logging.getLevelName(args.log_level), logging_handler)

    if args.wandb:
        import wandb
        wandb.init(project="popart-bert", sync_tensorboard=True)
        wandb_config = vars(args)
        wandb_config["global_batch_size"] = args.batch_size * args.replication_factor * args.gradient_accumulation_factor
        wandb.config.update(args)

    logger.info("Program Start")
    logger.info("Hostname: " + socket.gethostname())
    logger.info("Command Executed: " + str(sys.argv))

    # Run the main inference/training session by default
    if args.inference or not args.no_training:
        main(args)

    logger.info("Program Finished")
