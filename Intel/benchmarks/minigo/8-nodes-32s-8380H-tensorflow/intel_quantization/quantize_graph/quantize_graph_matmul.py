#  -*- coding: utf-8 -*-
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from intel_quantization.quantize_graph.quantize_graph_common import QuantizeGraphHelper as helper
from intel_quantization.quantize_graph.quantize_graph_base import QuantizeNodeBase

import logging


class FuseNodeStartWithMatmul(QuantizeNodeBase):
    patterns = [["MatMul", "BiasAdd"], ["MatMul", "BiasAdd", "Relu"]]

    def __init__(self, input_graph, output_node_names, perchannel,
                 start_node_name):
        super(FuseNodeStartWithMatmul,
              self).__init__(input_graph, output_node_names, perchannel,
                             start_node_name)

        self.sorted_patterns = sorted(self.patterns,
                                      key=lambda i: len(i),
                                      reverse=True)
        self.fusion_op_type = set(fusion[0] for fusion in self.patterns)
        self.fusion_mapping = {
            'MatMulBiasAdd': self.apply_matmul_biasadd_fusion,
            'MatMulBiasAddRelu': self.apply_matmul_biasadd_relu_fusion,
        }

    def apply_matmul_biasadd_relu_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)
        weight_name = normal_inputs[1]

        self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weight_name].node,
            self.per_channel)

        skip_node_name.append(weight_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[0]:
                logging.debug("matched node {} with input {}".format(
                    node.name, node.input))

                logging.debug("apply_conv_biasadd_fusion")

                quantized_node_name = node.name + "_eightbit_quantized_mat_mul"
                bias_node_name = self.node_name_mapping[
                    match_node_name[1]].node.input[1]
                relu_node_name = match_node_name[2]
                all_input_names = self._add_eightbit_prologue_nodes(
                    matched_node.node.name)
                quantized_node_input_names = all_input_names[:2] + [
                    bias_node_name
                ] + all_input_names[2:] + control_inputs

                quantized_matmul_node = helper.create_node(
                    "QuantizedMatMulWithBiasAndRelu", quantized_node_name,
                    quantized_node_input_names)

                helper.copy_attr(quantized_matmul_node, "transpose_a",
                                 node.attr["transpose_a"])
                helper.copy_attr(quantized_matmul_node, "transpose_b",
                                 node.attr["transpose_b"])
                helper.set_attr_dtype(quantized_matmul_node, "T1",
                                      dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2",
                                      dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput",
                                      dtypes.qint32)

                self.add_output_graph_node(quantized_matmul_node)
                requantize_type = dtypes.qint8 if self.per_channel else dtypes.quint8

                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, requantize_type, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, relu_node_name, requantize_type)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_matmul_biasadd_fusion(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)
        weight_name = normal_inputs[1]

        self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weight_name].node,
            self.per_channel)

        skip_node_name.append(weight_name)

        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[0]:
                logging.debug("matched node {} with input {}".format(
                    node.name, node.input))

                logging.debug("apply_conv_biasadd_fusion")

                quantized_node_name = node.name + "_eightbit_quantized_mat_mul"
                bias_node_name = self.node_name_mapping[
                    match_node_name[1]].node.input[1]
                all_input_names = self._add_eightbit_prologue_nodes(
                    matched_node.node.name)
                quantized_node_input_names = all_input_names[:2] + [
                    bias_node_name
                ] + all_input_names[2:] + control_inputs

                quantized_matmul_node = helper.create_node(
                    "QuantizedMatMulWithBias", quantized_node_name,
                    quantized_node_input_names)

                helper.copy_attr(quantized_matmul_node, "transpose_a",
                                 node.attr["transpose_a"])
                helper.copy_attr(quantized_matmul_node, "transpose_b",
                                 node.attr["transpose_b"])
                helper.set_attr_dtype(quantized_matmul_node, "T1",
                                      dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2",
                                      dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput",
                                      dtypes.qint32)
                helper.set_attr_dtype(quantized_matmul_node, "Tbias",
                                      dtypes.float32)

                self.add_output_graph_node(quantized_matmul_node)
                requantize_type = dtypes.qint8

                quantize_down_name = self._add_quantize_down_nodes(
                    node, quantized_node_name, requantize_type, False)
                self._intel_cpu_add_dequantize_result_node(
                    quantize_down_name, match_node_name[1], requantize_type)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def get_longest_fuse(self):
        self._get_op_list()
        matched_rule, _ = self._is_match(self.sorted_patterns)
        return matched_rule

    def apply_the_transform(self):
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match(self.sorted_patterns)
        if matched_node_name:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = ''.join(matched_rule)
            if fusion_name in self.fusion_mapping:
                self.fusion_mapping[fusion_name](matched_node_name)
            else:
                print("Unknown match {}".format(fusion_name))

            self.input_graph = self.output_graph
            self._reset_output_node_maps()

            self.output_graph = self.remove_redundant_quantization(
                self.output_graph)
            return self.output_graph
        else:
            logging.debug("No more match, exit...")
            return self.input_graph
