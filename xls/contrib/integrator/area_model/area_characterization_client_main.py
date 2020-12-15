# Lint as: python3
#
# Copyright 2020 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sweeps to characterize datapoints from a synthesis server.

These datapoints can be used in an area model (where they may be interpolated)
-- the results emitted on stdout are in xls.delay_model.DelayModel prototext
format.
"""

import math 
import numpy as np

from typing import List, Sequence, Optional, Tuple

from absl import app
from absl import flags
from absl import logging

import grpc

from xls.delay_model import delay_model
from xls.delay_model import delay_model_pb2
from xls.delay_model import op_module_generator
from xls.ir.python import bits
from xls.synthesis import client_credentials
from xls.synthesis import synthesis_pb2
from xls.synthesis import synthesis_service_pb2_grpc

FLAGS = flags.FLAGS
flags.DEFINE_integer('port', 10000, 'Port to connect to synthesis server on.')

FREE_OPS = ('kArray kArrayConcat kArrayIndex kBitSlice kConcat kIdentity'
            'kLiteral kParam kReverse kTuple kTupleIndex kZeroExt kSignExt').split()

# Configure regression data sweeps
verbose=False
flat_bit_count_sweep_ceiling=71
flat_bit_count_sweep_floor=8
operand_count_sweep_ceiling=16
base_loop_stride=2
operand_loop_stride=1
nested_loop_stride_2x=base_loop_stride * 2
nested_loop_stride_3x=nested_loop_stride_2x * 3

# Configure cross fold validation
num_cross_validation_folds=5
max_fold_geomean_error = 0.15

# Helper functions for setting expression fields
def _set_add_expression(expr: delay_model_pb2.DelayExpression):
    expr.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.ADD

def _set_sub_expression(expr: delay_model_pb2.DelayExpression):
    expr.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.SUB

def _set_divide_expression(expr: delay_model_pb2.DelayExpression):
    expr.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.DIVIDE

def _set_max_expression(expr: delay_model_pb2.DelayExpression):
    expr.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.MAX

def _set_min_expression(expr: delay_model_pb2.DelayExpression):
    expr.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.MIN

def _set_multiply_expression(expr: delay_model_pb2.DelayExpression):
    expr.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.MULTIPLY

def _set_power_expression(expr: delay_model_pb2.DelayExpression):
    expr.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.POWER

def _set_constant_expression(expr: delay_model_pb2.DelayExpression, value: int):
    expr.constant = value

def _set_result_bit_count_expression_factor(expr: delay_model_pb2.DelayExpression, add_constant=0):
    if(add_constant != 0):
        _set_add_expression(expr)
        _set_result_bit_count_expression_factor(expr.rhs_expression)
        _set_constant_expression(expr.lhs_expression, add_constant)
    else:
        expr.factor.source = delay_model_pb2.DelayFactor.RESULT_BIT_COUNT

def _set_operand_count_expression_factor(expr: delay_model_pb2.DelayExpression, add_constant=0):
    if(add_constant != 0):
        _set_add_expression(expr)
        _set_operand_count_expression_factor(expr.rhs_expression)
        _set_constant_expression(expr.lhs_expression, add_constant)
    else:
        expr.factor.source = delay_model_pb2.DelayFactor.OPERAND_COUNT

def _set_operand_bit_count_expression_factor(expr: delay_model_pb2.DelayExpression, operand_idx: int,
    add_constant=0):
    if(add_constant != 0):
        _set_add_expression(expr)
        _set_operand_bit_count_expression_factor(expr.rhs_expression, operand_idx)
        _set_constant_expression(expr.lhs_expression, add_constant)
    else:
        expr.factor.source = delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT
        expr.factor.operand_number = operand_idx

def _new_expression(op_model: delay_model_pb2.OpModel) -> delay_model_pb2.DelayExpression:
    """Add a new expression to op_model and return"""
    return op_model.estimator.regression.expressions.add()

def _new_regression_op_model(model: delay_model_pb2.DelayModel, kop: str, result_bit_count: bool=False, 
    operand_count: bool=False, operand_bit_counts: List[int]  = []) -> delay_model_pb2.OpModel:
    """Add to 'model' a new regression model for op 'kop'.  result_bit_count, operand_count,
    and operand_bit_count factors are added according to the keyword args"""
    add_op_model = model.op_models.add(op=kop)
    add_op_model.estimator.regression.num_cross_validation_folds = num_cross_validation_folds
    add_op_model.estimator.regression.max_fold_geomean_error = max_fold_geomean_error
    if(result_bit_count):
        result_bits_expr = _new_expression(add_op_model)
        _set_result_bit_count_expression_factor(result_bits_expr)
    if(operand_count):
        operand_count_expr = _new_expression(add_op_model)
        _set_operand_count_expression_factor(operand_count_expr)
    for operand_idx in operand_bit_counts:
        operand_bit_count_expr = _new_expression(add_op_model) 
        _set_operand_bit_count_expression_factor(operand_bit_count_expr, operand_idx)
    return add_op_model


def _synth(stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
           verilog_text: str,
           top_module_name: str) -> synthesis_pb2.CompileResponse:
  request = synthesis_pb2.CompileRequest()
  request.module_text = verilog_text
  request.top_module_name = top_module_name
  logging.vlog(3, '--- Request')
  logging.vlog(3, request)

  return stub.Compile(request)


def _record_area(response: synthesis_pb2.CompileResponse, result: delay_model_pb2.DataPoint):
  """Extracts area result from the server response and writes it to the datapoint"""
  cell_histogram = response.instance_count.cell_histogram
  if "SB_LUT4" in cell_histogram:
    result.delay = cell_histogram["SB_LUT4"]
  else:
    result.delay = 0


def _build_data_point(
  op: str, kop: str, num_node_bits: int, num_operand_bits: List[int], 
  stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
  attributes: Sequence[Tuple[str, str]] = (),
  operand_attributes: Sequence[Tuple[str, int]] = (),
  operand_span_attributes: Sequence[Tuple[str, int, int]] = (),
  literal_operand: Optional[int] = None) -> delay_model_pb2.DataPoint:
  """Characterize an operation via synthesis server.

  Args:
    op: Operation name to use for generating an IR package; e.g. 'add'.
    kop: Operation name to emit into datapoints, generally in kConstant form for
      use in the delay model; e.g. 'kAdd'.
    num_node_bits: The number of bits of the operation result.
    num_operand_bits: The number of bits of each operation.
    stub: Handle to the synthesis server.

  Args forwarded to generate_ir_package:
    attributes: Attributes to include in the operation mnemonic. For example,
      "new_bit_count" in extend operations.
    operand_attributes: Attributes consisting of an operand to include in the 
      operation mnemonic. Each element of the sequence specifies the attribute
      name and the index of the first operand (within 'operand_types').
      For example, used for "defualt" in sel operations.
    operand_span_attributes: Attributes consisting of 1 or more operand to include 
      in the operation mnemonic. Each element of the sequence specifies the attribute
      name, the index of the first operand (within 'operand_types'), and 
      the number of operands in the span.
      For example, used for "cases" in sel operations.
    literal_operand: Optionally specifies that the given operand number should
      be substituted with a randomly generated literal instead of a function
      parameter.

  Returns:
    datapoint produced via the synthesis server.
  """
  op_type = f'bits[{num_node_bits}]'
  operand_types = []
  for operand_bit_count in num_operand_bits:
    operand_types.append(f'bits[{operand_bit_count}]')
  ir_text = op_module_generator.generate_ir_package(op, op_type,
                                                    operand_types,
                                                    attributes,
                                                    operand_attributes,
                                                    operand_span_attributes,
                                                    literal_operand) 
  module_name = f'{op}_{num_node_bits}'
  mod_generator_result = op_module_generator.generate_verilog_module(
      module_name, ir_text)
  top_name = module_name + '_wrapper'
  verilog_text = op_module_generator.generate_parallel_module(
      [mod_generator_result], top_name)

  response = _synth(stub, verilog_text, top_name)
  result = delay_model_pb2.DataPoint()
  _record_area(response, result)
  result.operation.op = kop
  result.operation.bit_count = num_node_bits
  for operand_bit_count in num_operand_bits:
    operand = result.operation.operands.add()
    operand.bit_count = operand_bit_count 
  return result

def _run_nary_op(
    op: str, kop: str, stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    num_inputs: int) -> List[delay_model_pb2.DataPoint]:
  """Characterizes an nary op"""
  results = []
  for bit_count in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, base_loop_stride):
    results.append(_build_data_point(op, kop,  bit_count, [bit_count]*num_inputs, stub))
    if(verbose):
      print('# nary_op: ' + op + ', ' + str(bit_count) + ' bits, ' 
          + str(num_inputs) + ' inputs' + ' --> ' + str(results[-1].delay))

  return results

def _run_linear_bin_op_and_add(
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization for the given linear bin_op and adds it to the model."""
  add_op_model = _new_regression_op_model(model, kop, result_bit_count=True)
  model.data_points.extend(_run_nary_op(op, kop, stub, num_inputs=2))
  # Validate model
  delay_model.DelayModel(model)

def _run_quadratic_bin_op_and_add(
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub, signed: bool = False) -> None:
  """Runs characterization for the quadratic bin_op and adds it to the model."""
  add_op_model = _new_regression_op_model(model, kop)

  # result_bit_count * result_bit_count
  # This is because the sign bit is special.
  expr = _new_expression(add_op_model)
  _set_multiply_expression(expr)
  constant = -1 if signed else 0
  _set_result_bit_count_expression_factor(expr.lhs_expression, add_constant = constant)
  _set_result_bit_count_expression_factor(expr.rhs_expression, add_constant = constant)

  model.data_points.extend(_run_nary_op(op, kop, stub, num_inputs=2))
  # Validate model
  delay_model.DelayModel(model)

def _run_unary_op_and_add(
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub, signed=False) -> None:
  """Runs characterization for the given unary_op and adds it to the model."""
  add_op_model = _new_regression_op_model(model, kop)
  expr = _new_expression(add_op_model)
  constant = -1 if signed else 0
  _set_result_bit_count_expression_factor(expr, add_constant=-1)

  model.data_points.extend(_run_nary_op(op, kop, stub, num_inputs=1))
  # Validate model
  delay_model.DelayModel(model)


def _run_nary_op_and_add(
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization for the given nary_op and adds it to the model."""
  add_op_model = _new_regression_op_model(model, kop)
  expr = _new_expression(add_op_model)
  _set_multiply_expression(expr)
  _set_result_bit_count_expression_factor(expr.lhs_expression)
  # Note - for most operand counts, this works much better as
  # operand_count-1.  However, for low operand count (e.g. 2,4)
  # we can fit multiple ops inside a LUT, so the number of operands
  # has little effect until higher operand counts. So, weirdly,
  # we get lower error overall by just using operand_count...
  _set_operand_count_expression_factor(expr.rhs_expression)

  for input_count in range(2, operand_count_sweep_ceiling, operand_loop_stride):
    model.data_points.extend(_run_nary_op(op, kop, stub, num_inputs=input_count))
  # Validate model
  delay_model.DelayModel(model)

def _run_single_bit_result_op_and_add (
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub, num_inputs: int) -> None:
  """Runs characterization for the given op that always produce a single-bit output"""
  """and adds it to the model. The op has one or more inputs, all of the same bit type"""
  add_op_model = _new_regression_op_model(model, kop, operand_bit_counts=[0])

  for input_bits in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, base_loop_stride):
    if(verbose):
      print('# reduction_op: ' + op + ', ' + str(input_bits) + ' bits')
    model.data_points.append(_build_data_point(op, kop, 1, [input_bits] * num_inputs, stub))

  # Validate model
  delay_model.DelayModel(model)

def _run_reduction_op_and_add(
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  return _run_single_bit_result_op_and_add(op, kop, model, stub, num_inputs=1)

def _run_comparison_op_and_add(
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  return _run_single_bit_result_op_and_add(op, kop, model, stub, num_inputs=2)

def _run_select_op_and_add (
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization for the select op"""
  add_op_model = _new_regression_op_model(model, kop)

  # operand_count * result_bit_count
  # Alternatively, try pow(2, operand_bit_count(0)) * result_bit_count
  expr = _new_expression(add_op_model)
  _set_multiply_expression(expr)
  _set_operand_count_expression_factor(expr.lhs_expression, add_constant=-2)
  _set_result_bit_count_expression_factor(expr.rhs_expression)

  # Enumerate cases and bitwidth.
  # Note: at 7 and 8 cases, there is a weird dip in LUTs at around 40 bits wide
  # Why? No idea...
  for num_cases in range(2, operand_count_sweep_ceiling, operand_loop_stride):
    for bit_count in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, base_loop_stride):
      cases_span_attribute = ('cases', 1, num_cases)

      # Handle differently if num_cases is a power of 2.
      select_bits = bits.min_bit_count_unsigned(num_cases-1)
      if math.pow(2, select_bits) == num_cases:
        model.data_points.append(_build_data_point(op, kop, bit_count, [select_bits] + ([bit_count] * num_cases), 
          stub, operand_span_attributes=[cases_span_attribute]))
      else:
        default_attribute = ('default', num_cases+1)
        model.data_points.append(_build_data_point(op, kop, bit_count, [select_bits] + ([bit_count] * (num_cases+1)), 
          stub, operand_span_attributes=[cases_span_attribute], operand_attributes=[default_attribute]))
      if(verbose):
        print('# select_op: ' + op + ', ' + str(bit_count) + ' bits, ' + str(num_cases) + ' cases' \
            + ' --> ' + str(model.data_points[-1]))

  # Validate model
  delay_model.DelayModel(model)


def _run_one_hot_select_op_and_add (
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization for the one hot select op"""
  add_op_model = _new_regression_op_model(model, kop)

  # operand_bit_count(0) * result_bit_count
  expr = _new_expression(add_op_model)
  _set_multiply_expression(expr)
  _set_operand_bit_count_expression_factor(expr.lhs_expression, 0)
  _set_result_bit_count_expression_factor(expr.rhs_expression)

  # Enumerate cases and bitwidth.
  for num_cases in range(2, operand_count_sweep_ceiling, operand_loop_stride):
    for bit_count in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, base_loop_stride):
      cases_span_attribute = ('cases', 1, num_cases)
      select_bits = num_cases
      model.data_points.append(_build_data_point(op, kop, bit_count, [select_bits] + ([bit_count] * num_cases), 
        stub, operand_span_attributes=[cases_span_attribute]))
      if(verbose):
        print('# one_hot_select_op: ' + op + ', ' + str(bit_count) + ' bits, ' + str(num_cases) + ' cases' \
            + ' --> ' + str(model.data_points[-1].delay))

  # Validate model
  delay_model.DelayModel(model)


def _run_encode_op_and_add (
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization for the encode op"""
  add_op_model = _new_regression_op_model(model, kop, operand_bit_counts = [0])

  # input_bits should be at least 2 bits.
  for input_bits in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, base_loop_stride):
    node_bits = bits.min_bit_count_unsigned(input_bits-1)
    model.data_points.append(_build_data_point(op, kop, node_bits, [input_bits], stub))
    if(verbose):
      print('# encode_op: ' + op + ', ' + str(input_bits) + ' input bits ' \
          + ' --> ' + str(model.data_points[-1].delay))

  # Validate model
  delay_model.DelayModel(model)


def _run_decode_op_and_add (
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization for the decode op"""
  add_op_model = _new_regression_op_model(model, kop, result_bit_count=True)

  # node_bits should be at least 2 bits.
  for node_bits in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, base_loop_stride):
    input_bits = bits.min_bit_count_unsigned(node_bits-1)
    model.data_points.append(_build_data_point(op, kop, node_bits, [input_bits], stub, 
      attributes=[('width', str(node_bits))]))
    if(verbose):
      print('# decode_op: ' + op + ', ' + str(node_bits) + ' bits ' \
          + ' --> ' + str(model.data_points[-1].delay))

  # Validate model
  delay_model.DelayModel(model)


def _run_dynamic_bit_slice_op_and_add(
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization for the dynamic bit slice op"""
  add_op_model = _new_regression_op_model(model, kop)

  # ~= result_bit_count * operand_bit_count[1] (start bits)
  # Hard to model this well - in theory, this should be something
  # more like result_bit_count * 2 ^ start bits.  However,
  # as we add more result bits, more work gets eliminated / reduced
  # (iff 2 ^ start bits + result width > input bits).
  mul_expr = _new_expression(add_op_model)
  _set_multiply_expression(mul_expr)
  _set_result_bit_count_expression_factor(mul_expr.lhs_expression)
  _set_operand_bit_count_expression_factor(mul_expr.rhs_expression, 1)

  # input_bits should be at least 2 bits
  idx = 0
  for input_bits in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, nested_loop_stride_3x):
    for start_bits in range(3, bits.min_bit_count_unsigned(input_bits-1)+1):
      for node_bits in range(1, input_bits, nested_loop_stride_3x):
        model.data_points.append(_build_data_point(op, kop, node_bits, [input_bits, start_bits], stub, 
          attributes=[('width', str(node_bits))]))
        if(verbose):
          print('# idx ' + str(idx) + ': dynamic_bit_slice_op: ' + op + ', ' + str(start_bits) + ' start bits, ' + str(input_bits) 
              + ' input_bits, ' + str(node_bits) + ' width' + ' --> ' + str(model.data_points[-1].delay))
        idx = idx + 1

  # Validate model
  delay_model.DelayModel(model)


def _run_one_hot_op_and_add(
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization for the one hot op"""
  add_op_model = _new_regression_op_model(model, kop)

  # operand_bit_counts[0] * operand_bit_count[0]
  # The highest priority gate has ~1 gate (actually, a wire)
  # The lowest priority gate has O(n) gates (reduction tree of higher
  # priority inputs)
  # sum(1->n) is quadratic
  expr = _new_expression(add_op_model)
  _set_multiply_expression(expr)
  _set_operand_bit_count_expression_factor(expr.lhs_expression, 0)
  _set_operand_bit_count_expression_factor(expr.rhs_expression, 0)

  for bit_count in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, base_loop_stride):
    # lsb / msb priority or the same logic but mirror image.
    model.data_points.append(_build_data_point(op, kop,  bit_count+1, [bit_count], stub,
      attributes=[('lsb_prio', 'true')]))
    if(verbose):
      print('# one_hot: ' + op + ', ' + str(bit_count) + ' input bits' \
          + ' --> ' + str(model.data_points[-1].delay))

  # Validate model
  delay_model.DelayModel(model)


def  _run_mul_op_and_add(
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization for a mul op"""

  def _get_big_op_bitcount_expr():
    """
    Returns larger bit count of the two operands
    max(multiplier_bit_count, multiplicand_bit_count)
    """
    expr = delay_model_pb2.DelayExpression()
    _set_max_expression(expr)
    _set_operand_bit_count_expression_factor(expr.lhs_expression, 0)
    _set_operand_bit_count_expression_factor(expr.rhs_expression, 1)
    return expr

  def _get_small_op_bitcount_expr():
    """
    Returns smaller bit count of the two operands
    min(multiplier_bit_count, multiplicand_bit_count)
    """
    expr = delay_model_pb2.DelayExpression()
    _set_min_expression(expr)
    _set_operand_bit_count_expression_factor(expr.lhs_expression, 0)
    _set_operand_bit_count_expression_factor(expr.rhs_expression, 1)
    return expr

  def _get_zero_expr():
    """
    Returns a constant 0 expression.
    """
    expr = delay_model_pb2.DelayExpression()
    _set_constant_expression(expr, 0)
    return expr

  def _get_meaningful_width_bits_expr():
    """
    Returns the maximum number of meaningful result bits
    multiplier_bit_count + multiplicand_bit_count
    """
    expr = delay_model_pb2.DelayExpression()
    _set_add_expression(expr)
    _set_operand_bit_count_expression_factor(expr.lhs_expression, 0)
    _set_operand_bit_count_expression_factor(expr.rhs_expression, 1)
    return expr

  def _get_bounded_width_offset_domain(begin_expr, end_expr):
    """
    Gives the bounded offset of result_bit_count into 
    the range[begin_expr, end_expr]
    e.g. for begin_expr = 2, end_expr = 4, we map:
    1 --> 0
    2 --> 0
    3 --> 1
    4 --> 2
    5 --> 2
    6 --> 2
    etc.
    expr = min(end_expr, max(begin_expr, result_bit_count)) - begin_expr
    """
    expr = delay_model_pb2.DelayExpression()
    _set_sub_expression(expr)
    expr.rhs_expression.CopyFrom(begin_expr)

    min_expr = expr.lhs_expression
    _set_min_expression(min_expr)
    min_expr.lhs_expression.CopyFrom(end_expr)
    
    max_expr= min_expr.rhs_expression
    _set_max_expression(max_expr)
    max_expr.lhs_expression.CopyFrom(begin_expr)
    _set_result_bit_count_expression_factor(max_expr.rhs_expression)

    return expr

  def _get_rectangle_area(width_expr, height_expr):
    """
    Return the area of a rectangle of dimensions width_expr * height_expr
    """
    expr = delay_model_pb2.DelayExpression()
    _set_multiply_expression(expr)
    expr.lhs_expression.CopyFrom(width_expr)
    expr.rhs_expression.CopyFrom(height_expr)
    return expr

  def _get_triangle_area(width_expr):
    """
    Return the area of a isosceles right triangle with width width_expr
    expr = _get_triangle_area(width_expr, width_expr) / 2
    """
    expr = delay_model_pb2.DelayExpression()
    _set_divide_expression(expr)
    sqr_expression = _get_rectangle_area(width_expr, width_expr)
    expr.lhs_expression.CopyFrom(sqr_expression)
    _set_constant_expression(expr.rhs_expression, 2)
    return expr

  def _get_partial_triangle_area(width_expr, max_width_expr):
    """
    Return the area of a partial isosceles right triangle with width width_expr
    and maximum width max_width_expr
    e.g.
         |    /|    -|
         |   / |     |
         |  /  |     | 
         | /   |     | 
         |/    |     | 
         |     |     | heigh = maximum_width
        /|     |     | 
       / |area |     | 
      /  |     |     | 
     /   |     |     | 
    /____|_____|    -| 
         |

         |_____|
          width

    |___________|
    maximum_width

    expr = rectangle_area(width, maximum_width) - triangle_area(width)

    """
    expr = delay_model_pb2.DelayExpression()
    _set_sub_expression(expr)

    rectangle_expr = _get_rectangle_area(width_expr, max_width_expr)
    expr.lhs_expression.CopyFrom(rectangle_expr)

    triangle_expr = _get_triangle_area(width_expr)
    expr.rhs_expression.CopyFrom(triangle_expr)
    return expr



  """
  Compute for multiply can be divided into 3 regions.

  Regions:
   A         B      C
       |---------|-----   -
      /|         |    /    | 
     / |         |   /     |
    /  |         |  /      | height = min(op[0], op[1])
   /   |         | /       |
  /    |         |/        |
  -----|---------|        - 

  |______________|
  max(op[0],op[1])

  |____|
  min(op[0], op[1])

  |____________________|
  max(op[0],op[1]) + min(op[0], op[1])

  *Math works out the same whether op[0] or [op1] is larger.
   
  """

  '''
  expr = area(region C) + (area(region B) + area(region A))
  '''
  # Top level add
  add_op_model = _new_regression_op_model(model, kop)
  outer_add_expr= _new_expression(add_op_model)
  _set_add_expression(outer_add_expr)

  # precompute min/max(op[0], op[1])
  big_op_bitcount_expr = _get_big_op_bitcount_expr()
  small_op_bitcount_expr = _get_small_op_bitcount_expr()

  # Region C
  region_c_domain_expr = _get_bounded_width_offset_domain(_get_zero_expr(), small_op_bitcount_expr)
  region_c_area_expr = _get_triangle_area(region_c_domain_expr)
  outer_add_expr.lhs_expression.CopyFrom(region_c_area_expr)

  # Inner add
  inner_add_expr = outer_add_expr.rhs_expression
  _set_add_expression(inner_add_expr)

  # Region B
  #region_b_domain_expr = _get_bounded_width_offset_domain(small_op_bitcount_expr, big_op_bitcount_expr)
  region_b_domain_expr = _get_bounded_width_offset_domain(small_op_bitcount_expr, big_op_bitcount_expr)
  region_b_area_expr = _get_rectangle_area(region_b_domain_expr, small_op_bitcount_expr)
  inner_add_expr.lhs_expression.CopyFrom(region_b_area_expr)

  # Region A
  region_a_domain_expr = _get_bounded_width_offset_domain(big_op_bitcount_expr, 
      _get_meaningful_width_bits_expr())
  region_a_area_expr = _get_partial_triangle_area(region_a_domain_expr, small_op_bitcount_expr)
  inner_add_expr.rhs_expression.CopyFrom(region_a_area_expr)


  # All bit counts should be at least 2
  for mplier_count in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, nested_loop_stride_3x):
    for mcand_count in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, nested_loop_stride_3x):
      for node_count in range(flat_bit_count_sweep_floor, flat_bit_count_sweep_ceiling, nested_loop_stride_3x):
        model.data_points.append(_build_data_point(op, kop, node_count, [mplier_count, mcand_count], stub))
        if(verbose):
          print('# mul: ' + op + ', ' + str(mplier_count) + ' * ' + str(mcand_count) + 
              ', result bits ' + str(node_count) + ' --> ' +str(model.data_points[-1].delay))

  # Validate model
  delay_model.DelayModel(model)


def run_characterization(
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization via 'stub', DelayModel to stdout as prototext."""
  model = delay_model_pb2.DelayModel()

  # Bin ops
  _run_linear_bin_op_and_add('add', 'kAdd', model, stub)
  _run_linear_bin_op_and_add('sub', 'kSub', model, stub)
  # Observed shift data is noisy.
  _run_linear_bin_op_and_add('shll', 'kShll', model, stub)
  _run_linear_bin_op_and_add('shrl', 'kShrl', model, stub)
  _run_linear_bin_op_and_add('shra', 'kShra', model, stub)

  _run_quadratic_bin_op_and_add('sdiv', 'kSDiv', model, stub, signed=True)
  _run_quadratic_bin_op_and_add('smod', 'kSMod', model, stub, signed=True)
  _run_quadratic_bin_op_and_add('udiv', 'kUDiv', model, stub)
  _run_quadratic_bin_op_and_add('umod', 'kUMod', model, stub)

  # Unary ops
  _run_unary_op_and_add('neg', 'kNeg', model, stub, signed=True)
  _run_unary_op_and_add('not', 'kNot', model, stub)

  # Nary ops
  _run_nary_op_and_add('and', 'kAnd', model, stub)
  _run_nary_op_and_add('nand', 'kNAnd', model, stub)
  _run_nary_op_and_add('nor', 'kNor', model, stub)
  _run_nary_op_and_add('or', 'kOr', model, stub)
  _run_nary_op_and_add('xor', 'kXor', model, stub)

  # Reduction ops
  _run_reduction_op_and_add('and_reduce', 'kAndReduce', model, stub)
  _run_reduction_op_and_add('or_reduce', 'kOrReduce', model, stub)
  _run_reduction_op_and_add('xor_reduce', 'kXorReduce', model, stub)

  # Comparison ops
  _run_comparison_op_and_add('eq', 'kEq', model, stub)
  _run_comparison_op_and_add('ne', 'kNe', model, stub)
  # Note: Could optimize for sign - accuracy gains from
  # sign have been marginal so far, though. These ops
  # also cost less than smul / sdiv anyway.
  _run_comparison_op_and_add('sge', 'kSGe', model, stub)
  _run_comparison_op_and_add('sgt', 'kSGt', model, stub)
  _run_comparison_op_and_add('sle', 'kSLe', model, stub)
  _run_comparison_op_and_add('slt', 'kSLt', model, stub)
  _run_comparison_op_and_add('uge', 'kUGe', model, stub)
  _run_comparison_op_and_add('ugt', 'kUGt', model, stub)
  _run_comparison_op_and_add('ule', 'kULe', model, stub)
  _run_comparison_op_and_add('ult', 'kULt', model, stub)


  # Select ops
  # For functions only called for 1 op, could just encode
  # op and kOp into function.  However, perfer consistency
  # and readability of passing them in as args.
  # Note: Select op observed data is really weird, see _run_select_op_and_add
  _run_select_op_and_add('sel', 'kSel', model, stub)
  _run_one_hot_select_op_and_add('one_hot_sel', 'kOneHotSel', model, stub)
  
  # Encode ops
  _run_encode_op_and_add('encode', 'kEncode', model, stub)
  _run_decode_op_and_add('decode', 'kDecode', model, stub)

  # Dynamic bit slice op
  _run_dynamic_bit_slice_op_and_add('dynamic_bit_slice', 'kDynamicBitSlice', model, stub)

  # One hot op
  _run_one_hot_op_and_add('one_hot', 'kOneHot', model, stub)

  # Mul ops
  # Note: Modeling smul w/ sign bit as with sdiv decreases accuracy.
  _run_mul_op_and_add('smul', 'kSMul', model, stub)
  _run_mul_op_and_add('umul', 'kUMul', model, stub)

  # Add free ops.
  for free_op in FREE_OPS:
    entry = model.op_models.add(op=free_op)
    entry.estimator.fixed = 0

  # Final validation
  delay_model.DelayModel(model)

  print('# proto-file: xls/delay_model/delay_model.proto')
  print('# proto-message: xls.delay_model.DelayModel')
  print(model)


def main(argv):
  if len(argv) != 1:
    raise app.UsageError('Unexpected arguments.')

  channel_creds = client_credentials.get_credentials()
  with grpc.secure_channel(f'localhost:{FLAGS.port}', channel_creds) as channel:
    grpc.channel_ready_future(channel).result()
    stub = synthesis_service_pb2_grpc.SynthesisServiceStub(channel)

    run_characterization(stub)


if __name__ == '__main__':
  app.run(main)
