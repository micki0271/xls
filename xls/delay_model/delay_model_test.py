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
#
# Lint as: python3

"""Tests for xls.delay_model.models.delay_model."""

import re
from typing import Text

from google.protobuf import text_format
from xls.delay_model import delay_model
from xls.delay_model import delay_model_pb2
from absl.testing import absltest


def _parse_data_point(s: Text) -> delay_model_pb2.DataPoint:
  """Parses a text proto representation of a DataPoint."""
  return text_format.Parse(s, delay_model_pb2.DataPoint())


def _parse_operation(s: Text) -> delay_model_pb2.Operation:
  """Parses a text proto representation of an Operation."""
  return text_format.Parse(s, delay_model_pb2.Operation())


class DelayModelTest(absltest.TestCase):

  def assertEqualIgnoringWhitespace(self, a: Text, b: Text):
    """Asserts the two strings are equal ignoring all whitespace."""

    def _strip_ws(s: Text) -> Text:
      return re.sub(r'\s+', '', s.strip())

    self.assertEqual(_strip_ws(a), _strip_ws(b))

  def assertEqualIgnoringWhitespaceAndFloats(self, a: Text, b: Text):
    """Asserts the two strings are equal ignoring whitespace and floats.

    Floats are replaced with 0.0.

    Args:
      a: Input string.
      b: Input string.
    """

    def _floats_to_zero(s: Text) -> Text:
      return re.sub(r'-?[0-9]+\.[0-9e+-]+', '0.0', s)

    self.assertEqualIgnoringWhitespace(_floats_to_zero(a), _floats_to_zero(b))

  def test_fixed_estimator(self):
    foo = delay_model.FixedEstimator('kFoo', 42)
    self.assertEqual(
        foo.operation_delay(_parse_operation('op: "kBar" bit_count: 1')), 42)
    self.assertEqual(
        foo.operation_delay(_parse_operation('op: "kBar" bit_count: 123')), 42)
    self.assertEqual(foo.cpp_delay_code('node'), 'return 42;')

  def test_alias_estimator(self):
    foo = delay_model.AliasEstimator('kFoo', 'kBar')
    self.assertEqual(foo.cpp_delay_code('node'), """return BarDelay(node);""")

  def test_bounding_box_estimator(self):
    data_points_str = [
        'operation { op: "kBar" bit_count: 3 operands { bit_count: 7 } } ' +
        'delay: 33 delay_offset: 10',
        'operation { op: "kBar" bit_count: 12 operands { bit_count: 42 } }' +
        'delay: 100 delay_offset: 0',
        'operation { op: "kBar" bit_count: 32 operands { bit_count: 10 } }' +
        'delay: 123 delay_offset: 1',
        'operation { op: "kBar" bit_count: 64 operands { bit_count: 64 } }' +
        'delay: 1234 delay_offset: 0',
    ]
    result_bit_count = delay_model_pb2.DelayFactor()
    result_bit_count.source = delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    operand0_bit_count = delay_model_pb2.DelayFactor()
    operand0_bit_count.source = delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT
    operand0_bit_count.operand_number = 0
    bar = delay_model.BoundingBoxEstimator(
        'kBar', (result_bit_count, operand0_bit_count),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertEqual(
        bar.operation_delay(
            _parse_operation(
                'op: "kBar" bit_count: 1 operands { bit_count: 2 }')), 23)
    self.assertEqual(
        bar.operation_delay(
            _parse_operation(
                'op: "kBar" bit_count: 10 operands { bit_count: 32 }')), 100)
    self.assertEqual(
        bar.operation_delay(
            _parse_operation(
                'op: "kBar" bit_count: 32 operands { bit_count: 9 }')), 122)
    self.assertEqual(
        bar.operation_delay(
            _parse_operation(
                'op: "kBar" bit_count: 32 operands { bit_count: 33 }')), 1234)
    self.assertEqual(
        bar.operation_delay(
            _parse_operation(
                'op: "kBar" bit_count: 64 operands { bit_count: 64 }')), 1234)
    self.assertEqualIgnoringWhitespace(
        bar.cpp_delay_code('node'), """
          if (node->GetType()->GetFlatBitCount() <= 3 &&
              node->operand(0)->GetType()->GetFlatBitCount() <= 7) {
            return 23;
          }
          if (node->GetType()->GetFlatBitCount() <= 12 &&
              node->operand(0)->GetType()->GetFlatBitCount() <= 42) {
            return 100;
          }
          if (node->GetType()->GetFlatBitCount() <= 32 &&
              node->operand(0)->GetType()->GetFlatBitCount() <= 10) {
            return 122;
          }
          if (node->GetType()->GetFlatBitCount() <= 64 &&
              node->operand(0)->GetType()->GetFlatBitCount() <= 64) {
            return 1234;
          }
          return absl::UnimplementedError(
              "Unhandled node for delay estimation: " +
              node->ToStringWithOperandTypes());
        """)
    with self.assertRaises(delay_model.Error) as e:
      bar.operation_delay(
          _parse_operation(
              'op: "kBar" bit_count: 65 operands { bit_count: 64 }'))
    self.assertIn('Operation outside bounding box', str(e.exception))

  def test_one_factor_regression_estimator(self):
    data_points_str = [
        'operation { op: "kFoo" bit_count: 2 } delay: 210 delay_offset: 10',
        'operation { op: "kFoo" bit_count: 4 } delay: 410 delay_offset: 10',
        'operation { op: "kFoo" bit_count: 6 } delay: 610 delay_offset: 10',
        'operation { op: "kFoo" bit_count: 8 } delay: 810 delay_offset: 10',
        'operation { op: "kFoo" bit_count: 10 } delay: 1010 delay_offset: 10',
    ]
    result_bit_count = delay_model_pb2.DelayExpression()
    result_bit_count.factor.source = delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    foo = delay_model.RegressionEstimator(
        'kFoo', (result_bit_count,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation('op: "kFoo" bit_count: 2')),
        200,
        delta=2)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation('op: "kFoo" bit_count: 3')),
        300,
        delta=2)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation('op: "kFoo" bit_count: 5')),
        500,
        delta=2)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation('op: "kFoo" bit_count: 42')),
        4200,
        delta=2)
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * static_cast<float>(node->GetType()->GetFlatBitCount()) +
              0.0 * std::log2(static_cast<float>(node->GetType()->GetFlatBitCount())));
        """)

  def test_one_regression_estimator_operand_count(self):

    def gen_operation(operand_count):
      operands_str = 'operands { bit_count: 42 }'
      return 'op: "kFoo" bit_count: 42 %s' % ' '.join(
          [operands_str] * operand_count)

    data_points_str = [
        'operation { %s } delay: 10 delay_offset: 0' % gen_operation(1),
        'operation { %s } delay: 11 delay_offset: 0' % gen_operation(2),
        'operation { %s } delay: 12 delay_offset: 0' % gen_operation(4),
        'operation { %s } delay: 13 delay_offset: 0' % gen_operation(8),
    ]
    operand_count = delay_model_pb2.DelayExpression()
    operand_count.factor.source = delay_model_pb2.DelayFactor.Source.OPERAND_COUNT
    foo = delay_model.RegressionEstimator(
        'kFoo', (operand_count,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(1))), 10, delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(4))), 12, delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(256))), 18, delta=1)
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * static_cast<float>(node->operand_count()) +
              0.0 * std::log2(static_cast<float>(node->operand_count())));
        """)

  def test_two_factor_regression_estimator(self):

    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 100 delay_offset: 0' % gen_operation(1, 2),
        'operation { %s } delay: 125 delay_offset: 0' % gen_operation(4, 1),
        'operation { %s } delay: 150 delay_offset: 0' % gen_operation(4, 6),
        'operation { %s } delay: 175 delay_offset: 0' % gen_operation(7, 13),
        'operation { %s } delay: 200 delay_offset: 0' % gen_operation(10, 12),
        'operation { %s } delay: 400 delay_offset: 0' % gen_operation(30, 15),
    ]
    result_bit_count = delay_model_pb2.DelayExpression()
    result_bit_count.factor.source = delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    operand_bit_count = delay_model_pb2.DelayExpression()
    operand_bit_count.factor.source = delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT
    operand_bit_count.factor.operand_number = 1
    foo = delay_model.RegressionEstimator(
        'kFoo', (result_bit_count, operand_bit_count),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(1, 2))),
        100,
        delta=10)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(10, 12))),
        200,
        delta=10)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(8, 8))),
        200,
        delta=50)
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * static_cast<float>(node->GetType()->GetFlatBitCount()) +
              0.0 * std::log2(static_cast<float>(node->GetType()->GetFlatBitCount())) +
              0.0 * static_cast<float>(node->operand(1)->GetType()->GetFlatBitCount()) +
              0.0 * std::log2(static_cast<float>(node->operand(1)->GetType()->GetFlatBitCount())));
        """)

  def test_fixed_op_model(self):
    op_model = delay_model.OpModel(
        text_format.Parse('op: "kFoo" estimator { fixed: 42 }',
                          delay_model_pb2.OpModel()), ())
    self.assertEqual(op_model.op, 'kFoo')
    self.assertEqual(
        op_model.estimator.operation_delay(
            _parse_operation('op: "kBar" bit_count: 123')), 42)
    self.assertEqualIgnoringWhitespace(
        op_model.cpp_delay_function(),
        """absl::StatusOr<int64> FooDelay(Node* node) {
             return 42;
           }""")

  def test_fixed_op_model_with_specialization(self):
    op_model = delay_model.OpModel(
        text_format.Parse(
            'op: "kFoo" estimator { fixed: 42 } '
            'specializations { kind: OPERANDS_IDENTICAL estimator { fixed: 123 } }',
            delay_model_pb2.OpModel()), ())
    self.assertEqual(op_model.op, 'kFoo')
    self.assertEqual(
        op_model.estimator.operation_delay(
            _parse_operation('op: "kBar" bit_count: 123')), 42)
    self.assertEqual(
        op_model.specializations[
            delay_model_pb2.SpecializationKind.OPERANDS_IDENTICAL]
        .operation_delay(_parse_operation('op: "kBar" bit_count: 123')), 123)
    self.assertEqualIgnoringWhitespace(
        op_model.cpp_delay_function(), """
          absl::StatusOr<int64> FooDelay(Node* node) {
            if (std::all_of(node->operands().begin(), node->operands().end(),
                [&](Node* n) { return n == node->operand(0); })) {
              return 123;
            }
            return 42;
          }
        """)

  def test_regression_estimator_binop_delay_expression_add(self):
    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 3   delay_offset: 0' % gen_operation(1, 2),
        'operation { %s } delay: 5   delay_offset: 0' % gen_operation(4, 1),
        'operation { %s } delay: 10  delay_offset: 0' % gen_operation(4, 6),
        'operation { %s } delay: 20  delay_offset: 0' % gen_operation(7, 13),
        'operation { %s } delay: 22  delay_offset: 0' % gen_operation(10, 12),
        'operation { %s } delay: 45  delay_offset: 0' % gen_operation(30, 15),
    ]
    expression = delay_model_pb2.DelayExpression()
    expression.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.ADD
    expression.lhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    expression.rhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT
    expression.rhs_expression.factor.operand_number = 1

    foo = delay_model.RegressionEstimator(
        'kFoo', (expression,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(15, 15))),
        30,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(45, 20))),
        65,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(20, 45))),
        65,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(10, 25))),
        35,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(100, 250))),
        350,
        delta=1)

    expression_str = r"""(static_cast<float>(node->GetType()->GetFlatBitCount())
        + static_cast<float>(node->operand(1)->GetType()->GetFlatBitCount()))"""
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * {} +
              0.0 * std::log2({}));
        """.format(expression_str, expression_str))


  def test_regression_estimator_binop_delay_expression_sub(self):
    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 1   delay_offset: 0' % gen_operation(2, 1),
        'operation { %s } delay: 3   delay_offset: 0' % gen_operation(4, 1),
        'operation { %s } delay: 2   delay_offset: 0' % gen_operation(6, 4),
        'operation { %s } delay: 6   delay_offset: 0' % gen_operation(13, 7),
        'operation { %s } delay: 7   delay_offset: 0' % gen_operation(12, 5),
        'operation { %s } delay: 15  delay_offset: 0' % gen_operation(30, 15),
    ]
    expression = delay_model_pb2.DelayExpression()
    expression.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.SUB
    expression.lhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    expression.rhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT
    expression.rhs_expression.factor.operand_number = 1

    foo = delay_model.RegressionEstimator(
        'kFoo', (expression,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(16, 15))),
        1,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(45, 20))),
        25,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(20, 5))),
        15,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(100, 25))),
        75,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(250, 100))),
        150,
        delta=1)

    expression_str = r"""(static_cast<float>(node->GetType()->GetFlatBitCount())
        - static_cast<float>(node->operand(1)->GetType()->GetFlatBitCount()))"""
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * {} +
              0.0 * std::log2({}));
        """.format(expression_str, expression_str))
 

  def test_regression_estimator_binop_delay_expression_divide(self):
    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 5   delay_offset: 0' % gen_operation(10, 2),
        'operation { %s } delay: 4   delay_offset: 0' % gen_operation(4, 1),
        'operation { %s } delay: 4  delay_offset: 0' % gen_operation(20, 5),
        'operation { %s } delay: 7  delay_offset: 0' % gen_operation(49, 7),
        'operation { %s } delay: 5  delay_offset: 0' % gen_operation(50, 10),
        'operation { %s } delay: 2  delay_offset: 0' % gen_operation(30, 15),
    ]
    expression = delay_model_pb2.DelayExpression()
    expression.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.DIVIDE
    expression.lhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    expression.rhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT
    expression.rhs_expression.factor.operand_number = 1

    foo = delay_model.RegressionEstimator(
        'kFoo', (expression,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(15, 15.0))),
        1,
        delta=1)
    # Note: operation_delay will round to nearest int.
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(45, 20))),
        2.25,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(20, 45))),
        0.4444,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(81, 9))),
        9,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(256, 2))),
        128,
        delta=1)

    expression_str = r"""(static_cast<float>(node->GetType()->GetFlatBitCount())
        / static_cast<float>(node->operand(1)->GetType()->GetFlatBitCount()))"""
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * {} +
              0.0 * std::log2({}));
        """.format(expression_str, expression_str))


  def test_regression_estimator_binop_delay_expression_max(self):
    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 2   delay_offset: 0' % gen_operation(1, 2),
        'operation { %s } delay: 4   delay_offset: 0' % gen_operation(4, 1),
        'operation { %s } delay: 6  delay_offset: 0' % gen_operation(4, 6),
        'operation { %s } delay: 13  delay_offset: 0' % gen_operation(7, 13),
        'operation { %s } delay: 12  delay_offset: 0' % gen_operation(10, 12),
        'operation { %s } delay: 30  delay_offset: 0' % gen_operation(30, 15),
    ]
    expression = delay_model_pb2.DelayExpression()
    expression.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.MAX
    expression.lhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    expression.rhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT
    expression.rhs_expression.factor.operand_number = 1

    foo = delay_model.RegressionEstimator(
        'kFoo', (expression,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(15, 15))),
        15,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(45, 20))),
        45,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(20, 45))),
        45,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(10, 25))),
        25,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(100, 250))),
        250,
        delta=1)

    expression_str = r"""std::max(static_cast<float>(node->GetType()->GetFlatBitCount()),
        static_cast<float>(node->operand(1)->GetType()->GetFlatBitCount()))"""
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * {} +
              0.0 * std::log2({}));
        """.format(expression_str, expression_str))


  def test_regression_estimator_binop_delay_expression_min(self):
    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 1   delay_offset: 0' % gen_operation(1, 2),
        'operation { %s } delay: 1   delay_offset: 0' % gen_operation(4, 1),
        'operation { %s } delay: 4  delay_offset: 0' % gen_operation(4, 6),
        'operation { %s } delay: 7  delay_offset: 0' % gen_operation(7, 13),
        'operation { %s } delay: 10  delay_offset: 0' % gen_operation(10, 12),
        'operation { %s } delay: 15  delay_offset: 0' % gen_operation(30, 15),
    ]
    expression = delay_model_pb2.DelayExpression()
    expression.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.MIN
    expression.lhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    expression.rhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT
    expression.rhs_expression.factor.operand_number = 1

    foo = delay_model.RegressionEstimator(
        'kFoo', (expression,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(15, 15))),
        15,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(45, 20))),
        20,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(20, 45))),
        20,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(10, 25))),
        10,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(100, 250))),
        100,
        delta=1)

    expression_str = r"""std::min(static_cast<float>(node->GetType()->GetFlatBitCount()),
        static_cast<float>(node->operand(1)->GetType()->GetFlatBitCount()))"""
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * {} +
              0.0 * std::log2({}));
        """.format(expression_str, expression_str))

  def test_regression_estimator_binop_delay_expression_multiply(self):
    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 2   delay_offset: 0' % gen_operation(1, 2),
        'operation { %s } delay: 4   delay_offset: 0' % gen_operation(4, 1),
        'operation { %s } delay: 24  delay_offset: 0' % gen_operation(4, 6),
        'operation { %s } delay: 91  delay_offset: 0' % gen_operation(7, 13),
        'operation { %s } delay: 120 delay_offset: 0' % gen_operation(10, 12),
        'operation { %s } delay: 450 delay_offset: 0' % gen_operation(30, 15),
    ]
    expression = delay_model_pb2.DelayExpression()
    expression.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.MULTIPLY
    expression.lhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT
    expression.rhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT
    expression.rhs_expression.factor.operand_number = 1

    foo = delay_model.RegressionEstimator(
        'kFoo', (expression,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(15, 15))),
        225,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(45, 20))),
        900,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(20, 45))),
        900,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(10, 25))),
        250,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(2, 8))),
        16,
        delta=1)

    expression_str = r"""(static_cast<float>(node->GetType()->GetFlatBitCount())
        * static_cast<float>(node->operand(1)->GetType()->GetFlatBitCount()))"""
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * {} +
              0.0 * std::log2({}));
        """.format(expression_str, expression_str))

  def test_regression_estimator_binop_delay_expression_power(self):
    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 2   delay_offset: 0' % gen_operation(1, 2),
        'operation { %s } delay: 4   delay_offset: 0' % gen_operation(2, 1),
        'operation { %s } delay: 8   delay_offset: 0' % gen_operation(3, 6),
        'operation { %s } delay: 16  delay_offset: 0' % gen_operation(4, 13),
        'operation { %s } delay: 32  delay_offset: 0' % gen_operation(5, 12),
        'operation { %s } delay: 64  delay_offset: 0' % gen_operation(6, 15),
    ]
    expression = delay_model_pb2.DelayExpression()
    expression.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.POWER
    expression.lhs_expression.constant = 2
    expression.rhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT

    foo = delay_model.RegressionEstimator(
        'kFoo', (expression,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(7, 15))),
        128,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(8, 20))),
        256,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(9, 45))),
        512,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(10, 25))),
        1024,
        delta=1)

    expression_str = r"""pow(static_cast<float>(2),
        static_cast<float>(node->GetType()->GetFlatBitCount()))"""
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * {} +
              0.0 * std::log2({}));
        """.format(expression_str, expression_str))

  def test_regression_estimator_constant_delay_expression(self):
    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 99  delay_offset: 0' % gen_operation(1, 2),
        'operation { %s } delay: 99  delay_offset: 0' % gen_operation(4, 1),
        'operation { %s } delay: 99  delay_offset: 0' % gen_operation(4, 6),
        'operation { %s } delay: 99  delay_offset: 0' % gen_operation(7, 13),
        'operation { %s } delay: 99  delay_offset: 0' % gen_operation(10, 12),
        'operation { %s } delay: 99  delay_offset: 0' % gen_operation(30, 15),
    ]
    expression = delay_model_pb2.DelayExpression()
    expression.constant = 99

    # Not especially usuful tests since regression includes a
    # constant variable that could mask issues...
    foo = delay_model.RegressionEstimator(
        'kFoo', (expression,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(15, 15))),
        99,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(45, 20))),
        99,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(20, 45))),
        99,
        delta=1)

    # This test is useful though!
    expression_str = r"""static_cast<float>(99)"""
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * {} +
              0.0 * std::log2({}));
        """.format(expression_str, expression_str))

  def test_regression_estimator_nested_delay_expression(self):
    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    data_points_str = [
        'operation { %s } delay: 3   delay_offset: 0' % gen_operation(1, 2),
        'operation { %s } delay: 5   delay_offset: 0' % gen_operation(4, 1),
        'operation { %s } delay: 10  delay_offset: 0' % gen_operation(4, 6),
        'operation { %s } delay: 20  delay_offset: 0' % gen_operation(7, 13),
        'operation { %s } delay: 20  delay_offset: 0' % gen_operation(10, 12),
        'operation { %s } delay: 20  delay_offset: 0' % gen_operation(30, 15),
    ]
    # min(1, RESULT_BIT_COUNT + OPERAND_BIT_COUNT)
    expression = delay_model_pb2.DelayExpression()
    expression.bin_op = delay_model_pb2.DelayExpression.BinaryOperation.MIN
    expression.lhs_expression.constant = 20
    expression.rhs_expression.bin_op = \
        delay_model_pb2.DelayExpression.BinaryOperation.ADD
    expression.rhs_expression.lhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.RESULT_BIT_COUNT 
    expression.rhs_expression.rhs_expression.factor.source = \
        delay_model_pb2.DelayFactor.Source.OPERAND_BIT_COUNT 
    expression.rhs_expression.rhs_expression.factor.operand_number = 1

    foo = delay_model.RegressionEstimator(
        'kFoo', (expression,),
        tuple(_parse_data_point(s) for s in data_points_str))
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(15, 15))),
        20,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(45, 20))),
        20,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(5, 5))),
        10,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(1, 15))),
        16,
        delta=1)
    self.assertAlmostEqual(
        foo.operation_delay(_parse_operation(gen_operation(6, 3))),
        9,
        delta=1)

    expression_str = r"""std::min(static_cast<float>(20), 
        (static_cast<float>(node->GetType()->GetFlatBitCount()) +
        static_cast<float>(node->operand(1)->GetType()->GetFlatBitCount())))"""
    self.assertEqualIgnoringWhitespaceAndFloats(
        foo.cpp_delay_code('node'), r"""
          return std::round(
              0.0 + 0.0 * {} +
              0.0 * std::log2({}));
        """.format(expression_str, expression_str))


  def test_regression_op_model_with_bounding_box_specialization(self):

    def gen_data_point(bit_count, delay, specialization=''):
      return _parse_data_point(
          'operation { op: "kFoo" bit_count: %d %s} delay: %d delay_offset: 0' %
          (bit_count, specialization, delay))

    op_model = delay_model.OpModel(
        text_format.Parse(
            'op: "kFoo" estimator { regression { expressions { factor { source: RESULT_BIT_COUNT } } } }'
            'specializations { kind: OPERANDS_IDENTICAL '
            'estimator { bounding_box { factors { source: RESULT_BIT_COUNT } } } }',
            delay_model_pb2.OpModel()),
        [gen_data_point(bc, 10 * bc) for bc in range(1, 10)] + [
            gen_data_point(bc, 2 * bc, 'specialization: OPERANDS_IDENTICAL')
            for bc in range(1, 3)
        ])
    self.assertEqual(op_model.op, 'kFoo')
    self.assertEqualIgnoringWhitespaceAndFloats(
        op_model.cpp_delay_function(), """
          absl::StatusOr<int64> FooDelay(Node* node) {
            if (std::all_of(node->operands().begin(), node->operands().end(),
                [&](Node* n) { return n == node->operand(0); })) {
              if (node->GetType()->GetFlatBitCount() <= 1) { return 2; }
              if (node->GetType()->GetFlatBitCount() <= 2) { return 4; }
              return absl::UnimplementedError(
                "Unhandled node for delay estimation: " +
                node->ToStringWithOperandTypes());
            }
            return std::round(
                0.0 + 0.0 * static_cast<float>(node->GetType()->GetFlatBitCount()) +
                0.0 * std::log2(static_cast<float>(node->GetType()->GetFlatBitCount())));
          }
        """)

  def test_regression_estimator_generate_validation_sets(self):
    raw_data_points = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
    training_dps, testing_dps = \
        zip(*delay_model.RegressionEstimator.generate_k_fold_cross_validation_train_and_test_sets(
            raw_data_points, num_cross_validation_folds=3))
    self.assertEqual(training_dps, ([(5, 5), (2, 2), (0, 0), (4, 4)],
                                    [(3, 3), (1, 1), (0, 0), (4, 4)],
                                    [(3, 3), (1, 1), (5, 5), (2, 2)]))
    self.assertEqual(testing_dps,  ([(3, 3), (1, 1)],
                                    [(5, 5), (2, 2)],
                                    [(0, 0), (4, 4)]))


  def test_regression_estimator_cross_validation_insufficient_data_for_folds(self):

    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    def gen_data_point(result_bit_count, operand_bit_count, delay):
      return 'data_points{{ operation {{ {} }} delay: {} delay_offset: 0}}'.format(
          gen_operation(result_bit_count, operand_bit_count), delay)

    data_points_str = [
        gen_data_point(1,2,100),
        gen_data_point(4,1,125),
        gen_data_point(4,6,150),
        gen_data_point(7,13,175),
        gen_data_point(10,12,200),
        gen_data_point(30,15,400),
    ]
    proto_text = """
    op_models {
      op: "kFoo"
      estimator {
        regression {
          expressions {
            factor {
              source: OPERAND_BIT_COUNT
              operand_number: 1
            }
          }
          expressions {
            factor {
              source: RESULT_BIT_COUNT
            }
          }
          num_cross_validation_folds: 8
          max_data_point_error: 99.0
          max_fold_geomean_error: 99.0
        }
      }
    }
    """
    proto_text = proto_text + '\n'.join(data_points_str)

    with self.assertRaises(delay_model.Error) as e:
      dm = delay_model.DelayModel(
          text_format.Parse(proto_text, delay_model_pb2.DelayModel()))
    self.assertEqualIgnoringWhitespaceAndFloats('kFoo: Too few data points to cross '
        'validate: 6 data points, 8 folds', str(e.exception))


  def test_regression_estimator_cross_validation_data_point_exceeds_max_error(self):

    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    def gen_data_point(result_bit_count, operand_bit_count, delay):
      return 'data_points{{ operation {{ {} }} delay: {} delay_offset: 0}}'.format(
          gen_operation(result_bit_count, operand_bit_count), delay)

    data_points_str = [
        gen_data_point(1,  2,   1 ),
        gen_data_point(2,  2,   2 ),
        gen_data_point(4,  1,   4 ),
        gen_data_point(5,  111, 5 ),
        gen_data_point(7,  13,  7 ),
        gen_data_point(8,  2,   8 ),
        gen_data_point(10, 12,  10),
        gen_data_point(15, 6,   15),
        gen_data_point(20, 40,  20),
        # Outlier
        gen_data_point(30, 15,  50),
        #
        gen_data_point(31, 2,    31 ),
        gen_data_point(35, 2,    35 ),
        gen_data_point(40, 30,   40 ),
        gen_data_point(45, 9,    45 ),
        gen_data_point(50, 4,    50 ),
        gen_data_point(55, 400,  55 ),
        gen_data_point(70, 10,   70 ),
        gen_data_point(100, 50,  100),
        gen_data_point(125, 15,  125),
        gen_data_point(150, 100, 150),
    ]
    proto_text = """
    op_models {
      op: "kFoo"
      estimator {
        regression {
          expressions {
            factor {
              source: RESULT_BIT_COUNT
            }
          }
          max_data_point_error: 0.3
        }
      }
    }
    """
    proto_text = proto_text + '\n'.join(data_points_str)

    with self.assertRaises(delay_model.Error) as e:
      dm = delay_model.DelayModel(
          text_format.Parse(proto_text, delay_model_pb2.DelayModel()))
    self.assertEqualIgnoringWhitespaceAndFloats('kFoo: Regression model failed k-fold '
        'cross validation for data point (30, 50) with absolute error 0.0'
        ' > max 0.0', str(e.exception))
    self.assertIn('> max 0.3', str(e.exception))


  def test_regression_estimator_cross_validation_data_point_exceeds_geomean_error(self):
    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    def gen_data_point(result_bit_count, operand_bit_count, delay):
      return 'data_points{{ operation {{ {} }} delay: {} delay_offset: 0}}'.format(
          gen_operation(result_bit_count, operand_bit_count), delay)

    data_points_str = [
        gen_data_point(1,  2,   1 ),
        gen_data_point(2,  2,   2 ),
        gen_data_point(4,  1,   4 ),
        gen_data_point(5,  111, 5 ),
        gen_data_point(7,  13,  7 ),
        gen_data_point(8,  2,   8 ),
        gen_data_point(10, 12,  10),
        gen_data_point(15, 6,   15),
        gen_data_point(20, 40,  20),
        gen_data_point(30, 15,  30),
        gen_data_point(31, 2,    31 ),
        gen_data_point(35, 2,    35 ),
        gen_data_point(40, 30,   40 ),
        gen_data_point(45, 9,    45 ),
        gen_data_point(50, 4,    50 ),
        gen_data_point(55, 400,  55 ),
        gen_data_point(70, 10,   70 ),
        gen_data_point(100, 50,  100),
        gen_data_point(125, 15,  125),
        gen_data_point(150, 100, 150),
    ]
    proto_text = """
    op_models {
      op: "kFoo"
      estimator {
        regression {
          expressions {
            factor {
              source: OPERAND_BIT_COUNT
              operand_number: 1
            }
          }
          max_fold_geomean_error: 0.1
        }
      }
    }
    """
    proto_text = proto_text + '\n'.join(data_points_str)

    # Build regression model with operand_bit_count (uncorrelated with delay)
    # as only factor.

    with self.assertRaises(delay_model.Error) as e:
      dm = delay_model.DelayModel(
          text_format.Parse(proto_text, delay_model_pb2.DelayModel()))
      self.assertEqualIgnoringWhitespaceAndFloats('kFoo: Regression model failed '
      'k-fold cross validation for test set with geomean error 0.0 > max 0.0',
        str(e.exception))
    self.assertIn('> max 0.1', str(e.exception))


  def test_regression_estimator_cross_validation_passes(self):

    def gen_operation(result_bit_count, operand_bit_count):
      return 'op: "kFoo" bit_count: %d operands { } operands { bit_count: %d }' % (
          result_bit_count, operand_bit_count)

    def gen_data_point(result_bit_count, operand_bit_count, delay):
      return 'data_points{{ operation {{ {} }} delay: {} delay_offset: 0}}'.format(
          gen_operation(result_bit_count, operand_bit_count), delay)

    data_points_str = [
        gen_data_point(1,  2,   1 ),
        gen_data_point(2,  2,   2 ),
        gen_data_point(4,  1,   4 ),
        gen_data_point(5,  111, 5 ),
        gen_data_point(7,  13,  7 ),
        gen_data_point(8,  2,   8 ),
        gen_data_point(10, 12,  10),
        gen_data_point(15, 6,   15),
        gen_data_point(20, 40,  20),
        gen_data_point(30, 15,  30),
        gen_data_point(31, 2,    31 ),
        gen_data_point(35, 2,    35 ),
        gen_data_point(40, 30,   40 ),
        gen_data_point(45, 9,    45 ),
        gen_data_point(50, 4,    50 ),
        gen_data_point(55, 400,  55 ),
        gen_data_point(70, 10,   70 ),
        gen_data_point(100, 50,  100),
        gen_data_point(125, 15,  125),
        gen_data_point(150, 100, 150),
    ]
    proto_text = """
    op_models {
      op: "kFoo"
      estimator {
        regression {
          expressions {
            factor {
              source: RESULT_BIT_COUNT
            }
          }
          max_data_point_error: 0.15
          max_fold_geomean_error: 0.075
        }
      }
    }
    """
    proto_text = proto_text + '\n'.join(data_points_str)

    dm = delay_model.DelayModel(
        text_format.Parse(proto_text, delay_model_pb2.DelayModel()))


if __name__ == '__main__':
  absltest.main()
