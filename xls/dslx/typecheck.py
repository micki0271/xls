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

"""Implementation of type checking functionality on a parsed AST object."""

import functools
from typing import Dict, Optional, Text, Tuple, Union, Callable, List, Type

from absl import logging
import dataclasses

from xls.common.xls_error import XlsError
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_deduce
from xls.dslx.python import cpp_dslx_builtins
from xls.dslx.python import cpp_type_info as type_info
from xls.dslx.python import cpp_typecheck
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_deduce import xls_type_error as XlsTypeError
from xls.dslx.python.cpp_type_info import SymbolicBindings
from xls.dslx.span import PositionalError


@dataclasses.dataclass
class _TypecheckStackRecord:
  """A wrapper over information used to typecheck a top level AST node.

  Attributes:
    name: The name of this top-level node.
    kind: The class type (ast.Function, ast.Test, ast.StructDef, ast.TypeDef).
    user: The node in this module that needs 'name' to be typechecked. Used to
      detect the typechecking of the higher order function in map invocations.
  """
  name: Text
  kind: Type[ast.AstNode]
  user: Optional[ast.AstNode] = None


def check_top_node_in_module(f: Union[ast.Function, ast.Test, ast.StructDef,
                                      ast.TypeDef], ctx: cpp_deduce.DeduceCtx):
  """Type-checks function f in the given module.

  Args:
    f: Function/test/struct/typedef to type-check.
    ctx: Wraps a type_info, a mapping being populated with the inferred type
      for AST nodes. Also contains a module.

  Raises:
    TypeMissingError: When we attempt to resolve an AST node to a type that a)
      cannot be resolved via the type_info mapping and b) the AST node
      missing a type does not refer to a top-level function in the module
      (determined via function_map).
    XlsTypeError: When there is a type check failure.
  """
  # {name: (function, wip)}
  seen = {
      (f.name.identifier, type(f)): (f, True)
  }  # type: Dict[Tuple[Text, type], Tuple[Union[ast.Function, ast.Test, ast.StructDef, ast.TypeDef], bool]]

  stack = [_TypecheckStackRecord(f.name.identifier,
                                 type(f))]  # type: List[_TypecheckStackRecord]

  function_map = {f.name.identifier: f for f in ctx.module.get_functions()}
  while stack:
    try:
      f = seen[(stack[-1].name, stack[-1].kind)][0]
      if isinstance(f, ast.Function):
        cpp_typecheck.check_function(f, ctx)
      elif isinstance(f, ast.Test):
        cpp_typecheck.check_test(f, ctx)
      else:
        assert isinstance(f, (ast.StructDef, ast.TypeDef))
        # Nothing special, we just want to be able to catch any
        # TypeMissingErrors and try to resolve them.
        cpp_deduce.deduce(f, ctx)

      seen[(f.name.identifier, type(f))] = (f, False)  # Mark as done.
      stack_record = stack.pop()
      fn_name = ctx.peek_fn_stack().name

      def is_callee_map(n: Optional[ast.AstNode]) -> bool:
        return (n and isinstance(n, ast.Invocation) and
                isinstance(n.callee, ast.NameRef) and
                n.callee.identifier == 'map')

      if is_callee_map(stack_record.user):
        assert isinstance(f, ast.Function) and f.is_parametric()
        # We just typechecked a higher-order parametric function (from map()).
        # Let's go back to our parent type_info mapping.
        ctx.pop_derived_type_info()

      if stack_record.name == fn_name:
        # i.e. we just finished typechecking the body of the function we're
        # currently inside of.

        # NOTE: if this is a local parametric function, we don't revert to our
        # parent type_info until deduce._check_parametric_invocation() to
        # avoid entering an infite loop. See the try-catch in that function for
        # more details.
        ctx.pop_fn_stack_entry()

    except type_info.TypeMissingError as e:
      while True:
        fn_name = ctx.peek_fn_stack().name
        if (isinstance(e.node, ast.NameDef) and
            e.node.identifier in function_map):
          # If it's seen and not-done, we're recursing.
          if seen.get((e.node.identifier, ast.Function), (None, False))[1]:
            raise XlsError(
                'Recursion detected while typechecking; name: {}'.format(
                    e.node.identifier))
          callee = function_map[e.node.identifier]
          assert isinstance(callee, ast.Function), callee
          seen[(e.node.identifier, type(callee))] = (callee, True)
          stack.append(
              _TypecheckStackRecord(callee.name.identifier, type(callee),
                                    e.user))
          break
        if (isinstance(e.node, ast.BuiltinNameDef) and
            e.node.identifier in cpp_dslx_builtins.PARAMETRIC_BUILTIN_NAMES):
          logging.vlog(2, 'node: %r; identifier: %r, exception user: %r',
                       e.node, e.node.identifier, e.user)

          if isinstance(e.user, ast.Invocation):
            func = cpp_typecheck.instantiate_builtin_parametric(
                e.node, e.user, ctx)
            if func:
              # We need to figure out what to do with this higher order
              # parametric function.
              e.node = func
              continue
            break

        # Raise if this wasn't a function in this module or a builtin.
        raise


ImportFn = Callable[[Tuple[Text, ...]], Tuple[ast.Module, type_info.TypeInfo]]


def check_module(module: ast.Module,
                 f_import: Optional[ImportFn]) -> type_info.TypeInfo:
  """Validates type annotations on all functions within "module".

  Args:
    module: The module to type check functions for.
    f_import: Callback to import a module (a la a import statement). This may be
      None e.g. in unit testing situations where it's guaranteed there will be
      no import statements.
  Returns:
    Mapping from AST node to its deduced/checked type.
  Raises:
    XlsTypeError: If any of the function in f have typecheck errors.
  """
  assert f_import is None or callable(f_import), f_import
  ti = type_info.TypeInfo(module)
  import_cache = None if f_import is None else getattr(f_import, 'cache')
  additional_search_paths = () if f_import is None else getattr(
      f_import, 'additional_search_paths')
  ftypecheck = functools.partial(check_module, f_import=f_import)
  ctx = cpp_deduce.DeduceCtx(ti, module, cpp_deduce.deduce,
                             check_top_node_in_module, ftypecheck,
                             additional_search_paths, import_cache)

  # First populate type_info with constants, enums, and resolved imports.
  ctx.add_fn_stack_entry(
      'top', SymbolicBindings())  # No sym bindings in the global scope.
  for member in ctx.module.top:
    if isinstance(member, ast.Import):
      assert isinstance(member.name, tuple), member.name
      imported_module, imported_type_info = f_import(member.name)
      ctx.type_info.add_import(member, (imported_module, imported_type_info))
    elif isinstance(member, (ast.Constant, ast.EnumDef)):
      cpp_deduce.deduce(member, ctx)
    else:
      assert isinstance(member,
                        (ast.Function, ast.Test, ast.StructDef, ast.QuickCheck,
                         ast.TypeDef)), (type(member), member)
  ctx.pop_fn_stack_entry()

  quickcheck_map = {
      qc.f.name.identifier: qc for qc in ctx.module.get_quickchecks()
  }
  for qc in quickcheck_map.values():
    assert isinstance(qc, ast.QuickCheck), qc

    f = qc.f
    assert isinstance(f, ast.Function), f
    if f.is_parametric():
      # TODO(cdleary): 2020-08-09 See https://github.com/google/xls/issues/81
      raise PositionalError(
          'Quickchecking parametric '
          'functions is unsupported.', f.span)

    logging.vlog(2, 'Typechecking function: %s', f)
    ctx.add_fn_stack_entry(f.name.identifier,
                           SymbolicBindings())  # No symbolic bindings.
    check_top_node_in_module(f, ctx)

    quickcheck_f_body_type = ctx.type_info[f.body]
    if quickcheck_f_body_type != ConcreteType.U1:
      raise XlsTypeError(
          f.span,
          quickcheck_f_body_type,
          ConcreteType.U1,
          suffix='QuickCheck functions must return a bool.')

    logging.vlog(2, 'Finished typechecking function: %s', f)

  # We typecheck struct definitions using check_top_node_in_module() so that
  # we can typecheck function calls in parametric bindings, if any.
  struct_map = {s.name.identifier: s for s in ctx.module.get_structs()}
  for s in struct_map.values():
    assert isinstance(s, ast.StructDef), s
    logging.vlog(2, 'Typechecking struct %s', s)
    ctx.add_fn_stack_entry('top', SymbolicBindings())  # No symbolic bindings.
    check_top_node_in_module(s, ctx)
    logging.vlog(2, 'Finished typechecking struct: %s', s)

  typedef_map = {
      t.name.identifier: t
      for t in ctx.module.top
      if isinstance(t, ast.TypeDef)
  }
  for t in typedef_map.values():
    assert isinstance(t, ast.TypeDef), t
    logging.vlog(2, 'Typechecking typedef %s', t)
    ctx.add_fn_stack_entry('top', SymbolicBindings())  # No symbolic bindings.
    check_top_node_in_module(t, ctx)
    logging.vlog(2, 'Finished typechecking typedef: %s', t)

  function_map = {f.name.identifier: f for f in ctx.module.get_functions()}
  for f in function_map.values():
    assert isinstance(f, ast.Function), f
    if f.is_parametric():
      # Let's typecheck parametric functions per invocation.
      continue

    logging.vlog(2, 'Typechecking function: %s', f)
    ctx.add_fn_stack_entry(f.name.identifier,
                           SymbolicBindings())  # No symbolic bindings.
    check_top_node_in_module(f, ctx)
    logging.vlog(2, 'Finished typechecking function: %s', f)

  test_map = {t.name.identifier: t for t in ctx.module.get_tests()}
  for t in test_map.values():
    assert isinstance(t, ast.Test), t

    if isinstance(t, ast.TestFunction):
      # New-style test constructs are specified using a function.
      # This function shouldn't be parametric and shouldn't take any arguments.
      if t.fn.params:
        raise PositionalError("Test functions shouldn't take arguments.",
                              t.fn.span)

      if t.fn.is_parametric():
        raise PositionalError("Test functions shouldn't be parametric.",
                              t.fn.span)

    # No symbolic bindings inside of a test.
    ctx.add_fn_stack_entry('{}_test'.format(t.name.identifier),
                           SymbolicBindings())
    logging.vlog(2, 'Typechecking test: %s', t)
    if isinstance(t, ast.TestFunction):
      # New-style tests are wrapped in a function.
      check_top_node_in_module(t.fn, ctx)
    else:
      # Old-style tests are specified in a construct with a body
      # (see cpp_typecheck.check_test()).
      check_top_node_in_module(t, ctx)
    logging.vlog(2, 'Finished typechecking test: %s', t)

  return ctx.type_info
