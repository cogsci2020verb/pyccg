"""
Defines strategies for enumerating over logical expressions within an ontology.
"""

from copy import copy
from collections import namedtuple
from functools import lru_cache
import itertools

from pyccg import logic as l
from pyccg.logic import base as B
from pyccg.logic.util import next_bound_var
from pyccg.util import listify


class IterationContext(object):
  """
  Describes the inferential context in which we are enumerating logical forms for
  some lexical induction.

  Members:
    bound_vars: Bound variables (and their types) in the parent LF context. The
      returned expressions may reference these variables. List of `(name, type)`
      tuples.
    semantic_type: Requested type of the semantic expression. This can help
      greatly restrict the space of enumerations when the type system is strong.
    syntactic_type: Hypothesized CCG syntactic type of the lexical entry for
      which we are enumerating logical forms.
  """

  def __init__(self, bound_vars=(), semantic_type=None, syntactic_type=None):
    self.bound_vars = bound_vars
    self.semantic_type = semantic_type
    self.syntactic_type = syntactic_type

  def __hash__(self):
    return hash((self.bound_vars, self.semantic_type, self.syntactic_type))

  def clone(self):
    return copy(self)

  def clone_with_semantic_type(self, semantic_type):
    ret = copy(self)
    ret.semantic_type = semantic_type
    return ret


class ExpressionIterator(object):
  """
  Yields a sequence over legal logical expressions for an ontology.

  Iterators can serve two distinctive functions:

  1. Ordering: providing some expressions before others. For limited iteration,
     it can be crucial to pick the right ordering.
  2. Filtering: excluding irrelevant expressions based on *a priori*
     constraints. (Iterators which filter are called *unsound* and should
     update the class property `is_sound` accordingly.)

  """

  is_sound = True
  """
  States whether this iterator eventually yields the entire space of possible
  expressions licensed by the ontology.
  """

  def __init__(self, ontology):
    self.ontology = ontology

  def iter_expressions(self, max_depth, context):
    """
    Yield legal expressions within the ontology.

    Arguments:
      max_depth: Maximum tree depth to traverse.
      context: An `IterationContext` describing constraints from / features of
        the inferential context.
    """
    raise NotImplementedError()

  def reset(self):
    """
    Reset the state of this iterator. Called when e.g. the attached ontology is
    updated. Iterators should clear caches and ontology-state-specific
    information.
    """
    pass


class DefaultExpressionIterator(ExpressionIterator):
  """
  Default sound expression iterator, which exhaustively enumerates an
  ontology's LF space with little weighting.
  """

  is_sound = True

  def __init__(self, ontology):
    super().__init__(ontology)
    self.types = self.ontology.types

    # Types of expressions to enumerate
    self.EXPR_TYPES = [l.ApplicationExpression, l.ConstantExpression,
                       l.IndividualVariableExpression, l.FunctionVariableExpression,
                       l.LambdaExpression]

  def reset(self):
    self.iter_expressions.cache_clear()

  @lru_cache(maxsize=None)
  @listify
  def iter_expressions(self, max_depth, context,
                       function_weights=None,
                       use_unused_constants=False,
                       unused_constants_whitelist=None,
                       unused_constants_blacklist=None):
    """
    Enumerate all legal expressions.

    Arguments:
      max_depth: Maximum tree depth to traverse.
      function_weights: Override for function weights to determine the order in
        which we consider proposing function application expressions.
      use_unused_constants: If true, always use unused constants.
      unused_constants_whitelist: If not None, a set of constants (by name),
        all newly used constants for the current expression.
    """
    if max_depth == 0:
      return
    elif max_depth == 1 and not context.bound_vars:
      # require some bound variables to generate a valid lexical entry
      # semantics
      return

    unused_constants_whitelist = frozenset(unused_constants_whitelist or [])
    unused_constants_blacklist = frozenset(unused_constants_blacklist or [])

    for expr_type in self.EXPR_TYPES:
      if expr_type == l.ApplicationExpression:
        # Loop over functions according to their weights.
        fn_weight_key = (lambda fn: function_weights[fn.name]) if function_weights is not None \
                        else (lambda fn: fn.weight)
        fns_sorted = sorted(self.ontology.functions_dict.values(), key=fn_weight_key,
                            reverse=True)

        if max_depth > 1:
          for fn in fns_sorted:
            # If there is a present type request, only consider functions with
            # the correct return type.
            # print("\t" * (6 - max_depth), fn.name, fn.return_type, " // request: ", context.semantic_type, context.bound_vars)
            if context.semantic_type is not None and not fn.return_type.matches(context.semantic_type):
              continue

            # Special case: yield fast event queries without recursion.
            if fn.arity == 1 and fn.arg_types[0] == self.types.EVENT_TYPE:
              yield B.make_application(fn.name, (l.ConstantExpression(l.Variable("e")),))
            elif fn.arity == 0:
              # 0-arity functions are represented in the logic as
              # `ConstantExpression`s.
              # print("\t" * (6 - max_depth + 1), "yielding const ", fn.name)
              yield l.ConstantExpression(l.Variable(fn.name))
            else:
              # print("\t" * (6 - max_depth), fn, fn.arg_types)
              all_arg_semantic_types = list(fn.arg_types)

              def product_sub_args(i, ret, blacklist, whitelist):
                if i >= len(all_arg_semantic_types):
                  yield ret
                  return

                arg_semantic_type = all_arg_semantic_types[i]
                sub_context = context.clone_with_semantic_type(arg_semantic_type)
                results = self.iter_expressions(max_depth=max_depth - 1,
                                                context=context,
                                                function_weights=function_weights,
                                                use_unused_constants=use_unused_constants,
                                                unused_constants_whitelist=frozenset(whitelist),
                                                unused_constants_blacklist=frozenset(blacklist))

                new_blacklist = blacklist
                for expr in results:
                  new_whitelist = whitelist | {c.name for c in expr.constants()}
                  for sub_expr in product_sub_args(i + 1, ret + (expr, ), new_blacklist, new_whitelist):
                    yield sub_expr
                    new_blacklist = new_blacklist | {c.name for arg in sub_expr for c in arg.constants()}

              for arg_combs in product_sub_args(0, tuple(), unused_constants_blacklist, unused_constants_whitelist):
                candidate = B.make_application(fn.name, arg_combs)
                valid = self.ontology._valid_application_expr(candidate)
                # print("\t" * (6 - max_depth + 1), "valid %s? %s" % (candidate, valid))
                if valid:
                  yield candidate
      elif expr_type == l.LambdaExpression and max_depth > 1:
        if context.semantic_type is None or not isinstance(context.semantic_type, l.ComplexType):
          continue

        for num_args in range(1, len(context.semantic_type.flat)):
          for bound_var_types in itertools.product(self.ontology.observed_argument_types, repeat=num_args):
            # TODO typecheck with type request

            bound_vars = list(context.bound_vars)
            subexpr_bound_vars = []
            for new_type in bound_var_types:
              subexpr_bound_vars.append(next_bound_var(bound_vars + subexpr_bound_vars, new_type))
            all_bound_vars = tuple(bound_vars + subexpr_bound_vars)

            if context.semantic_type is not None:
              # TODO strong assumption -- assumes that lambda variables are used first
              subexpr_semantic_type_flat = context.semantic_type.flat[num_args:]
              subexpr_semantic_type = self.types[subexpr_semantic_type_flat]
            else:
              subexpr_semantic_type = None

            # Prepare enumeration context
            sub_context = context.clone()
            sub_context.bound_vars = all_bound_vars
            sub_context.semantic_type = subexpr_semantic_type

            results = self.iter_expressions(max_depth=max_depth - 1,
                                            context=sub_context,
                                            function_weights=function_weights,
                                            use_unused_constants=use_unused_constants,
                                            unused_constants_whitelist=unused_constants_whitelist,
                                            unused_constants_blacklist=unused_constants_blacklist)

            for expr in results:
              candidate = expr
              for var in subexpr_bound_vars:
                candidate = l.LambdaExpression(var, candidate)
              valid = self.ontology._valid_lambda_expr(candidate, bound_vars)
              # print("\t" * (6 - max_depth), "valid lambda %s? %s" % (candidate, valid))
              if valid:
                # Assign variable types before returning.
                extra_types = {bound_var.name: bound_var.type
                               for bound_var in subexpr_bound_vars}

                try:
                  # TODO make sure variable names are unique before this happens
                  self.ontology.typecheck(candidate, extra_types)
                except l.InconsistentTypeHierarchyException:
                  pass
                else:
                  yield candidate
      elif expr_type == l.IndividualVariableExpression:
        for bound_var in context.bound_vars:
          if context.semantic_type and not bound_var.type.matches(context.semantic_type):
            continue

          # print("\t" * (6-max_depth), "var %s" % bound_var)

          yield l.IndividualVariableExpression(bound_var)
      elif expr_type == l.ConstantExpression:
        if use_unused_constants:
          try:
            for constant in self.ontology.constant_system.iter_new_constants(
                semantic_type=context.semantic_type,
                unused_constants_whitelist=unused_constants_whitelist,
                unused_constants_blacklist=unused_constants_blacklist
            ):

              yield l.ConstantExpression(constant)
          except ValueError:
            pass
        else:
          for constant in self.ontology.constants:
            if context.semantic_type is not None and not constant.type.matches(context.semantic_type):
              continue

            yield l.ConstantExpression(constant)
      elif expr_type == l.FunctionVariableExpression:
        # NB we don't support enumerating bound variables with function types
        # right now -- the following only considers yielding fixed functions
        # from the ontology.
        for function in self.ontology.functions:
          # TODO(Jiayuan Mao @ 04/10): check the correctness of the following lie.
          # I currently skip nullary functions since it has been handled as constant variables
          # in L2311.
          if function.arity == 0:
              continue

          # Be a little strict here to avoid excessive enumeration -- only
          # consider emitting functions when the type request specifically
          # demands a function, not e.g. AnyType
          if context.semantic_type is None or context.semantic_type == self.types.ANY_TYPE \
              or not function.type.matches(context.semantic_type):
            continue

          yield l.FunctionVariableExpression(l.Variable(function.name, function.type))
