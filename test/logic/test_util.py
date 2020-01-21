from nose.tools import *

from pyccg.logic import *
from pyccg.logic.util import *


def test_read_ec_sexpr():
  expr, bound_vars = read_ec_sexpr("(lambda (lambda (lambda (foo (bar $0 $1) (baz $1 $2) blah))))")
  eq_(expr, Expression.fromstring(r"\a b c.foo(bar(c,b),baz(b,a),blah)"))
  eq_(len(bound_vars), 3)


def test_read_ec_sexpr_de_bruijn():
  """
  properly handle de Bruijn indexing in EC lambda expressions.
  """
  expr, bound_vars = read_ec_sexpr("(lambda ((lambda ($0 (lambda $0))) (lambda ($1 $0))))")
  print(expr)
  eq_(expr, Expression.fromstring(r"\A.((\B.B(\C.C))(\C.A(C)))"))


def test_read_ec_sexpr_nested():
  """
  B.read_ec_sexpr should support reading in applications where the function
  itself is an expression (i.e. there is some not-yet-reduced beta reduction
  candidate).
  """
  expr, bound_vars = read_ec_sexpr("(lambda ((lambda (foo $0)) $0))")
  eq_(expr, Expression.fromstring(r"\a.((\b.foo(b))(a))"))


def test_read_ec_sexpr_higher_order_param():
  expr, bound_vars = read_ec_sexpr("(lambda (lambda ($0 $1)))")
  eq_(expr, Expression.fromstring(r"\a P.P(a)"))

