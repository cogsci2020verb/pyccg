"""
Structured perceptron algorithm for learning CCG weights.
"""

from collections import Counter
import logging
import functools

import numpy as np
import torch as T

from pyccg import chart
from pyccg.util import softmax, NoParsesError, NoParsesSyntaxError

L = logging.getLogger(__name__)


def update_perceptron_batch(learner, data, learning_rate=0.1, parser=None):
  """
  Execute a batch perceptron weight update with the given training data.

  Args:
    learner: WordLearner with weights
    data: List of `(x, y)` tuples, where `x` is a list of string
      tokens and `y` is an LF string.
    learning_rate:

  Returns:
    l2 norm of total weight updates
  """

  if parser is None:
    parser = learner.make_parser()

  norm = 0.0
  for x, y in data:
    weighted_results = parser.parse(x, return_aux=True)

    max_result, max_score, _ = weighted_results[0]
    correct_result, correct_score = None, None

    for result, score, _ in weighted_results:
      root_token, _ = result.label()
      if str(root_token.semantics()) == y:
        correct_result, correct_score = result, score
        break
    else:
      raise NoParsesError("no valid parse derived", x)

    if correct_score < max_score:
      for result, sign in zip([correct_result, max_result], [1, -1]):
        for _, leaf_token in result.pos():
          delta = sign * learning_rate
          norm += delta ** 2
          leaf_token._weight += delta

  return norm


def update_reinforce(learner, sentence, model, success_fn,
                     learning_rate=10, parser=None):
  if parser is None:
    parser = learner.make_parser(ruleset=chart.DefaultRuleSet)

  norm = 0.0
  weighted_results = parser.parse(sentence, return_aux=True)
  if not weighted_results:
    raise NoParsesSyntaxError("No successful parses computed.", sentence)

  rewards = [success_fn(result, model)[1] for result, _, _ in weighted_results]
  parses = [parse for parse, _, _ in weighted_results]
  logps = [logp for _, logp, _ in weighted_results]
  update_reinforce_with_cached_results(learner, sentence, parses, logps, rewards)

  return weighted_results


def update_reinforce_with_cached_results(learner, sentence, parses,
                                         logps, rewards):
  if not parses:
    # NB(Jiayuan Mao @ 09/19): we need to check the parsing again, since the
    # missing lexicons might have been induced in an instance in the same
    # batch.
    parser = learner.make_parser(ruleset=chart.DefaultRuleSet)
    if len(parser.parse(sentence)) == 0:
      raise NoParsesSyntaxError("No successful parses computed.", "")
    else:
      return []

  # Set up REINFORCE loss.
  loss = [reward * -logp for reward, logp in zip(rewards, logps)]
  loss = T.stack(loss).sum()
  loss.backward()

  return parses


def _update_distant_success_fn(parse_result, model, answer):
  root_token, _ = parse_result.label()
  L.debug('Semantics: {}; answer: {}; groundtruth: {}.'.format(root_token.semantics(), model.evaluate(root_token.semantics()), answer))

  try:
    if hasattr(model, "evaluate_and_score"):
      success, answer_score = model.evaluate_and_score(root_token.semantics(), answer)
    else:
      pred_answer = model.evaluate(root_token.semantics())
      success = pred_answer == answer
      answer_score = float(success)

  except (TypeError, AttributeError) as e:
    # Type inconsistency. TODO catch this in the iter_expression
    # stage, or typecheck before evaluating.
    success = False
    answer_score = 0.0
  except AssertionError as e:
    # Precondition of semantics failed to pass.
    success = False
    answer_score = 0.0

  return success, answer_score


def update_nscl(learner, sentence, model, answer,
                              **update_perceptron_kwargs):

  L.debug("Desired answer: %s", answer)
  success_fn = functools.partial(_update_distant_success_fn, answer=answer)
  return update_reinforce(learner, sentence, model, success_fn, **update_perceptron_kwargs)


def update_nscl_with_cached_results(learner, sentence, model, parses,
    normalized_scores, answer_scores, **update_perceptron_kwargs):
  # sentence and model will be ignored.
  return update_with_cached_results(learner, sentence, parses,
      normalized_scores, answer_scores, **update_perceptron_kwargs)


def update_distant(learner, sentence, model, answer,
                   **update_kwargs):

  L.debug("Desired answer: %s", answer)
  success_fn = functools.partial(_update_distant_success_fn, answer=answer)
  return update_reinforce(learner, sentence, model, success_fn, **update_kwargs)


def update_perceptron_cross_situational(learner, sentence, model,
                                        **update_perceptron_kwargs):
  def success_fn(parse_result, model):
    root_token, _ = parse_result.label()
    sentence_semantics = root_token.semantics()

    try:
      success = model.evaluate(sentence_semantics) == True
    except:
      success = False

    return success, 0.0

  return update_perceptron(learner, sentence, model, success_fn,
                           **update_perceptron_kwargs)


def update_perceptron_2afc(learner, sentence, models,
                           **update_perceptron_kwargs):
  def success_fn(parse_result, models):
    model1, model2 = models
    root_token, _ = parse_result.label()
    sentence_semantics = root_token.semantics()

    try:
      model1_success = model1.evaluate(sentence_semantics) == True
    except:
      model1_success = False
    try:
      model2_success = model2.evaluate(sentence_semantics) == True
    except:
      model2_success = False

    return (model1_success or model2_success), 0.0

  return update_perceptron(learner, sentence, models, success_fn,
                           **update_perceptron_kwargs)
