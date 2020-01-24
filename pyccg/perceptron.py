"""
Structured perceptron algorithm for learning CCG weights.
"""

from collections import Counter
import logging
import functools
import numpy as np

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


def update_perceptron(learner, sentence, model, success_fn,
                      learning_rate=10, parser=None,
                      update_method="perceptron"):
  if parser is None:
    parser = learner.make_parser(ruleset=chart.DefaultRuleSet)

  norm = 0.0
  weighted_results = parser.parse(sentence, return_aux=True)
  if not weighted_results:
    raise NoParsesSyntaxError("No successful parses computed.", sentence)

  max_score, max_incorrect_score = -np.inf, -np.inf
  correct_results, incorrect_results = [], []

  for result, score, _ in weighted_results:
    success, answer_score = success_fn(result, model)
    if success:
      score += answer_score
      if score > max_score:
        max_score = score
        correct_results = [(score, result)]
      elif score == max_score:
        correct_results.append((score, result))
    else:
      if score > max_incorrect_score:
        max_incorrect_score = score
        incorrect_results = [(score, result)]
      elif score == max_incorrect_score:
        incorrect_results.append((score, result))

  if not correct_results:
    raise NoParsesError("No parses derived are successful.", sentence)
  elif not incorrect_results:
    # L.warning("No incorrect parses. Skipping update.")
    return weighted_results, 0.0

  # Sort results by descending parse score.
  correct_results = sorted(correct_results, key=lambda r: r[0], reverse=True)
  incorrect_results = sorted(incorrect_results, key=lambda r: r[0], reverse=True)

  if update_method == "perceptron":
    correct_results = correct_results[:1]
    incorrect_results = incorrect_results[:1]

  # TODO margin?

  # Update to separate max-scoring parse from max-scoring correct parse if
  # necessary.
  positive_mass = 1 / len(correct_results)
  negative_mass = 1 / len(incorrect_results)

  token_deltas = Counter()
  observed_leaf_sequences = set()
  for results, delta in zip([correct_results, incorrect_results],
                             [positive_mass, -negative_mass]):
    parse_results = [r[1] for r in results]
    if update_method == "reinforce":
      parse_scores = delta * softmax(np.array([r[0] for r in results]))
    else:
      parse_scores = np.repeat(delta, len(results))

    for score_delta, result in zip(parse_scores, parse_results):
      leaf_seq = tuple(leaf_token for _, leaf_token in result.pos())
      if leaf_seq not in observed_leaf_sequences:
        observed_leaf_sequences.add(leaf_seq)
        for leaf_token in leaf_seq:
          token_deltas[leaf_token] += score_delta

  for token, delta in token_deltas.items():
    delta *= learning_rate
    norm += delta ** 2

    L.info("Applying delta: %+.03f %s", delta, token)
    token._weight += delta

  return weighted_results, norm


def update_perceptron_with_cached_results(learner, sentence, parses, normalized_scores, answer_scores, learning_rate=10, update_method="perceptron"):
  assert update_method == 'reinforce', 'Only reinforce is implemented for update_perceptron_with_cached_results'

  if not parses:
    # NB(Jiayuan Mao @ 09/19): we need to check the parsing again, since the missing lexicons might have been induced in
    # an instance in the same batch.
    parser = learner.make_parser()
    if len(parser.parse(sentence)) == 0:
      raise NoParsesSyntaxError("No successful parses computed.", "")
    else:
      return [], 0

  token_deltas = Counter()
  for parse, score, answer_score in zip(parses, normalized_scores, answer_scores):
    leaf_seq = tuple(leaf_token for _, leaf_token in parse.pos())
    for leaf_token in leaf_seq:
      token_deltas[leaf_token] += answer_score * score

  norm = 0.0
  for token, delta in token_deltas.items():
    delta *= learning_rate
    norm += delta ** 2
    token._weight += delta

  return parses, norm


def _update_perceptron_distant_success_fn(parse_result, model, answer):
  root_token, _ = parse_result.label()
  L.debug('Semantics: {}; answer: {}; groundtruth: {}.'.format(root_token.semantics(), model.evaluate(root_token.semantics()), answer))

  answer_score = 0.0
  try:
    if hasattr(model, "evaluate_and_score"):
      success, answer_score = model.evaluate_and_score(root_token.semantics(), answer)
    else:
      pred_answer = model.evaluate(root_token.semantics())
      success = pred_answer == answer

  except (TypeError, AttributeError) as e:
    # Type inconsistency. TODO catch this in the iter_expression
    # stage, or typecheck before evaluating.
    success = False
  except AssertionError as e:
    # Precondition of semantics failed to pass.
    success = False

  return success, answer_score


def update_perceptron_nscl(learner, sentence, model, answer,
                              **update_perceptron_kwargs):

  L.debug("Desired answer: %s", answer)
  success_fn = functools.partial(_update_perceptron_distant_success_fn, answer=answer)
  return update_perceptron(learner, sentence, model, success_fn, **update_perceptron_kwargs)


def update_perceptron_nscl_with_cached_results(learner, sentence, model, parses, normalized_scores, answer_scores, **update_perceptron_kwargs):
  # sentence and model will be ignored.
  return update_perceptron_with_cached_results(learner, sentence, parses, normalized_scores, answer_scores, **update_perceptron_kwargs)


def update_perceptron_distant(learner, sentence, model, answer,
                              **update_perceptron_kwargs):

  L.debug("Desired answer: %s", answer)
  success_fn = functools.partial(_update_perceptron_distant_success_fn, answer=answer)
  return update_perceptron(learner, sentence, model, success_fn, **update_perceptron_kwargs)


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
