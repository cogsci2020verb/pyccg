"""
Defines models for scoring inferred chart parses, and for
updating weights in virtue of success / failure of parses.
"""

from copy import copy

from nltk.tree import Tree
import numpy as np
import torch as T
from torch.nn import functional as F


# TODO feature: support partial parses, so that chart parser can use a Scorer
# to prune

# TODO feature: more explicitly handle log-probs vs scores


class Scorer(object):
  """
  Interface for scoring inferred chart parses (syntactic or
  syntactic+semantic).
  """

  requires_semantics = False
  """
  If `True`, this scorer fails on parses missing semantic representations.
  """

  def __init__(self, lexicon):
    self._lexicon = lexicon

  def __call__(self, parse):
    if isinstance(parse, Tree):
      return self.score(parse)
    elif isinstance(parse, list):
      return self.score_batch(parse)
    else:
      raise ValueError("Don't know how to score parse of type %s: %s" % (type(parse), parse))

  def clone_with_lexicon(self, lexicon):
    clone = copy(self)
    clone.lexicon = lexicon
    return clone

  def score(self, parse):
    raise NotImplementedError()

  def score_batch(self, parses):
    """
    Score a batch of predicted parses.

    Returns `numpy.ndarray` of floats with the same length as `parses`.
    """
    return np.array([self.score(parse) for parse in parses])

  def update_with_scores(self, results, incorrect_results=None):
    """
    Update scorer weights based on the weighted results (blocked into "correct"
    and "incorrect" groups).

    Args:
      results: List of `(score, parse)` tuples, where `score` comes
        from some downstream evaluation
      incorrect_results: same format as `results`, but incorrect!
        Only relevant for downstream evaluations which explicitly separate
        "correct" and "incorrect" parses. If provided, `results` should only
        contain parses which are correct -- i.e., the two sets of parses should
        be disjoint. (This is not checked by the Scorer.)
    """
    ...


class LexiconScorer(Scorer):
  """
  Scores parses based on the weights of their component lexical entries.

  This scorer assigns *log-probabilities* to parses by constructing probability
  distributions over lexical entries:

    p(entry) = p(categ(entry)) p(entry | categ(entry))

  where both the prior and likelihood are normalized distributions computed
  from lexicon weights.
  """

  def score(self, parse):
    categs, categ_priors = self._lexicon.total_category_masses()
    categ_priors = F.softmax(categ_priors)
    categ_to_idx = dict(zip(categs, range(len(categs))))

    _, total_categ_masses = self._lexicon.total_category_masses(exponentiate=True)

    logp = T.zeros(())
    for _, token in parse.pos():
      categ_idx = categ_to_idx[token.categ()]

      prior = categ_priors[categ_idx]
      if prior == 0:
        return -np.inf

      # TODO prefer softmax distribution
      likelihood = T.exp(max(token.weight(), 1e-6)) / total_categ_masses[categ_idx]
      logp += T.log(prior) + T.log(likelihood)

    return logp
