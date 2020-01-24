"""
Defines models for scoring inferred chart parses, and for
updating weights in virtue of success / failure of parses.
"""

from nltk.tree import Tree
import numpy as np


# TODO feature: support partial parses, so that chart parser can use a Scorer
# to prune


class Scorer(object):
  """
  Interface for scoring inferred chart parses (syntactic or
  syntactic+semantic).
  """

  def __call__(self, parse):
    if isinstance(parse, Tree):
      return self.score(parse)
    elif isinstance(parse, list):
      return self.score_batch(parse)
    else:
      raise ValueError("Don't know how to score parse of type %s: %s" % (type(parse), parse))

  def score(self, parse):
    raise NotImplementedError()

  def score_batch(self, parses):
    """
    Score a batch of predicted parses.

    Returns `numpy.ndarray` of floats with the same length as `parses`.
    """
    return np.array([self.score(parse) for parse in parses])


class LexiconScorer(Scorer):
  """
  Scores parses based on the weights of their component lexical entries.

  This scorer assigns *log-probabilities* to parses by constructing probability
  distributions over lexical entries:

    p(entry) = p(categ(entry)) p(entry | categ(entry))

  where both the prior and likelihood are normalized distributions computed
  from lexicon weights.
  """

  def __init__(self, lexicon):
    self._lexicon = lexicon

  def score(self, parse):
    category_priors = self._lexicon.observed_category_distribution()
    total_category_masses = self._lexicon.total_category_masses()

    logp = 0.0
    for _, token in parse.pos():
      prior = category_priors[token.categ()]
      if prior == 0:
        return -np.inf

      likelihood = max(token.weight(), 1e-6) / total_category_masses[token.categ()]
      logp += np.log(prior) + np.log(likelihood)

    return logp
