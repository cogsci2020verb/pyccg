"""
Defines models for scoring inferred chart parses, and for
updating weights in virtue of success / failure of parses.
"""

from collections import Counter
from copy import copy

from nltk.tree import Tree
import numpy as np
import torch as T
from torch import nn
from torch.nn import functional as F


# TODO feature: support partial parses, so that chart parser can use a Scorer
# to prune

# TODO feature: more explicitly handle log-probs vs scores


class Scorer(nn.Module):
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

  def forward(self, parse):
    raise NotImplementedError()

  def score(self, parse):
    return self.forward(parse)

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

  def __init__(self, lexicon, update_method="perceptron"):
    # TODO ensure that lexicon weights are tensors

    super().__init__(lexicon)
    self.update_method = update_method

  def forward(self, parse):
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


class FrameSemanticsScorer(Scorer):

  def __init__(self, lexicon, all_frames):
    super().__init__(lexicon)

    self.all_frames = all_frames

    self.all_predicates = [] # TODO
    self.predicate_to_idx = {pred: idx for idx, pred in enumerate(self.all_predicates)}

    # Represent unnormalized frame distributions as an embedding layer
    self.frame_dist = nn.Embedding(len(all_predicates), len(self.all_frames))
    nn.init.zeros_(self.frame_dist.weight)

  def forward(self, parse):
    # TODO pass along frame somehow.
    # TODO look up frame index.
    ret = self.frame_dist(frame_index)
    predicate_logps = F.log_softmax(ret)

    # NB this is a hacky way to get the root verb -- might break.
    root_verb = next(tok for tok in parse.pos()
                     if str(tok.categ()) in ("(S\N)", "((S\N)/N)"))

    score = T.zeros(())
    for predicate in root_verb.semantics().predicates():
      predicate_idx = 0 # TODO
      score += predicate_logps[self.predicate_to_idx[predicate]]

    return score
