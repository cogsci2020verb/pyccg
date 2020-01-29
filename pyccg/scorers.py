"""
Defines models for scoring inferred chart parses, and for
updating weights in virtue of success / failure of parses.
"""

from collections import Counter
from copy import copy
import logging

from nltk.tree import Tree
import numpy as np
import torch as T
from torch import nn
from torch.nn import functional as F

from pyccg import logic as l

L = logging.getLogger(__name__)


# TODO feature: support partial parses, so that chart parser can use a Scorer
# to prune


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
    super().__init__()
    self._lexicon = lexicon

  def __call__(self, parse, sentence_meta=None):
    if isinstance(parse, Tree):
      return self.score(parse, sentence_meta=sentence_meta)
    elif isinstance(parse, list):
      return self.score_batch(parse, sentence_meta=sentence_meta)
    else:
      raise ValueError("Don't know how to score parse of type %s: %s" % (type(parse), parse))

  def __add__(self, scorer):
    if isinstance(scorer, CompositeScorer):
      # We're not a CompositeScorer. Let this scorer absorb us.
      return scorer + self
    return CompositeScorer(scorer, self)

  def clone_with_lexicon(self, lexicon):
    if self.requires_semantics and not lexicon.has_semantics:
      # Check that the lexicon assigns semantic representations
      L.warn("Semantics-sensitive scorer is being cloned with a semantics-free lexicon. "
             "Going to return an empty scorer.")
      return EmptyScorer(lexicon)

    clone = copy(self)
    clone.lexicon = lexicon
    return clone

  def forward(self, parse, sentence_meta=None):
    raise NotImplementedError()

  def score(self, parse, sentence_meta=None):
    return self.forward(parse, sentence_meta=sentence_meta)

  def score_batch(self, parses, sentence_metas=None):
    """
    Score a batch of predicted parses.

    Returns `numpy.ndarray` of floats with the same length as `parses`.
    """
    if sentence_metas is None:
      sentence_metas = [None] * len(parses)
    return np.array([self.score(parse, sentence_meta=sentence_meta)
                     for parse, sentence_meta in zip(parses, sentence_metas)])


class EmptyScorer(Scorer):
  def forward(self, parse):
    return T.zeros(())


class CompositeScorer(Scorer):
  """
  Scorer which composes multiple independent scorers.
  """

  def __init__(self, *scorers):
    self.scorers = scorers

  def __add__(self, scorer):
    self.scorers.append(scorer)

  def clone_with_lexicon(self, lexicon):
    scorers = [scorer.clone_with_lexicon(lexicon) for scorer in self.scorers]
    scorers = [scorer for scorer in scorers if not isinstance(scorer, EmptyScorer)]

    if len(scorers) == 0:
      return EmptyScorer(lexicon)
    return CompositeScorer(*scorers)

  def parameters(self):
    ret = []
    for scorer in self.scorers:
      ret.extend(scorer.parameters())
    return ret

  def forward(self, parse, sentence_meta=None):
    return T.zeros(()) + sum(scorer(parse, sentence_meta=sentence_meta) for scorer in self.scorers)


class LexiconScorer(Scorer):
  """
  Scores parses based on the weights of their component lexical entries.

  This scorer assigns *log-probabilities* to parses by constructing probability
  distributions over lexical entries:

    p(entry) = p(categ(entry)) p(entry | categ(entry))

  where both the prior and likelihood are normalized distributions computed
  from lexicon weights.
  """

  def parameters(self):
    return [e.weight() for e in self._lexicon.all_entries]

  def forward(self, parse, sentence_meta=None):
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


class RootSemanticLengthScorer(Scorer):
  """
  Exponentially penalizes (or rewards) parses based on the length of the
  semantic form of their root verb.
  """

  requires_semantics = True

  def __init__(self, lexicon, parameter=0.9, max_length=20, inverse=False,
               root_types=(r"(S\N)", r"((S\N)/N)")):
    """
    Args:
      lexicon:
      max_length: Maximum expected length of a semantic expression. Only
        required if `inverse` is `True`, in order to yield a proper probability
        distribution.
    """
    super().__init__(lexicon)

    if inverse:
      length_weights = [np.power(parameter, max_length - length)
                        for length in range(max_length)]
    else:
      length_weights = [np.power(parameter, length) for length in range(max_length)]
    length_weights = np.array(length_weights)
    length_weights /= length_weights.sum()

    self.length_weights = T.tensor(np.log(length_weights))

    self.root_types = root_types

  def parameters(self): return []

  def forward(self, parse, sentence_meta=None):
    score = T.zeros(())
    try:
      root_verb = next(tok for _, tok in parse.pos()
                       if str(tok.categ()) in self.root_types)
    except:
      return score

    n_predicates = len(root_verb.semantics().predicates_list())
    if n_predicates < len(self.length_weights):
      return self.length_weights[n_predicates]

    # TODO this is probably bad default behavior :)
    return T.tensor(-np.inf)


class FrameSemanticsScorer(Scorer):

  requires_semantics = True

  def __init__(self, lexicon, frames, root_types=(r"(S\N)", r"((S\N)/N)")):
    """
    Args:
      lexicon:
      frames: Collection of all possible frame strings
      root_types: CCG syntactic type strings of lexical entries for which we
        are collecting frames
    """
    super().__init__(lexicon)

    self.frames = frames
    self.frame_to_idx = {frame: T.tensor(idx, requires_grad=False)
                         for idx, frame in enumerate(sorted(self.frames))}

    self.root_types = set(root_types)

    ontology = self._lexicon.ontology
    self.predicates = [l.Variable(val.name) for val in ontology.functions + ontology.constants]
    self.predicate_to_idx = {pred: idx for idx, pred in enumerate(sorted(self.predicates))}

    # Represent unnormalized frame distributions as an embedding layer
    self.frame_dist = nn.Embedding(len(self.frames), len(self.predicates))
    nn.init.zeros_(self.frame_dist.weight)

  def parameters(self):
    return self.frame_dist.parameters()

  # TODO override clone_with_lexicon

  def forward(self, parse, sentence_meta=None):
    if sentence_meta is None or sentence_meta.get("frame_str", None) is None:
      raise ValueError("FrameSemanticsScorer requires a sentence_meta key frame_str")

    frame = sentence_meta["frame_str"]
    try:
      frame_idx = self.frame_to_idx[frame]
    except KeyError:
      raise ValueError("Unknown frame string %s" % sentence)

    ret = self.frame_dist(self.frame_to_idx[frame])
    predicate_logps = F.log_softmax(ret)

    score = T.zeros(())
    try:
        root_verb = next(tok for _, tok in parse.pos()
                         if str(tok.categ()) in self.root_types)
    except:
        root_verb = None
        return score

    for predicate in root_verb.semantics().predicates():
      score += predicate_logps[self.predicate_to_idx[predicate]]

    return score
