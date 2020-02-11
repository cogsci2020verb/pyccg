"""
Defines models for scoring inferred chart parses, and for
updating weights in virtue of success / failure of parses.
"""

from collections import Counter
from copy import copy
import logging

import itertools
import functools
import operator

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
    clone._lexicon = lexicon
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
  def forward(self, parse, sentence_meta=None):
    return T.zeros(())


class CompositeScorer(Scorer):
  """
  Scorer which composes multiple independent scorers.
  """

  def __init__(self, *scorers):
    self.scorers = list(scorers)

  def __add__(self, scorer):
    self.scorers.append(scorer)
    return self

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

  eps = T.tensor(1e-6)

  def parameters(self):
    return [e.weight() for e in self._lexicon.all_entries]

  def forward(self, parse, sentence_meta=None):
    categs, categ_priors = self._lexicon.total_category_masses()
    categ_priors = F.softmax(categ_priors)
    categ_to_idx = dict(zip(categs, range(len(categs))))

    total_categ_masses = dict(zip(*self._lexicon.total_category_masses(exponentiate=True)))

    logp = T.zeros(())
    for _, token in parse.pos():
      categ_idx = categ_to_idx[token.categ()]

      # Uniform prior
      prior = T.tensor(1.)

      likelihood = T.exp(max(token.weight(), self.eps)) / total_categ_masses[token.categ()]
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

  def __init__(self, lexicon, frames,
               root_types=(r"(S\N)", r"((S\N)/N)"),
               frame_weights=None,
               predicates=None,
               predicate_mode=None):
    """
    Args:
      lexicon:
      frames: Collection of all possible frame strings
      root_types: CCG syntactic type strings of lexical entries for which we
        are collecting frames64G
      frame_weights: Optional pre-trained `frames * predicates` matrix. The
        `i`th row of this matrix corresponds to the predicate weights for the
        `i`th entry in `frames`.
      predicates: Required iff `frame_weights` is provided. The `j`th entry in
        this list corresponds to the `j`th column of `frame_weights`.
      predicate_mode:  Must be ['unigrams', 'bigrams']. Whether predicate 
      rows are a distribution over unigrams or 'bigrams' (conjuncts containing an argument filter and a constant) in the ontology.
    """
    super().__init__(lexicon)

    self.frames = frames

    # Sort the list of frames when preparing row indices.
    # BUT if pretrained frame_weights was provided, make sure we preserve the
    # correct frame--index mappings.
    frame_iterator = enumerate(self.frames) if frame_weights is not None else enumerate(sorted(self.frames))
    self.frame_to_idx = {frame: T.tensor(idx, requires_grad=False)
                         for idx, frame in frame_iterator}

    self.root_types = set(root_types)
    self.gradients_disabled = False
    self.predicate_mode = predicate_mode

    ontology = self._lexicon.ontology

    if frame_weights is not None:
      if predicates is None:
        raise ValueError("If pretrained frame weights are provided, you must "
                         "also provide ordered an `predicates` list.")

      self.predicates, self.predicate_to_idx = [], {}
      
      if predicate_mode == 'unigram':
          all_keys = set(ontology.constants_dict.keys()) | set(ontology.functions_dict.keys())
      elif predicate_mode == 'bigram':
          all_conjunct_filters = set(ontology.functions_dict.keys())
          all_conjunct_constants = set(ontology.constants_dict.keys())
          all_keys = set(itertools.product(all_conjunct_filters, all_conjunct_constants))
          
          self.conjunct_filters, self.conjunct_constants = set(), set()
      else:
          raise ValueError("Unsupported predicate mode {}. Must be 'unigram' or 'bigram'".format(predicate_mode))
      
      for idx, predicate in enumerate(predicates):
        if predicate not in all_keys:
          raise ValueError("Unknown pre-trained predicate n-gram `%s` not available in this ontology" % str(predicate))
        if predicate_mode == 'unigram':
            pred = l.Variable(predicate)
        elif predicate_mode == 'bigram':
            if predicate not in all_keys:
                raise ValueError("Unknown pre-trained predicate n-gram `%s` not available in this ontology" % str(predicate))
            
            conjunct_filter, conjunct_constant = predicate
            pred = (l.Variable(conjunct_filter), l.Variable(conjunct_constant))
            self.conjunct_filters.add(conjunct_filter)
            self.conjunct_constants.add(conjunct_constant)
        
        self.predicates.append(pred)
        self.predicate_to_idx[pred] = idx
    else:
      if predicate_mode == 'unigram': 
          self.predicates = [l.Variable(val.name) for val in ontology.functions + ontology.constants]
          self.predicate_to_idx = {pred: idx for idx, pred in enumerate(sorted(self.predicates))}
      elif predicate_mode == 'bigram':
          self.conjunct_filters = set(ontology.functions_dict.keys())
          self.conjunct_constants = set(ontology.constants_dict.keys())
          all_keys = sorted(list(set(itertools.product(self.conjunct_filters, self.conjunct_constants))))

          self.predicates = [(l.Variable(filter), l.Variable(constant)) for (filter, constant) in all_keys]
          self.predicate_to_idx = {pred: idx for idx, pred in enumerate(sorted(self.predicates))}

    # Represent unnormalized frame distributions as an embedding layer
    weights_shape = (len(self.frames), len(self.predicates))
    if frame_weights is not None:
      assert frame_weights.shape == weights_shape, \
          "pretrained %s != expected %s" % (frame_weights.shape, weights_shape)
      self.frame_dist = nn.Embedding.from_pretrained(T.tensor(frame_weights))
    else:
      self.frame_dist = nn.Embedding(*weights_shape)
      nn.init.zeros_(self.frame_dist.weight)

  def parameters(self):
    # HACK: some de-pickled scorers are missing the gradients_disabled entry ..
    if hasattr(self, "gradients_disabled") and self.gradients_disabled:
      return []
    return self.frame_dist.parameters()

  # TODO override clone_with_lexicon

  def disable_gradients(self):
    self.gradients_disabled = True
    self.frame_dist.weight.requires_grad = False

  def get_predicate_weights(self, frames):
      """Gets predicates weights for provided frames"""
      frames = sorted(frames)
      f_indices = T.stack([self.frame_to_idx[f] for f in frames])

      frame_weights = self.frame_dist.weight.index_select(0, f_indices).detach()
      frame_weights = F.softmax(frame_weights, dim=1)

      return frames, frame_weights.numpy()

  def forward(self, parse, sentence_meta=None):
    if sentence_meta is None or sentence_meta.get("frame_str", None) is None:
      raise ValueError("FrameSemanticsScorer requires a sentence_meta key frame_str")

    frame = sentence_meta["frame_str"]
    try:
      frame_idx = self.frame_to_idx[frame]
    except KeyError:
      # Uniform weights over predicates.
      predicate_weights = T.ones((len(self.predicate_to_idx),))
    else:
      predicate_weights = self.frame_dist(self.frame_to_idx[frame])

    predicate_logps = F.log_softmax(predicate_weights)

    score = T.zeros(())
    try:
        root_verb = next(tok for _, tok in parse.pos()
                         if str(tok.categ()) in self.root_types)
    except:
        root_verb = None
        return score

    # DEV: skip predicates.
    # for predicate in root_verb.semantics().predicates():
    #   score += predicate_logps[self.predicate_to_idx[predicate]]
 
    if self.predicate_mode == 'unigram':
        for constant in root_verb.semantics().constants():
          constant = l.Variable(constant.name)
          if constant in self.predicate_to_idx:
            score += predicate_logps[self.predicate_to_idx[constant]]

    elif self.predicate_mode == 'bigram':
        try:
            constants = root_verb.semantics().constants_list()
            constants = [l.Variable(c.name) for c in constants]
            constants.reverse()
            # DEV: only get conjuncts: predicates with associated constants.
            predicates = root_verb.semantics().predicates_list()[-len(constants):]
            bigrams = list(zip(predicates, constants))
            for bigram in bigrams:
                if bigram in self.predicate_to_idx:
                    score += predicate_logps[self.predicate_to_idx[bigram]]

    return score
