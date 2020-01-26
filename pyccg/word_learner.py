from copy import copy
import itertools
import logging

import numpy as np
from torch import optim

from pyccg import chart
from pyccg.lexicon import predict_zero_shot, \
    get_candidate_categories, get_semantic_arity, \
    augment_lexicon_nscl, augment_lexicon_distant, augment_lexicon_cross_situational, augment_lexicon_2afc, \
    build_bootstrap_likelihood
from pyccg.perceptron import \
    update_nscl, update_nscl_with_cached_results, \
    update_distant, update_perceptron_cross_situational, update_perceptron_2afc
from pyccg.scorers import LexiconScorer
from pyccg.util import Distribution, NoParsesError, NoParsesSyntaxError


L = logging.getLogger(__name__)


class WordLearner(object):

  def __init__(self, lexicon, bootstrap=False,
               scorer=None,
               learning_rate=10.0,
               beta=3.0,
               syntax_prior_smooth=1e-3,
               meaning_prior_smooth=1e-3,
               bootstrap_alpha=0.25,
               prune_entries=None,
               zero_shot_limit=5,
               limit_induction=False):
    """
    Args:
      lexicon:
      bootstrap: If `True`, enable syntactic bootstrapping.
    """
    self.lexicon = lexicon

    self.bootstrap = bootstrap

    if scorer is None:
      scorer = LexiconScorer(self.lexicon)
    self.scorer = scorer

    self.optimizer = optim.SGD(lexicon.parameters(), lr=learning_rate)

    # Learning hyperparameters
    self.learning_rate = learning_rate
    self.beta = beta
    self.syntax_prior_smooth = syntax_prior_smooth
    self.meaning_prior_smooth = meaning_prior_smooth
    self.bootstrap_alpha = bootstrap_alpha
    self.prune_entries = prune_entries
    self.zero_shot_limit = zero_shot_limit
    self.limit_induction = limit_induction

  @property
  def ontology(self):
    return self.lexicon.ontology

  def make_parser(self, lexicon=None, ruleset=chart.DefaultRuleSet):
    """
    Construct a CCG parser from the current learner state.
    """
    if lexicon is None:
      lexicon = self.lexicon
    return chart.WeightedCCGChartParser(lexicon=lexicon,
                                        scorer=self.scorer,
                                        ruleset=ruleset)

  def prepare_lexical_induction(self, sentence):
    """
    Find the tokens in a sentence which need to be updated such that the
    sentence will parse.

    Args:
      sentence: Sequence of tokens

    Returns:
      query_tokens: List of tokens which need to be updated
      query_token_syntaxes: Dict mapping tokens to weighted list of candidate
        syntaxes (as returned by `get_candidate_categoies`)
    """
    query_tokens = [word for word in sentence
                    if not self.lexicon.get_entries(word)]
    if len(query_tokens) > 0:
      # Missing lexical entries -- induce entries for words which clearly
      # require an entry inserted
      L.info("Novel words: %s", " ".join(query_tokens))
      query_token_syntaxes = get_candidate_categories(
          self.lexicon, self.scorer, query_tokens, sentence,
          smooth=self.syntax_prior_smooth)

      return query_tokens, query_token_syntaxes

    L.info("No novel words; searching for new entries for known wordforms.")
    # Lexical entries are present for all words, but parse still failed.
    # That means we are missing entries for one or more wordforms.
    # For now: blindly try updating each word's entries.
    #
    # TODO: Does not handle case where multiple words need a joint update.
    query_tokens, query_token_syntaxes = list(set(sentence)), {}
    for token in sentence:
      query_token_syntaxes.update(
          get_candidate_categories(self.lexicon, [token], sentence,
                                   smooth=self.syntax_prior_smooth))

    # Sort query token list by increasing maximum weight of existing lexical
    # entry. This is a little hack to help the learner prefer to try to infer
    # new meanings for words it is currently more uncertain about.
    query_tokens = sorted(query_tokens,
        key=lambda tok: max(self.lexicon.get_entries(tok),
                            key=lambda entry: entry.weight()).weight())
    return query_tokens, query_token_syntaxes

    raise ValueError(
        "unable to find new entries which will make the sentence parse: %s" % sentence)

  def do_lexical_induction(self, sentence, model, augment_lexicon_fn,
                           **augment_lexicon_args):
    """
    Perform necessary lexical induction such that `sentence` can be parsed
    under `model`.

    Returns:
      aug_lexicon: augmented lexicon, a modified copy of `self.lexicon`
    """
    if "queue_limit" not in augment_lexicon_args:
      augment_lexicon_args["queue_limit"] = self.zero_shot_limit

    # Find tokens for which we need to insert lexical entries.
    query_tokens, query_token_syntaxes = \
        self.prepare_lexical_induction(sentence)
    L.info("Inducing new lexical entries for words: %s", ", ".join(query_tokens))

    # Augment the lexicon with all entries for novel words which yield the
    # correct answer to the sentence under some parse. Restrict the search by
    # the supported syntaxes for the novel words (`query_token_syntaxes`).
    #
    # HACK: For now, only induce one word meaning at a time.
    try:
      lex = augment_lexicon_fn(self.lexicon, query_tokens, query_token_syntaxes,
                              sentence, self.ontology, model,
                              self._build_likelihood_fns(sentence, model),
                              beta=self.beta,
                              **augment_lexicon_args)
    except NoParsesError:
      # TODO(Jiayuan Mao @ 04/10): suppress the warnings for now.
      # L.warning("Failed to induce any novel meanings.")
      return self.lexicon

    return lex

  def _build_likelihood_fns(self, sentence, model):
    ret = []
    if self.bootstrap:
      ret.append(build_bootstrap_likelihood(
        self.lexicon, sentence, self.ontology,
        alpha=self.bootstrap_alpha,
        meaning_prior_smooth=self.meaning_prior_smooth))

    return ret

  def predict_zero_shot_tokens(self, sentence, model):
    """
    Yield zero-shot predictions on the syntax and meaning of words in the
    sentence requiring novel lexical entries.

    Args:
      sentence: List of token strings

    Returns:
      syntaxes: Dict mapping tokens to posterior distributions over syntactic
        categories
      joint_candidates: Dict mapping tokens to posterior distributions over
        tuples `(syntax, lf)`
    """
    # Find tokens for which we need to insert lexical entries.
    query_tokens, query_token_syntaxes = self.prepare_lexical_induction(sentence)
    candidates, _ = predict_zero_shot(
        self.lexicon, query_tokens, query_token_syntaxes, sentence,
        self.ontology, model, self._build_likelihood_fns(sentence, model))
    return query_token_syntaxes, candidates

  def predict_zero_shot_2afc(self, sentence, model1, model2):
    """
    Yield zero-shot predictions on a 2AFC sentence, marginalizing over possible
    novel lexical entries required to parse the sentence.

    TODO explain marginalization process in more detail

    Args:
      sentence: List of token strings
      models:

    Returns:
      model_scores: `Distribution` over scene models (with support `models`),
        `p(referred scene | sentence)`
    """
    parser = self.make_parser()
    weighted_results = parser.parse(sentence, True)
    if len(weighted_results) == 0:
      L.warning("Parse failed for sentence '%s'", " ".join(sentence))

      aug_lexicon = self.do_lexical_induction(sentence, (model1, model2),
                                              augment_lexicon_fn=augment_lexicon_2afc,
                                              queue_limit=50)
      parser = self.make_parser(lexicon=aug_lexicon)
      weighted_results = parser.parse(sentence, True)

    dist = Distribution()

    for result, score, _ in weighted_results:
      semantics = result.label()[0].semantics()
      try:
        model1_pass = model1.evaluate(semantics) == True
      except: pass
      else:
        if model1_pass:
          dist[model1] += np.exp(score)

      try:
        model2_pass = model2.evaluate(semantics) == True
      except: pass
      else:
        if model2_pass:
          dist[model2] += np.exp(score)

    return dist.ensure_support((model1, model2)).normalize()

  def _update_with_example(self, sentence, model,
                           augment_lexicon_fn, update_fn,
                           augment_lexicon_args=None,
                           update_args=None):
    """
    Observe a new `sentence -> answer` pair in the context of some `model` and
    update learner weights.

    Args:
      sentence: List of token strings
      model: `Model` instance
      answer: Desired result from `model.evaluate(lf_result(sentence))`

    Returns:
      weighted_results: List of weighted parse results for the example.
    """

    augment_lexicon_args = augment_lexicon_args or {}

    self.optimizer.zero_grad()

    try:
      weighted_results = update_fn(
          self, sentence, model,
          learning_rate=self.learning_rate,
          **update_args)
    except NoParsesSyntaxError as e:
      # No parse succeeded -- attempt lexical induction.
      # TODO(Jiayuan Mao @ 04/10): suppress the warnings for now.
      L.warning("Parse failed for sentence '%s'", " ".join(sentence))
      # L.warning(e)

      # Track optimizer parameter set before lexical induction.
      old_params = set([id(param) for param in self.optimizer.param_groups[0]])

      self.lexicon = self.do_lexical_induction(sentence, model, augment_lexicon_fn,
                                               **augment_lexicon_args)

      # Add new lexicon parameters to optimizer.
      # TODO can't get this to work -- for now we just reinitialize. Fine for stateless SGD.
      # new_params_map = {id(param): param for param in self.lexicon.parameters()}
      # new_params = set(new_params_map.keys()) - set(old_params)
      # self.optimizer.add_param_group({"params": [new_params_map[p_id] for p_id in new_params]})
      self.optimizer = optim.SGD(self.lexicon.parameters(),
                                 lr=self.learning_rate)

      # TODO(Jiayuan Mao @ 04/10): suppress the printing for now.
      # self.lexicon.debug_print()

      # Attempt a new parameter update.
      try:
        weighted_results = update_fn(
            self, sentence, model,
            learning_rate=self.learning_rate,
            **update_args)
      except NoParsesError as e:
        # TODO attempt lexical induction?
        return []
    except NoParsesError as e:
      # TODO(hans): should attempt lexical induction here rather than quitting.
      # the above handler only catches syntax failures, where we have no valid
      # syntactic entry for words in the sentence
      L.warning("Syntactic parse worked, but semantic parse failed for sentence '%s'",
                " ".join(sentence))

      # TODO(Jiayuan Mao @ 04/10): add handler for NoParsesError, meaning that there is no parses being able to
      # answer the question correctly.
      return []

    self.optimizer.step()

    if self.prune_entries is not None:
      prune_count = self.lexicon.prune(max_entries=self.prune_entries)
      L.info("Pruned %i entries from lexicon.", prune_count)

    return weighted_results

  def update_with_nscl(self, sentence, model, answer, augment_lexicon_args=None, update_args=None):
    """
    Observe a new `sentence -> answer` pair in the context of some `model` and
    update learner weights. This function assumes that the `model` is jointly
    optimized with the lexicon set.

    Args:
      sentence: List of token strings
      model: `Model` instance
      answer: Desired result from `model.evaluate(lf_result(sentence))`

    Returns:
      weighted_results: List of weighted parse results for the example.
    """
    augment_lexicon_args = augment_lexicon_args or {}
    update_args = update_args or {}

    kwargs = {"answer": answer}
    augment_lexicon_args.update(kwargs)
    update_args.update(kwargs)

    return self._update_with_example(
        sentence, model,
        augment_lexicon_fn=augment_lexicon_nscl,
        update_fn=update_perceptron_nscl,
        augment_lexicon_args=augment_lexicon_args,
        update_args=update_args)

  def update_with_nscl_cached_results(self, sentence, model, answer, parses,
                                      normalized_scores, answer_scores,
                                      augment_lexicon_args=None, update_args=None):
    augment_lexicon_args = augment_lexicon_args or {}
    update_args = update_args or {}

    kwargs = {"answer": answer}
    augment_lexicon_args.update(kwargs)
    kwargs = {"parses": parses, "normalized_scores": normalized_scores, "answer_scores": answer_scores}
    update_args.update(kwargs)

    return self._update_with_example(
        sentence, model,
        augment_lexicon_fn=augment_lexicon_nscl,
        update_fn=update_perceptron_nscl_with_cached_results,
        augment_lexicon_args=augment_lexicon_args,
        update_args=update_args
    )


  def update_with_distant(self, sentence, model, answer, augment_lexicon_args=None, update_args=None):
    """
    Observe a new `sentence -> answer` pair in the context of some `model` and
    update learner weights.

    Args:
      sentence: List of token strings
      model: `Model` instance
      answer: Desired result from `model.evaluate(lf_result(sentence))`

    Returns:
      weighted_results: List of weighted parse results for the example.
    """
    augment_lexicon_args = augment_lexicon_args or {}
    update_args = update_args or {}

    kwargs = {"answer": answer}
    augment_lexicon_args.update(kwargs)
    update_args.update(kwargs)

    return self._update_with_example(
        sentence, model,
        augment_lexicon_fn=augment_lexicon_distant,
        update_fn=update_distant,
        augment_lexicon_args=augment_lexicon_args,
        update_args=update_args)

  def update_with_cross_situational(self, sentence, model):
    """
    Observe a new `sentence` in the context of a scene reference `model`.
    Assume that `sentence` is true of `model`, and use it to update learner
    weights.

    Args:
      sentence: List of token strings
      model: `Model` instance

    Returns:
      weighted_results: List of weighted parse results for the example.
    """
    return self._update_with_example(
        sentence, model,
        augment_lexicon_fn=augment_lexicon_cross_situational,
        update_fn=update_perceptron_cross_situational)

  def update_with_2afc(self, sentence, model1, model2):
    """
    Observe a new `sentence` in the context of two possible scene references
    `model1` and `model2`, where `sentence` is true of at least one of the
    scenes. Update learner weights.

    Args:
      sentence: List of token strings
      model1: `Model` instance
      model2: `Model` instance

    Returns:
      weighted_results: List of weighted results for the example, where each
        result is a pair `(model, parse_result)`. `parse_result` is the CCG
        syntax/semantics parse result, and `model` identifies the scene for
        which the semantic parse is true.
    """
    return self._update_with_example(
        sentence, (model1, model2),
        augment_lexicon_fn=augment_lexicon_2afc,
        update_fn=update_perceptron_2afc)
