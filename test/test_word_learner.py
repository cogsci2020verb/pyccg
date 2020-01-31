import sys

from nose.tools import *

from pyccg import chart
from pyccg import lexicon as lex
from pyccg import logic as log
from pyccg import scorers
from pyccg.model import Model
from pyccg.word_learner import *


def _make_mock_learner(**kwargs):
  ## Specify ontology.
  types = log.TypeSystem(["object", "location", "boolean", "action"])
  functions = [
    types.new_function("go", ("location", "action"), lambda x: ("go", x)),
  ]
  constants = [
    types.new_constant("there", "location"),
    types.new_constant("here", "location"),
    types.new_constant("away", "location"),
  ]
  ont = log.Ontology(types, functions, constants)

  ## Specify initial lexicon.
  lexicon = lex.Lexicon.fromstring(r"""
  :- S, N

  goto => S/N {\x.go(x)}
  go => S/N {\x.go(x)}
  there => N {there}
  away => N {away}
  """, ontology=ont, include_semantics=True)

  scorer = LexiconScorer(lexicon)

  return WordLearner(lexicon, learning_rate=0.1, scorer=scorer, **kwargs)


def _make_mock_model(learner):
  """
  Mock learner does not have any grounding -- just build a spurious Model
  instance.
  """
  return Model({"objects": []}, learner.ontology)


def _assert_entries_match(lexicon1, lexicon2, token):
  """
  Assert that two lexicons have the same entries for some token, but without
  checking weight differences.
  """
  eq_(set((tok.categ(), tok.semantics()) for tok in lexicon1._entries[token]),
      set((tok.categ(), tok.semantics()) for tok in lexicon2._entries[token]))


def _assert_all_entries_match(lexicon1, lexicon2):
  """
  Assert that two lexicons have entirely the same tokens and entries, but without checking weight differences.
  """
  eq_(set(lexicon1._entries.keys()), set(lexicon2._entries.keys()))
  for token in lexicon1._entries:
    _assert_entries_match(lexicon1, lexicon2, token)


def test_update_distant_existing_words():
  """
  update_distant with no novel words
  """
  sentence = "goto there".split()
  answer = ("go", "there")
  learner = _make_mock_learner()
  old_lex = learner.lexicon.clone()

  model = _make_mock_model(learner)
  results = learner.update_with_distant(sentence, model, answer)
  ok_(len(results) > 0, "Parser has >0 parses for valid sentence")

  old_lex.debug_print()
  print("====\n")
  learner.lexicon.debug_print()

  _assert_all_entries_match(old_lex, learner.lexicon)


def test_update_distant_existing_words_reinforce():
  """
  update_distant with no novel words
  """
  sentence = "goto there".split()
  answer = ("go", "there")

  learner = _make_mock_learner()
  old_lex = learner.lexicon.clone()

  model = _make_mock_model(learner)
  print(learner.lexicon.parameters())

  # Precondition for below tests to make sense: weights should start the same.
  # (If not, just change this)
  eq_(learner.lexicon._entries["goto"][0].weight(),
      learner.lexicon._entries["go"][0].weight())

  results = learner.update_with_distant(sentence, model, answer)
  print(learner.lexicon.parameters())
  ok_(len(results) > 0, "Parser has >0 parses for valid sentence")

  old_lex.debug_print()
  print("====\n")
  learner.lexicon.debug_print()

  _assert_all_entries_match(old_lex, learner.lexicon)

  # Weight update should support the used tokens
  ok_(learner.lexicon._entries["goto"][0].weight() > learner.lexicon._entries["go"][0].weight())


def test_update_distant_one_novel_word():
  """
  update_distant with one novel word
  """
  sentence = "goto here".split()
  answer = ("go", "here")
  learner = _make_mock_learner()
  old_lex = learner.lexicon.clone()

  model = _make_mock_model(learner)
  results = learner.update_with_distant(sentence, model, answer)
  ok_(len(results) > 0, "Parser has >0 parses for valid sentence")

  old_lex.debug_print()
  print("====\n")
  learner.lexicon.debug_print()

  # other words should not have changed.
  for word in ["there", "goto"]:
    _assert_entries_match(old_lex, learner.lexicon, word)

  eq_(len(learner.lexicon._entries["here"]), 1, "One valid new word entry")
  entry = learner.lexicon._entries["here"][0]
  eq_(str(entry.categ()), "N")
  eq_(str(entry.semantics()), "here")


def test_update_distant_one_novel_word_multiple_scorers():
  """
  update_distant with one novel word and multiple scorers. Make sure the
  parameters have gradients after update
  """
  sentence = "goto here".split()
  answer = ("go", "here")
  learner = _make_mock_learner()

  # Add some custom scorers to the learner
  root_types = [r"(S/N)"]
  frames = ["nsubj _"]
  learner.scorer += scorers.RootSemanticLengthScorer(learner.lexicon, root_types=root_types)
  learner.scorer += scorers.FrameSemanticsScorer(learner.lexicon, frames=frames, root_types=root_types)

  model = _make_mock_model(learner)
  sentence_meta = {"frame_str": "nsubj _"}
  results = learner.update_with_distant(sentence, model, answer, sentence_meta=sentence_meta)
  ok_(len(results) > 0, "Parser has >0 parses for valid sentence")

  learner.lexicon.debug_print()

  for _ in range(2):
    print("=====")
    old_weights = {(tok._token, tok.categ(), tok.semantics()): tok.weight().item()
                   for tok in learner.lexicon.all_entries}

    learner.update_with_distant(sentence, model, answer, sentence_meta=sentence_meta)
    learner.lexicon.debug_print()

    new_weights = {(tok._token, tok.categ(), tok.semantics()): tok.weight().item()
                   for tok in learner.lexicon.all_entries}

    eq_(set(new_weights.keys()), set(old_weights.keys()))
    ok_(not any(new_weights[k] == old_weights[k] for k in new_weights),
        "All weights should have updated")


def test_update_distant_one_novel_sense():
  """
  update_distant with one novel sense for an existing wordform
  """
  sentence = "goto there".split()
  answer = ("go", "here")
  learner = _make_mock_learner()
  old_lex = learner.lexicon.clone()

  model = _make_mock_model(learner)
  results = learner.update_with_distant(sentence, model, answer)
  ok_(len(results) > 0, "Parser has >0 parses for valid sentence")

  old_lex.debug_print()
  print("====\n")
  learner.lexicon.debug_print()

  # other words should not have changed.
  _assert_entries_match(old_lex, learner.lexicon, "goto")

  eq_(len(learner.lexicon._entries["there"]), 1, "New entry for 'there'")
  entry = learner.lexicon._entries["there"][0]
  eq_(str(entry.categ()), "N")
  eq_(str(entry.semantics()), "here")

  # expected_entries = [
  #   ("N", "here"),
  #   ("N", "there")
  # ]
  # entries = [(str(entry.categ()), str(entry.semantics())) for entry in learner.lexicon._entries["there"]]
  # eq_(set(expected_entries), entries)


def test_update_distant_two_novel_words():
  """
  update_distant with two novel words
  """
  sentence = "allez y".split()
  answer = ("go", "there")
  learner = _make_mock_learner()
  old_lex = learner.lexicon.clone()

  model = _make_mock_model(learner)
  results = learner.update_with_distant(sentence, model, answer)
  ok_(len(results) > 0, "Parser has >0 parses for sentence")

  old_lex.debug_print()
  print("======\n")
  learner.lexicon.debug_print()

  expected_entries = {
    "allez": set([("(S/N)", r"go")]),
    "y": set([("N", r"there")])
  }
  for token, expected in expected_entries.items():
    entries = learner.lexicon._entries[token]
    eq_(set([(str(e.categ()), str(e.semantics())) for e in entries]),
        expected)


def test_learner_with_frame_scorer():
  learner = _make_mock_learner()

  frames = ["A _ A", "A _ B"]
  frame_scorer = scorers.FrameSemanticsScorer(learner.lexicon, frames, root_types=[r"(S/N)"])
  learner.add_scorer(frame_scorer)

  sentence = "goto there".split()
  sentence_meta = {"frame_str": frames[0]}
  answer = ("go", "there")
  model = _make_mock_model(learner)

  results = learner.update_with_distant(sentence, model, answer,
                                        sentence_meta=sentence_meta)

  ok_(frame_scorer.frame_dist.weight.min() != 0,
      "Frame scorer weights should be updated")
