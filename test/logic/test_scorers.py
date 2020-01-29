from nose.tools import *

from pyccg import lexicon as lex
from pyccg import logic as log
from pyccg.scorers import *
from pyccg.word_learner import WordLearner


def _make_mock_learner(scorer=None, **kwargs):
  ## Specify ontology.
  types = log.TypeSystem(["object", "location", "boolean", "action"])
  functions = [
    types.new_function("go", ("location", "action"), lambda x: ("go", x)),
    types.new_function("id", ("action", "action"), lambda x: x),
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
  go => S/N {\x.id(go(x))}
  there => N {there}
  away => N {away}
  """, ontology=ont, include_semantics=True)

  if scorer is None:
    scorer = LexiconScorer(lexicon)

  return WordLearner(lexicon, scorer=scorer, **kwargs)


def _make_mock_model(learner):
  """
  Mock learner does not have any grounding -- just build a spurious Model
  instance.
  """
  return Model({"objects": []}, learner.ontology)


def test_length_scorer():
  sentence = "go there".split()
  answer = ("go", "there")

  learner = _make_mock_learner()
  length_scorer = RootSemanticLengthScorer(learner.lexicon, inverse=True, root_types=(r"(S/N)",))
  learner.scorer += length_scorer

  # model = _make_mock_model(learner)
  results = learner.make_parser().parse(sentence, return_aux=True)
  scores = [score for _, score, _ in results]

  eq_(len(results), 2)

  longer_parse_idx = next(i for i, (parse, _, _) in enumerate(results)
                          if "id" in str(parse.label()[0].semantics()))
  ok_(scores[longer_parse_idx] > scores[1 - longer_parse_idx])
