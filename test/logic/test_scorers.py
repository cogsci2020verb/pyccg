from nose.tools import *

from torch.nn import functional as F

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


def test_frame_scorer_pretrained():
  """
  Load a FrameScorer with a pretrained weight matrix, which only accounts for a
  subset of actual possible predicates.
  """
  learner = _make_mock_learner()
  ontology = learner.ontology

  all_frames = ["nsubj _", "nsubj _ dobj"]
  tracked_constants = ["there", "here"]
  untracked_constants = set(ontology.constants_dict.keys()) - set(tracked_constants)

  frame_weights = np.random.random(size=(len(all_frames), len(tracked_constants)))

  frame_scorer = FrameSemanticsScorer(learner.lexicon, all_frames,
      root_types=(r"N",),
      frame_weights=frame_weights,
      predicates=tracked_constants)
  frame_scorer.disable_gradients()
  learner.scorer = frame_scorer

  sentence = "go there".split()
  answer = ("go", "there")

  import pandas as pd
  frame_log_softmax = F.log_softmax(T.tensor(frame_weights), dim=1).numpy()
  frame_log_softmax = pd.DataFrame(frame_log_softmax, index=all_frames, columns=tracked_constants)
  print(frame_log_softmax)

  for frame in all_frames:
    results = learner.make_parser().parse(sentence,
                                          sentence_meta={"frame_str": frame},
                                          return_aux=True)

    result_weight = results[0][1]
    eq_(result_weight, frame_log_softmax.loc[frame, "there"])
