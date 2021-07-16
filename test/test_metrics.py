# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit Tests for E2E Convrec modules."""
import unittest

from data import build_redial
import numpy as np
import sacrebleu
from trainer import metrics


class TestMetrics(unittest.TestCase):
  """Class for testing the metrics functions."""

  def test_sklearn_recall(self):
    """Tests for trainer.metrics.sklearn_recall."""
    y_true = np.array(["cat", "dog", "pig", "cat", "dog", "pig"])
    y_pred = np.array(["cat", "pig", "dog", "cat", "cat", "dog"])

    titles = np.array(["the dark knight (2008)", "star wars (1977)",
                       "spiderman (2002)"])
    titles_pred = np.array(["the wrong movie (0000)", "star wars (1977)",
                            "another mistake (2002)"])
    recall = round(metrics.sklearn_recall(y_true, y_pred)["sklearn_recall"], 2)
    self.assertEqual(recall, .33, "expected .33")

    recall = round(metrics.sklearn_recall(titles,
                                          titles_pred)["sklearn_recall"], 2)
    self.assertEqual(recall, .33, "expected .33")

  def test_replace_titles(self):
    """Tests for trainer.metrics.replace_titles."""
    test_inputs = [
        "I like the movie @ Unforgiven (2002) @ and @ Dirty Harry (1982) @",
        "@ movie (2002) @ is a good move @ haha @"
    ]

    test_outputs = [
        "I like the movie __unk__ and __unk__",
        "__unk__ is a good move @ haha @"  # shouldn't remove @ haha @ - no year
    ]

    for test_input, test_output in zip(test_inputs, test_outputs):
      self.assertEqual(([test_output], [test_output]),
                       metrics.replace_titles([test_input], [test_input]))

    self.assertEqual((test_outputs, test_outputs),
                     metrics.replace_titles(test_inputs, test_inputs))

  def test_t2t_bleu(self):
    """Tests for trainer.metrics.t2t_bleu."""
    targets = [
        "This is the first sentence",
        "this sentence is next",
        "this one is the last one"
    ]
    predictions = [
        "This sentence is first",
        "next, this one",
        "lastly we have this one"
    ]
    self.assertEqual(metrics.t2t_bleu(targets, targets)["t2t_bleu"], 100,
                     "identical text should score 100")
    sacrebleu_score = sacrebleu.corpus_bleu(predictions, [targets]).score
    t2t_score = metrics.t2t_bleu(targets, predictions)["t2t_bleu"]
    diff = abs(sacrebleu_score - t2t_score)
    self.assertLess(diff, 3, "t2t score should be similar to sacrebleu")

  def test_bleu_no_titles(self):
    """Tests for trainer.metrics.bleu_no_titles."""
    tar_titles = [
        "You should watch @ Inception (2010) @",
        ("I don't think it's bad @ all, @ Iron Man (2008) @"
         " is better though"),
        ("You should watch @ Sense and Sensibility (1995) @ or @ The "
         "Postman (Postino, Il) (1994) @")
    ]
    tar_removed = [
        "You should watch __unk__",
        "I don't think it's bad @ all, __unk__ is better though",
        "You should watch __unk__ or __unk__"
    ]
    pred_titles = [
        "I think you should watch @ Inception (2010) @",
        ("yeah but it's not as good as @ Pulp Fiction (1994) @ "
         "is better though"),
        ("You should try @ Sense and Sensibility (1995) @ or @ The "
         "Postman (Postino, Il) (1994) @")
    ]
    pred_removed = [
        "I think you should watch __unk__",
        "yeah but it's not as good as __unk__ is better though",
        "You should try __unk__ or __unk__"
    ]
    self.assertEqual(metrics.bleu_no_titles(tar_titles,
                                            pred_titles)["bleu_no_titles"],
                     metrics.t2t_bleu(tar_removed, pred_removed)["t2t_bleu"],
                     "score should equal t2t bleu with manually removed titles")

  def test_recall_from_metadata(self):
    """Tests for trainer.metrics.recall_from_metadata."""
    metadata = [
        # case 1
        {
            "user_movies": ["Inception (2010)"],
            "assistant_movies": []},
        # case 2
        {
            "user_movies": [],
            "assistant_movies": ["Iron Man (2008)"]},
        # case 3
        {
            "user_movies": ["The Postman (Postino, Il) (1994)"],
            "assistant_movies": ["Sense and Sensibility (1995)"]},
        # case 4
        {
            "user_movies": ["Iron Man (2008)"],
            "assistant_movies": ["Training Day (2001)"]},
        # case 5
        {
            "user_movies": ["Iron Man (2008)"],
            "assistant_movies": ["Richard III (1995)"]},
        # case 6
        {
            "user_movies": ["Iron Man (2008)"],
            "assistant_movies": ["Richard III (1995)"]},
        # case 7
        {
            "user_movies": [],
            "assistant_movies": []},
    ]
    predictions = [
        # case 1: 1 user movies 0 recs 0 matches -> 0
        "I think you'll like @ Inception (2010) @",
        # case 2: 0 user movies 1 recs 1 matches -> 100
        " @ Iron Man (2008) @ is a good one. I like @ Iron Man (2008) @ ",
        # case 3: 1 user movies 1 recs 1 matches -> 100
        ("You should watch @ The Postman (Postino, Il) (1994) @ or @ Sense and "
         "Sensibility (1995) @"),
        # case 4: 0 user movies 2 recs 1 matches -> 50
        "@ The Exorcist (1973) @ and @ Training Day (2001) @ are all good",
        # case 5: 0 user movies 1 recs 0 matches -> 0
        "hi how are you doing? Watch @ Star Trek: Generations (1994) @",
        # case 6: 0 user movies 1 recs 1 matches -> 100
        ("I don't think it's bad @ all, @ Richard III (1995) @, @ Richard III "
         "(1995) @ is better though"),
        # case 7: 0 user movies 0 recs 0 matches -> 0
        "what do you think you want to watch?"
    ]

    _, pred_titles = metrics.isolate_titles(predictions, predictions)

    # Test each case
    expected = [0, 100, 100, 50, 0, 100, 0]
    for pred, md, exp in zip(pred_titles, metadata, expected):
      self.assertEqual(metrics.recall_from_metadata([pred], [md]), exp,
                       "TEST CASE: pred - '%s' metadata - '%s'" % (pred, md))

    # Test all together - total 4 matches / 6 legitimate recs
    self.assertAlmostEqual(metrics.recall_from_metadata(pred_titles, metadata),
                           100 * (4.0/6))

    # load rd test dataset
    test_set = build_redial.read_jsonl("./data/redial/rd-test-formatted.jsonl")
    all_metadata = [ex["metadata"] for ex in test_set]
    targets = [ex["response"].lower() for ex in test_set]
    targets_titles, _ = metrics.isolate_titles(targets, targets)
    self.assertGreater(metrics.recall_from_metadata(targets_titles,
                                                    all_metadata), 99,
                       "Running recall on targets should result in ~100")

  def test_probe_pair_accuracy(self):
    """Tests for trainer.metrics.probe_pair_accuracy."""

    test_predictions = [
        np.arange(100),
        -1 * np.arange(100),
        [1, 2, 3, 2, 4, 5, 6, 4, 2, 6]
    ]

    expected_outputs = [0.0, 1.0, .4]

    for pred, expected_output in zip(test_predictions, expected_outputs):
      output = metrics.probe_pair_accuracy([], pred)["pair_accuracy"]
      self.assertEqual(output, expected_output,
                       f"incorrect ppa for {pred}: {output}")

if __name__ == "__main__":
  unittest.main()
