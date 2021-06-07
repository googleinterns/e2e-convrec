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
"""Scripts For Loading Predictions and Providing Evaluation Metrics."""

import os
import collections
import re
import tensorflow.compat.v1 as tf
import t5
import tensorflow_datasets as tfds
from tensor2tensor.utils import bleu_hook
from trainer import constants
from data import build_redial

def _prediction_file_to_ckpt(path):
  """Extract the global step from a prediction filename."""
  return int(path.split("_")[-2])

def load_predictions(task_name, model_dir):
  """Loads the most recent predictions in as ([(input, target, pred)], step)."""
  # Grab the dataset for this task.
  ds = t5.data.TaskRegistry.get(task_name).get_dataset(
      split="validation",
      sequence_length={"inputs": constants.INPUT_LENGTH,
                       "targets": constants.TARGET_LENGTH},
      shuffle=False)

  # Grab the paths of all logged predictions.
  prediction_files = tf.io.gfile.glob(
      os.path.join(
          model_dir,
          "validation_eval/%s_*_predictions" % task_name))

  # Get most recent prediction file by sorting by their step.
  latest_prediction_file = sorted(
      prediction_files, key=_prediction_file_to_ckpt)[-1]

  # stores the inputs, targets, predictions and checkpoint_step in a dict
  results = collections.defaultdict(list)
  results["checkpoint_step"] = _prediction_file_to_ckpt(latest_prediction_file)

  # Collect inputs, targets, and preds from the dataset and predictions file
  with tf.io.gfile.GFile(latest_prediction_file) as preds:
    for ex, pred in zip(tfds.as_numpy(ds), preds):
      results["inputs"].append(tf.compat.as_text(ex["inputs_plaintext"]))
      results["targets"].append(tf.compat.as_text(ex["targets_plaintext"]))
      results["predictions"].append(pred.strip())
  return results

def sklearn_recall(targets, predictions):
  """Uses the built in t5 sklearn_metrics_wrapper to calculate sklearn recall@1

  Args:
    targets: a list of strings, the target from the validation set
    preditcions: a list of strings, the model predictions

  Returns:
    a dictionary: {"sklearn_recall": recall_value}
  """
  prfs = t5.evaluation.metrics.sklearn_metrics_wrapper(
      "precision_recall_fscore_support",
      average='micro')(targets, predictions) \
      ["precision_recall_fscore_support"]
  return {"sklearn_recall": prfs[1]}

def t2t_bleu(targets, predictions):
  """Tokenizes with the bleu_tokenize method from the t2t library then
  calls the compute_bleu function

  Args:
    targets: a list of strings, the target from the validation set
    preditcions: a list of strings, the model predictions

  Returns:
    a dictionary: {"t2t_bleu": bleu_value}
  """
  targets_tokens = [bleu_hook.bleu_tokenize(x) for x in targets]
  predictions_tokens = [bleu_hook.bleu_tokenize(x) for x in predictions]
  return {"t2t_bleu": 100 * bleu_hook.compute_bleu(targets_tokens,
                                                   predictions_tokens)}

def replace_titles(targets, predictions):
  """Replaces titles with a __unk__ token. Returns an updated
  (targets, predictions) tuple"""
  replace_fn = \
    lambda text: re.sub(r"\@([^@]*\([^@]*\)[^@]*)\@", "__unk__", text)
  return (list(map(replace_fn, targets)), list(map(replace_fn, predictions)))

def bleu_no_titles(targets, predictions):
  """Bleu metric with titles removed. This allows for testing of the dialogue
  generation independent of movie recommendations. It also matches the format of
  previous research for easy comparison.

  Args:
    targets: a list of strings, the target from the validation set
    preditcions: a list of strings, the model predictions

  Returns:
    a dictionary: {"bleu_no_titles": bleu_value}
  """
  tars_no_titles, preds_no_titles = replace_titles(targets, predictions)
  return {"bleu_no_titles": t2t_bleu(tars_no_titles,
                                     preds_no_titles)["t2t_bleu"]}

def isolate_titles(targets, predictions):
  """Maps each target and prediction to a list of movies metioned.

  Args:
    targets: a list of strings, the target from the validation set
    preditcions: a list of strings, the model predictions

  Returns:
    a tuple containing a list of lists for both target and prediction titles
  """
  all_target_titles = []
  all_prediction_titles = []
  for tar, pred in zip(targets, predictions):
    target_titles = re.findall(r"\@([^@]*\([^@]*\)[^@]*)\@", tar)
    prediction_titles = re.findall(r"\@([^@]*\([^@]*\)[^@]*)\@", pred)
    all_target_titles.append([x.strip() for x in target_titles])
    all_prediction_titles.append([x.strip() for x in prediction_titles])
  return all_target_titles, all_prediction_titles

def recall_from_metadata(prediction_titles, all_metadata):
  """Calculates recall for movie suggestions in redial.

  Args:
    prediction_titles: a list of list containing the titles mentioned in
    each prediciton.

  Returns:
    the recall value, a float between 0 and 100
  """
  matches = 0.0
  total = 0.0
  for mentioned_movies, metadata in zip(prediction_titles, all_metadata):
    mentioned_movies = set(mentioned_movies) # get rid of duplicates
    # Recommendations are movies not menioned by the user themselves
    recs = set(filter(lambda x, md=metadata: x not in md["user_movies"],
                      mentioned_movies))
    # True positives are the recommendations which match the targets
    tp = set(filter(lambda x, md=metadata: x in md["assistant_movies"],
                    recs))
    matches += len(tp)
    total += len(recs)
  return 0 if total == 0 else 100 * matches / total

def rd_recall(targets, predictions):
  """Wrapper for rd_recall metric. Runs recall on movie mentions within the
  predictions and correct recommendations from the validation metadata

  Args:
    targets: a list of strings, the target from the validation set
    preditcions: a list of strings, the model predictions

  Returns:
    a dictionary: {"rd_recall": recall_value}
  """
  prediction_titles = isolate_titles(targets, predictions)[1]
  dataset = build_redial.read_jsonl(constants.RD_JSONL_PATH["validation"])
  all_metadata = [example["metadata"] for example in dataset]
  return {"rd_recall": recall_from_metadata(prediction_titles, all_metadata)}

def probe_pair_accuracy(targets, predictions):
  pairs = [(i, i+1) for i in range(0, len(predictions), 2)]
  correct = 0
  total = 0
  for i1, i2 in pairs:
    if predictions[i1] >= predictions[i2]:
      correct += 1
    total += 1
  
  if total == 0:
    return 0

  return {"pair accuracy" : float(correct) / total}
