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

import tensorflow.compat.v1 as tf
import t5
import os
import random
import nltk
import sacrebleu
from trainer import constants
import tensorflow_datasets as tfds
import json

def _prediction_file_to_ckpt(path):
  """Extract the global step from a prediction filename."""
  return int(path.split("_")[-2])

def load_predictions(task_name, model_dir):
  """Loads the most recent predictions in as ([(input, target, pred)], step)."""
  # Grab the dataset for this task.
  ds = t5.data.TaskRegistry.get(task_name).get_dataset(
      split="validation",
      sequence_length={"inputs": constants.INPUT_LENGTH, "targets": constants.TARGET_LENGTH},
      shuffle=False)

  # Grab the paths of all logged predictions.
  prediction_files = tf.io.gfile.glob(
      os.path.join(
          model_dir,
          "validation_eval/%s_*_predictions" % task_name))

  # Get most recent prediction file by sorting by their step.
  latest_prediction_file = sorted(
      prediction_files, key=_prediction_file_to_ckpt)[-1]

  checkpoint_step =_prediction_file_to_ckpt(latest_prediction_file)

  # Collect (inputs, targets, prediction) from the dataset and predictions file
  results = []
  with tf.io.gfile.GFile(latest_prediction_file) as preds:
    for ex, pred in zip(tfds.as_numpy(ds), preds):
      results.append((tf.compat.as_text(ex["inputs_plaintext"]),
                      tf.compat.as_text(ex["targets_plaintext"]),
                      pred.strip()))
  return (results, checkpoint_step)

def save_metrics(task_name, model_dir):
  """Prints and saves metrics for the most recent checkpoint."""
  results, checkpoint_step = load_predictions(task_name, model_dir)
  predictions = [line[2] for line in results]
  targets = [line[1] for line in results]

  hyp = list(map(lambda x: x.split(), predictions))
  ref = list(map(lambda x: [x.split()], targets))

  nltk_bs = nltk.translate.bleu_score.corpus_bleu(list_of_references=ref, hypotheses=hyp)
  sb_bs = str(sacrebleu.corpus_bleu(predictions, [targets]))

  print("NLTK BLEU SCORE: {:f}, SACREBLEU BLEU SCORE: {:s}, CHECKPOINT: {:d}".format(nltk_bs, sb_bs, checkpoint_step))
  # Writes to $MODEL_DIR$/validation_eval/metrics$CHECKPOINT_NUMBER.json
  metrics_path = os.path.join(
          model_dir,
          "validation_eval/metrics" + str(checkpoint_step) + ".json")
  json.dump({"nltk_bleu_score" : nltk_bs, "sacrebleu_blue_score" : sb_bs, "recall@1" : 0}, tf.io.gfile.GFile(metrics_path, "w"))

