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

"""Visualize the popularity bias for a given model's Probe 1."""

import json
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import plotly.graph_objects as go
import tensorflow.compat.v1 as tf
from trainer import constants


FLAGS = flags.FLAGS

flags.DEFINE_enum("size", "base", ["small", "base", "large", "3B", "11B"],
                  "model size")
flags.DEFINE_string("name", "default", "name/description of model version")
flags.DEFINE_string("subfolder", None, ("subfolder under size folder to put ",
                                        "model in. if None, the model folder",
                                        " will be in bucket/models/size"))


def tf_load_txt(filepath):
  """Load newline separated text from gs:// using tf.io.

  Args:
    filepath: path of the file to be read

  Returns:
    a list of strings contining the lines of the file
  """
  with tf.io.gfile.GFile(filepath, "r") as txt_file:
    data = []
    for row in list(txt_file):
      data.append(str(row.replace("\n", "")))
  return data


def load_probe_data(model_dir, probe):
  """Load the probe data of a given model.

  Args:
    model_dir: the directory of a given model
    probe: the name of the probe

  Returns:
    a tuple containing the inputs, targets, predictions and steps
  """
  eval_path = os.path.join(model_dir, "validation_eval")
  inputs = [x[2:-1] for x in tf_load_txt(os.path.join(eval_path,
                                                      f"{probe}_inputs"))]
  targets = tf_load_txt(os.path.join(eval_path, f"{probe}_targets"))
  prediction_path = os.path.join(eval_path, f"{probe}*_predictions")
  prediction_files = sorted(tf.io.gfile.glob(prediction_path),
                            key=lambda x: int(x.split("_")[-2]))
  predictions = []
  steps = []

  for pred_file in prediction_files:
    ckpt_step = int(pred_file.split("_")[-2])
    steps.append(ckpt_step)
    predictions.append(tf_load_txt(pred_file))

  return inputs, targets, predictions, steps


def main(_):

  # set the model dir
  model_dir = os.path.join(constants.MODELS_DIR, FLAGS.size)
  if FLAGS.subfolder is not None:
    model_dir = os.path.join(model_dir, FLAGS.subfolder)
  model_dir = os.path.join(model_dir, FLAGS.name)

  # load the popularity data
  with tf.io.gfile.GFile(constants.MATRIX_PATHS["movie_ids"], "r") as f:
    movie_ids = json.load(f)

  # load the probe 1 data for the given model
  inputs, targets, predictions, steps = load_probe_data(model_dir, "probe_1")
  predictions = predictions[-1]
  steps = steps[-1]
  movie_ids["popularity"] = {k.lower(): v for k, v
                             in movie_ids["popularity"].items()}

  # keep track of the correctly and incorrectly classified pairs
  correct = []
  incorrect = []

  pairs = [(i, i+1) for i in range(0, len(predictions), 2)]
  for i1, i2 in pairs:
    query = inputs[i1].split("@")[1].strip()
    related = targets[i1].split("@")[1].strip()
    random = targets[i2].split("@")[1].strip()
    if (related in movie_ids["popularity"] and random in movie_ids["popularity"]
        and query in movie_ids["popularity"]):
      if float(predictions[i1]) >= float(predictions[i2]):
        correct.append((query, related, random))
      else:
        incorrect.append((query, related, random))

  correct_popularities = [movie_ids["popularity"][x[0]] for x in correct]
  incorrect_popularities = [movie_ids["popularity"][x[0]] for x in incorrect]

  # plot the correctly and incorrectly classified pairs on a histogram
  fig = go.Figure()
  fig.add_trace(go.Histogram(x=correct_popularities, name="correct"))
  fig.add_trace(go.Histogram(x=incorrect_popularities, name="incorrect"))

  fig.update_layout(barmode="overlay")
  fig.update_traces(opacity=0.5)
  fig.update_layout(
      title="Correct vs Incorrect Popularity Distributions",
      xaxis_title="Popularity",
      yaxis_title="Frequency"
  )
  fig.show()

  # log mean/median differences
  logging.info("Correct ----------")
  logging.info("mean: %d median %d", np.mean(correct_popularities),
               np.median(correct_popularities))
  logging.info("Incorrect ----------")
  logging.info("mean: %d median %d", np.mean(incorrect_popularities),
               np.median(incorrect_popularities))

if __name__ == "__main__":
  app.run(main)
