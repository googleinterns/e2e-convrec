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
"""Preprocessing and Infeeding Tools For Redial and MovieLens Data."""
import functools
import json
from absl import logging
import tensorflow.compat.v1 as tf
from trainer import constants

def rd_jsonl_to_tsv(in_fname, out_fname):
  """Converts the redial jsonl to a tsv."""
  logging.info("Reading: " + in_fname)
  def fix_spacing(text):
    """Removes extra spaces."""
    # Remove incorrect spacing around punctuation.
    text = text.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
    text = text.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
    text = text.replace("( ", "(").replace(" )", ")")
    text = text.replace("`` ", "\"").replace(" ''", "\"")
    text = text.replace(" 's", "'s").replace("s ' ", "s' ")
    return text

  count = 0
  with tf.io.gfile.GFile(in_fname, "rb") as infile,\
      tf.io.gfile.GFile(out_fname, "w") as outfile:
    for line in infile:
      ex = json.loads(line)
      conversation = fix_spacing(ex["conversation"])
      response = fix_spacing(ex["response"])
      # Write this line as <conversation>\t<response>
      outfile.write("%s\t%s\n" % (conversation, response))
      count += 1
      tf.logging.log_every_n(
          tf.logging.INFO,
          "Wrote %d examples to %s." % (count, out_fname),
          1000)
    return count

def generic_dataset_fn(split, path, reverse=False, shuffle_files=False):
  """Returns a tf dataset of (conversation, response) pairs for redial."""
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(path[split])
  # Split each "<input>\t<target>" example into
  # a (input, target) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # reverse if necessary
  if reverse:
    ds = ds.map(lambda *ex: ex[::-1])

  # Map each tuple to a {"inputs": ... "targets": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
  return ds

def generic_preprocessor(ds, label):
  """Prepares text for input into model."""
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    return text

  def to_inputs_and_targets(ex):
    """apply preprocessing functions (in this case only lowercasing) and add
    task label"""
    return {
        "inputs":
            tf.strings.join(
                [label, normalize_text(ex["inputs"])]),
        "targets": normalize_text(ex["targets"])
    }
  return ds.map(to_inputs_and_targets,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

def dataset_fn_wrapper(dataset):
  """Returns the dataset function for the desired dataset.

  Args:
    a string representing the desired dataset/task (rd_recommendations,
    ml_sequences, ml_tags_normal, ml_tags_reversed, ml_tags_masked)

  Returns:
    a function that can be passed in as a T5 dataset function
    (split, shufffle_files) -> tf.data.dataset"""
  path = {
      "rd_recommendations": constants.RD_TSV_PATH,
      "ml_sequences": constants.ML_SEQ_TSV_PATH,
      "ml_tags_normal": constants.ML_TAGS_TSV_PATH,
      "ml_tags_reversed": constants.ML_TAGS_TSV_PATH,
      "ml_tags_masked": constants.ML_TAGS_MASKED_TSV_PATH
    }[dataset]

  reverse = dataset == "ml_tags_reversed"
  return lambda split, shuffle_files=False: generic_dataset_fn(split, path, reverse, shuffle_files)

def preprocessor_wrapper(task):
  """Returns the preprocessing function for the desired task.

  Args:
    a string representing the desired task (rd_recommendations, ml_sequences,
    ml_tags)

  Returns:
    a function that can be passed in as a T5 dataset function
    (tf.data.dataset) -> tf.data.dataset"""

  label = {
      "rd_recommendations": "redial conversation: ",
      "ml_sequences": "movielens sequence: ",
      "ml_tags": "movielens tags: "
    }[task]
  return lambda ds: generic_preprocessor(ds, label)
