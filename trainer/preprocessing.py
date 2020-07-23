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
import numpy as np
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
  # Split each "<conversation>\t<response>" example into
  # a (conversation, response) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"conversation": ... "response": ...} dict.
  if reverse:
    ds = ds.map(lambda *ex: ex[::-1])
  ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
  return ds

def generic_preprocessor(ds, label):
  """Prepares text for input into model."""
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    return text

  def to_inputs_and_targets(ex):
    """Map {"conversation": ..., "response": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
            tf.strings.join(
                [label, normalize_text(ex["inputs"])]),
        "targets": normalize_text(ex["targets"])
    }
  return ds.map(to_inputs_and_targets,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

def dataset_fn_wrapper(dataset):
  path = {
    "rd_recommendations": constants.RD_TSV_PATH,
    "ml_sequences": constants.ML_SEQ_TSV_PATH,
    "ml_tags_normal": constants.ML_TAGS_TSV_PATH,
    "ml_tags_reversed": constants.ML_TAGS_TSV_PATH,
    "ml_tags_masked": constants.ML_TAGS_MASKED_TSV_PATH
  }[dataset]

  reverse = True if dataset == "ml_tags_reversed" else False
  return lambda split, shuffle_files=False: generic_dataset_fn(split, path, reverse, shuffle_files)

def reverse_example(ex):
  return {
    "inputs": ex["targets"],
    "targets": ex["inputs"]
  }

# def mask_text(ex):
#   # tf.enable_eager_execution()
#   print(ex)
#   print(tf.strings.split([ex["targets"]], sep=", ").values)
#   print(tf.expand_dims(ex['inputs'], 0))
#   tokens = tf.concat([tf.expand_dims(ex['inputs'], 0), tf.strings.split(ex["targets"], sep=", ").values], 0)
#   print(tokens)
#   print(tf.size(tokens))
#   breaks = tf.less(tf.random_uniform([tf.size(tokens) - 1]), 0.15)

#   print(breaks)
#   indecies = np.random.choice(tf.size(tokens), int(np.ceil(tf.size(tokens)*.15)))
#   sentinel_tokens = ["<extra_id_%d>" % x for x in range(len(indecies) + 1)]

#   targets = []

#   for idx, st in zip(indecies, sentinel_tokens):
#     targets.extend((st, tokens[idx]))
#     tokens[idx] = st
#   targets.append(sentinel_tokens[-1]) # Add final mask token

#   print(tf.strings.reduce_join([tokens], separator=", "))
#   return {
#     "inputs": tf.strings.reduce_join(tokens, separator=", "),
#     "targets": tf.strings.reduce_join(tokens, separator=", ")
#   }

def preprocessor_wrapper(task, ml_tags_version="normal"):
  label = {
    "rd_recommendations": "redial conversation: ",
    "ml_sequences": "movielens sequence: ",
    "ml_tags": "movielens tags: "
  }[task]
  
  # custom_function = None
  # if task == "movielens_tags":
  #   custom_function = {
  #     "normal": None,
  #     "reversed": reverse_example,
  #     # "mask": mask_text
  #   }[ml_tags_version]
  return lambda ds: generic_preprocessor(ds, label)

# ds = generic_dataset_fn("train", {"train": "./data/movielens/ml-tags-train.tsv"})
# print(list(ds.take(5).as_numpy_iterator()))
# ds = generic_preprocessor(ds, "ml tags: ", custom_function=mask_text)
# print(list(ds.take(5).as_numpy_iterator()))
ds = dataset_fn_wrapper("ml_tags_reversed")("validation")
print(list(ds.take(5).as_numpy_iterator()))
ds2 = preprocessor_wrapper("ml_tags")(ds)
print(list(ds2.take(5).as_numpy_iterator()))
ds3 = preprocessor_wrapper("ml_tags", ml_tags_version="reversed")(ds)
print(list(ds2.take(5).as_numpy_iterator()))
