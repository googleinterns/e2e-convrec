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

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import functools
import os
import json

BASE_DIR = "gs://e2e_central"
DATA_DIR = os.path.join(BASE_DIR, "data")

rd_tsv_path = {
    "train": os.path.join(DATA_DIR, "rd-train.tsv"),
    "validation": os.path.join(DATA_DIR, "rd-validation.tsv")
}

def rd_jsonl_to_tsv(in_fname, out_fname):
  """Converts the redial jsonl to a tsv."""
  print("Reading: " + in_fname)
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
      print(line)
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

def rd_dataset_fn(split, shuffle_files=False):
  """Returns a tf dataset of (conversation, response) pairs for redial."""
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(rd_tsv_path[split])
  # Split each "<conversation>\t<response>" example into (conversation, response) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"conversation": ... "response": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["conversation", "response"], ex)))
  return ds

def conversation_preprocessor(ds):
  """Prepares text for input into model."""
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"conversation": ..., "response": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
            tf.strings.join(
                ["movie recommendation: ", normalize_text(ex["conversation"])]),
        "targets": normalize_text(ex["response"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

def flip_titles(title_string):
    """flips movie titles of form "Title, The (2020)" to "The Title (2020)".
    
    Args:
        title_string: a string representing a group of movie titles. For example:
            "Usual Suspects, The (1995) @ Matrix, The (1999) @ Rock, The (1996) @ Saving Private Ryan (1998)
    Returns:
        formatted string, for example:
            "The Usual Suspects (1995) @ The Matrix (1999) @  The Rock (1996) @ Saving Private Ryan (1998)"""

    movies = [title.split(" ") for title in title_string.split("@")]

    for movie in movies:
        if len(movie) > 1 and movie[-1] == "The" and movie[-2][-1] == ",":
            print(movie)
            movie.insert(0, "The")
            movie.pop()
            movie[-1] = movie[-1][:-1]
            print(movie)
    return " @ ".join([" ".join(movie) for movie in movies])
