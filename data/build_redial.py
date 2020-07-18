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
"""Preprocessing Scripts for the Redial Dataset. Reads redial data unformatted
from https://redialdata.github.io/website/ and saves it as a JSONL object of
{"conversation": input, "response": output} for easy use as training data."""
import json
import re
import os
import numpy as np
from tqdm import tqdm
from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "./data/redial",
                    "path to folder with redial data")
flags.DEFINE_boolean("extra_redial_stats", False, "print extra summaries")

RD_UNFORMATTED_FNAMES = {
    "train": "rd-train-data.jsonl",
    "test": "rd-test-data.jsonl"
}

RD_FORMATTED_FNAMES = {
    "train": "rd-train-formatted.jsonl",
    "test": "rd-test-formatted.jsonl"
}

def main(_):
  """Processes raw redial data in data_dir and saves the results."""
  logging.info("--Loading Redial Dataset--")
  train = read_jsonl(os.path.join(FLAGS.data_dir,
                                  RD_UNFORMATTED_FNAMES["train"]))
  test = read_jsonl(os.path.join(FLAGS.data_dir,
                                 RD_UNFORMATTED_FNAMES["test"]))

  logging.info("--Replacing Movie IDs--")
  for dialogue in tqdm(train):
    replace_ids(dialogue)
  for dialogue in tqdm(test):
    replace_ids(dialogue)

  logging.info("-Formatting For Training--")
  train_formatted = separate_responses(train)
  test_formatted = separate_responses(test)

  write_jsonl(os.path.join(FLAGS.data_dir, RD_FORMATTED_FNAMES["train"]),
              train_formatted)
  write_jsonl(os.path.join(FLAGS.data_dir, RD_FORMATTED_FNAMES["test"]),
              test_formatted)

  if FLAGS.extra_redial_stats:
    length_summary(FLAGS.data_dir)

def length_summary(data_dir):
  """prints a five number summary of the lengths of redial input/outputs."""
  print("--Loading Dataset For Summary--")
  train_data = read_jsonl(os.path.join(data_dir,
                                       RD_FORMATTED_FNAMES["train"]))
  test_data = read_jsonl(os.path.join(data_dir,
                                      RD_FORMATTED_FNAMES["validation"]))

  def len_function(key):
    return lambda x: len(x[key].split())

  lengths = {
      "train_inputs": list(map(len_function("conversation"), train_data)),
      "train_targets": list(map(len_function("response"), train_data)),
      "test_inputs": list(map(len_function("conversation"), test_data)),
      "test_targets": list(map(len_function("response"), test_data))
    }

  for name, length_array in lengths.items():
    logging.info(name.upper() + ": ")
    quartile_summary(length_array)

#Helper Functions

def read_jsonl(filename):
  """Reads a jsonl file and returns as array of dicts."""
  with tf.io.gfile.GFile(filename, 'r') as json_file:
    json_list = list(json_file)
  data = []
  for json_str in tqdm(json_list):
    data.append(json.loads(json_str))
  return data

def write_jsonl(filename, arr):
  """Writes array to jsonl."""
  with open(filename, 'w') as f:
    for line in arr:
      f.write(json.dumps(line) + "\n")

def replace_ids(dialogue):
  """Replaces movie ids in one redial dialogue with their corresponding movie
  titles. Each movie is surrounded by '@' tokens to separate from the rest of
  the dialogue. Done in-place on dialogue passed as argument"""
  movie_titles = dialogue["movieMentions"]
  for message in dialogue["messages"]:
    text = message["text"]
    replaced = []
    for word in text.split():
      if word[0] == "@" and re.sub('\\D', '', word) in movie_titles:
        movie_id = re.sub('\\D', '', word)
        replaced.append("@ " + movie_titles[movie_id] + " @")
      else:
        replaced.append(word)
    message["text"] = " ".join(replaced)

def separate_responses(dataset):
  """Creates a dataset of {"previous conversation" : "recommender response"}
  dictionaries for every response by the recommending party in every
  conversation in the dataset. Turns are separated by either a [Assistant] or
  [User] token which indicates if the next messages were said by a user or an
  assistant"""
  result = []
  for dialogue in tqdm(dataset):
    conversation = "" # the conversation history up until a turn
    turn = "" # consecutive messages made by the same actor
    prev_id = None
    metadata = {
        "user_movies": [],
        "assistant_movies": []
    }

    # The initator and respondent fill out surveys labeling the movies mentioned
    # This combines their answers to account for partial responses and defaults
    # to the initiator's answer in the case of an inconsistency
    combined_responses = {**dict(dialogue["respondentQuestions"]),
                          **dict(dialogue["initiatorQuestions"])}

    if dialogue["movieMentions"] != []:
      for movie_id, title in dialogue["movieMentions"].items():
        # if the movie is not labeled, default to "assistant_movies"
        if title is None:
          title = ""
        if movie_id not in combined_responses or \
            combined_responses[movie_id]["suggested"]:
          metadata["assistant_movies"].append(title.lower())
        else:
          metadata["user_movies"].append(title.lower())

    # Adding a dummy message from the the initiator makes sure we iterate
    # through the end of the array
    empty_message = {"senderWorkerId": dialogue["initiatorWorkerId"],
                     "text": ""}
    dialogue["messages"].append(empty_message)
    for message in dialogue["messages"]:
      if (prev_id is not None) and message["senderWorkerId"] != prev_id:
        if message["senderWorkerId"] == dialogue["initiatorWorkerId"]:
          # if the turn has switched to the user, add the
          # (conversation, response) pair to response
          result.append({"conversation": conversation.strip(),
                         "response": turn.strip(), "metadata": metadata})
          conversation += " [Assistant] " + turn.strip()
        else:
          conversation += " [User] " + turn.strip()
        turn = ""
      prev_id = message["senderWorkerId"]
      turn += message["text"] + " "
  return result

def array_preview(name, arr):
  """Helper function for checking out arrays."""
  if arr:
    logging.info(name.upper())
    logging.info("----------------")
    logging.info("shape: " + str(np.shape(arr)))
    logging.info("first element :")
    logging.info(arr[0])

def conversation_preview(example):
  conversation = example["conversation"] + " [Assistant] " + example["response"]
  for message in re.split(r"\[([^]]+)\]", conversation):
    print(message)

  print("USER MENTIONED MOVIES: ")
  print(example["metadata"]["user_movies"])

  print("ASSISTANT MENTIONED MOVIES: ")
  print(example["metadata"]["assistant_movies"])

def quartile_summary(arr):
  "Prints the five number summary for a 1d array"
  quartiles = np.percentile(arr, [25, 50, 75])

  logging.info('MIN: {:d}'.format(min(arr)))
  logging.info('Q1: {:d}'.format(int(quartiles[0])))
  logging.info('MED: {:d}'.format(int(quartiles[1])))
  logging.info('Q3: {:d}'.format(int(quartiles[2])))
  logging.info('MAX: {:d}'.format(max(arr)))

if __name__ == "__main__":
  app.run(main)
