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
"""Preprocessing Scripts for the Redial Dataset. Reads redial data unformatted from https://redialdata.github.io/website/ 
and saves it as a JSONL object of {"conversation": input, "response": output} for easy use as training data."""
import json
from tqdm import tqdm
import numpy as np
import re
import os
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "./redial", "path to folder with redial data")
flags.DEFINE_boolean("extra_redial_stats", False, "print extra summaries")

RD_UNFORMATTED_FNAMES = {
    "train": "rd-train-data.jsonl",
    "validation": "rd-test-data.jsonl"
}

RD_FORMATTED_FNAMES = {
    "train": "rd-train-formatted.jsonl",
    "validation": "rd-test-data.jsonl"
}

def main(argv):
    """Processes raw redial data in data_dir and saves the results in data folder."""
    print("--Loading Redial ataset--")
    train = read_jsonl(os.path.join(FLAGS.data_dir, RD_UNFORMATTED_FNAMES["train"]))
    test = read_jsonl(os.path.join(FLAGS.data_dir, RD_UNFORMATTED_FNAMES["test"]))


    print("--Replacing Movie IDs--")
    for dialogue in tqdm(train):
        replace_ids(dialogue)
    for dialogue in tqdm(test):
        replace_ids(dialogue)

    print("-Formatting For Training--")
    train_formatted = separate_responses(train)
    test_formatted = separate_responses(test)

    write_jsonl(os.path.join(FLAGS.data_dir, RD_FORMATTED_FNAMES["train"]), train_formatted)
    write_jsonl(os.path.join(FLAGS.data_dir, RD_FORMATTED_FNAMES["validation"]), test_formatted)
    
    if FLAGS.extra_redial_stats:
        length_summary(FLAGS.data_dir)

def length_summary(data_dir):
    """prints a five number summary of the lengths of redial input/outputs."""
    print("--Loading Dataset For Summary--")
    train_data = read_jsonl(os.path.join(data_dir, RD_FORMATTED_FNAMES["train"]))
    test_data = read_jsonl(os.path.join(data_dir, RD_FORMATTED_FNAMES["validation"]))

    def len_function(key):
        return lambda x: len(x[key].split())

    lengths = {"train_inputs" : list(map(len_function("conversation"), train_data)),
                "train_targets" : list(map(len_function("response"), train_data)),
                "test_inputs" : list(map(len_function("conversation"), test_data)),
                "test_targets" : list(map(len_function("response"), test_data))}

    for name, length_array in lengths.items():
        print(name.upper() + ": ")
        quartile_summary(length_array)

#Helper Functions

def read_jsonl(filename):
    """Reads a jsonl file and returns as array of dicts."""
    with open(filename, 'r') as json_file:
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
    """Replaces movie ids in one redial dialogue with their corresponding movie titles. Each movie is surrounded by '@'
    tokens to separate from the rest of the dialogue."""
    movie_titles = dialogue["movieMentions"]
    for message in dialogue["messages"]:
        text = message["text"]
        replaced = []
        for word in text.split():
            if word[0] == "@" and re.sub('\\D', '', word) in movie_titles:
                movie_id = re.sub('\\D', '', word)
                replaced.append( "@ " + movie_titles[movie_id] + " @")
            else:
                replaced.append(word)
        message["text"] = " ".join(replaced)

def separate_responses(dataset):
    """Creates a dataset of {"previous conversation" : "recommender response"} dictionaries for every response by the 
    recommending party in every conversation in the dataset. Turns are separated by either a [Assistant] or [User]
    token which indicates if the next messages were said by a user or an assistant"""
    result = []
    for dialogue in tqdm(dataset):
        conversation = "" # the conversation history up until a turn
        turn = "" # consecutive messages made by the same actor
        prev_id = None
        for message in dialogue["messages"]:
            if prev_id and message["senderWorkerId"] != prev_id:
                if message["senderWorkerId"] == dialogue["initiatorWorkerId"]:
                    # if the turn has switched to the user, add the (conversation, response) pair to response
                    result.append({"conversation" : conversation.strip(), "response" : turn.strip()})
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
        print(name.upper())
        print("----------------")
        print("shape: " + str(np.shape(arr)))
        print("first element :")
        print(arr[0])

def quartile_summary(arr):
    "Prints the five number summary for a 1d array"
    quartiles = np.percentile(arr, [25, 50, 75])

    print('MIN: {:d}'.format(min(arr)))
    print('Q1: {:d}'.format(int(quartiles[0])))
    print('MED: {:d}'.format(int(quartiles[1])))
    print('Q3: {:d}'.format(int(quartiles[2])))
    print('MAX: {:d}'.format(max(arr)))

if __name__ == "__main__":
   app.run(main)

