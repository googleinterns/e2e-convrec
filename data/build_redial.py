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
"""Preprocessing Scripts for the Redial Dataset."""
import json
from tqdm import tqdm
import numpy as np
import re
import os

def prepare_redial(data_dir):
    """Processes raw redial data in data_dir and saves the results in data folder."""
    print("--Loading Redial Dataset--")
    train = read_jsonl(os.path.join(data_dir, "rd-train-data.jsonl"))
    test = read_jsonl(os.path.join(data_dir,"rd-test-data.jsonl"))


    print("--Replacing Movie IDs--")
    for dialogue in tqdm(train):
        replace_ids(dialogue)
    for dialogue in tqdm(test):
        replace_ids(dialogue)

    print("-Formatting For Training--")
    train_formatted = separate_responses(train)
    test_formatted = separate_responses(test)

    write_jsonl("./redial/rd-train-formatted.jsonl", train_formatted)
    write_jsonl("./redial/rd-test-formatted.jsonl", test_formatted)

def length_summary(data_dir):
    """prints a five number summary of the lengths of redial input/outputs."""
    print("--Loading Dataset For Summary--")
    train_data = read_jsonl(os.path.join(data_dir, "rd-train-formatted.jsonl"))
    test_data = read_jsonl(os.path.join(data_dir, "rd-train-formatted.jsonl"))

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
    """Replaces movie ids in one redial dialogue with their corresponding movie titles."""
    movie_titles = dialogue["movieMentions"]
    for message in dialogue["messages"]:
        text = message["text"]
        replaced = []
        for word in text.split():
            if word[0] == "@" and re.sub('\\D', '', word) in movie_titles:
                replaced.append( "@ " + movie_titles[re.sub('\\D', '', word)] + " @")
            else:
                replaced.append(word)
        message["text"] = " ".join(replaced)

def separate_responses(dataset):
    """Creates a dataset of {"previous conversation" : "recommender response"} dictionaries
    for every response by the recommending party in every conversation in the dataset."""
    result = []
    for dialogue in tqdm(dataset):
        conversation = "" # the conversation history up until a turn
        turn = "" # consecutive messags made by the same 
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
    prepare_redial("./redial")

