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
"""Script for building the movielens tags/sequences from unformatted dataset"""
import collections
import functools
import glob
import json
import os
import random

from absl import app, logging, flags
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
import tqdm
from trainer import constants


flags.DEFINE_string("movielens_dir", "./data/movielens", \
  "path to the movielens folder")
flags.DEFINE_string("output_dir", "gs://e2e_central/data", "path to write TSVs")
flags.DEFINE_enum("task", "all", ["ml_tags", "ml_sequences", "all"], \
  "task to build datasets for: ml_tags, ml_sequences, or all")
flags.DEFINE_bool("mask", False, \
  "boolean create masked version of ml_tags dataset")
flags.DEFINE_integer("sample_rate", 3, \
  "rate for sampling multiple masked examples from one ml_tags example")
flags.DEFINE_float("seqs_test_size", .2, \
  "test split size for the sequences dataset")
flags.DEFINE_float("tags_test_size", .2, \
  "test split size for the tags dataset")
flags.DEFINE_float("mask_rate", .15, "percent of ml_tag to mask")
flags.DEFINE_bool("downsample_popular", True, \
  "downsample popular sequences data on movie frequency")
flags.DEFINE_float("downsample_cutoff", 1, "value for t for subsampling where"
                   + "examples are dropped with p=sqrt(t/frequence(x))")
flags.DEFINE_integer("random_seed", 1, "seed for random dataset spliting."
                     + "Choose -1 for a random seed")
FLAGS = flags.FLAGS

def main(_):
  """Builds the movielens datasets and saves tsvs in output_dir."""

  if FLAGS.random_seed != -1:
    random.seed(FLAGS.random_seed)

  # Define filepaths
  paths = {
      "sequences": sorted(glob.glob(os.path.join(FLAGS.movielens_dir,
                                                 "sequences", "*.csv*"))),
      "movies": os.path.join(FLAGS.movielens_dir,
                                       "ml-25m/movies.csv"),
      "tags": os.path.join(FLAGS.movielens_dir,
                                     "ml-25m/tags.csv"),
      "genome_tags": os.path.join(FLAGS.movielens_dir,
                                            "ml-25m/genome-tags.csv"),
      "genome_scores": os.path.join(FLAGS.movielens_dir,
                                              "ml-25m/genome-scores.csv")
  }

  # Load the movie and tag id files
  genome_tags = pd.read_csv(paths["genome_tags"])
  tag_decoder = dict(zip(genome_tags.tagId, genome_tags.tag))
  movies = pd.read_csv(paths["movies"])
  movies["title"] = movies["title"].apply(flip_titles)
  movie_decoder = dict(zip(movies.movieId, movies.title))
  genre_decoder = dict(zip(movies.title, movies.genres))

  if FLAGS.task in ["ml_sequences", "all"]:
    # Load the User Sequences
    logging.info("Loading MovieLens sequences")
    user_seqs_strings = np.concatenate([np.loadtxt(f, dtype=str, \
      delimiter="\n", unpack=False) for f in paths["sequences"]])
    user_seqs_ids = list(map(parse_user_seq, user_seqs_strings))
    user_seqs = list(map(lambda x: [movie_decoder[movie_id] for movie_id in x],
                         tqdm.tqdm(user_seqs_ids)))

    def downsample_data(sequences, t=1):
      """Downsample data using based on target frequency.

      Args:
        sequences: a list of strings containing the formatted sequence examples
        t: the parameter to be used as a cutoff value. examples will be dropped
          with the probability sqrt(t/f(x)) where f(x) is the frequency of the
          target

      Returns:
        a list of strings containing formatted sequence examples
      """
      random.shuffle(sequences)
      frequencies = collections.defaultdict(int)

      for sequence in sequences:
        target = sequence.split("\t")[1]
        frequencies[target] += 1

      downsampled_sequences = []
      for sequence in sequences:
        target = sequence.split("\t")[1]
        if random.random() > 1 - np.sqrt(t / frequencies[target]):
          downsampled_sequences.append(sequence)

      return downsampled_sequences


    # save a version of data with the full sequences of 10 for probes
    full_seqs = []
    for arr in tqdm.tqdm(user_seqs):
      example = "@ %s @" % " @ ".join(arr[:-1]) + "\t" + arr[-1]
      full_seqs.append(example)
    split_data = train_test_split(full_seqs,
                                  test_size=FLAGS.seqs_test_size,
                                  random_state=FLAGS.random_seed,
                                  shuffle=True)
    full_seqs_train, full_seqs_test = split_data
    write_tsv(tqdm.tqdm(full_seqs_train), constants.ML_SEQ_TSV_PATH["full_train"])
    write_tsv(tqdm.tqdm(full_seqs_test),
              constants.ML_SEQ_TSV_PATH["full_validation"])

    seqs_formatted = []
    for arr in tqdm.tqdm(user_seqs):
      random.shuffle(arr)
      for i in range(1, len(arr)):
        target = arr[i]
        example = "@ %s @" % " @ ".join(arr[:i]) + "\t" + target
        seqs_formatted.append(example)


    if FLAGS.downsample_popular:
      seqs_formatted = downsample_data(seqs_formatted,
                                       t=FLAGS.downsample_cutoff)
    seqs_train, seqs_test = train_test_split(seqs_formatted,
                                             test_size=FLAGS.seqs_test_size,
                                             random_state=FLAGS.random_seed,
                                             shuffle=True)

    # write tsvs to bucket
    logging.info("Writing TSVs")
    write_tsv(tqdm.tqdm(seqs_train), constants.ML_SEQ_TSV_PATH["train"])
    write_tsv(tqdm.tqdm(seqs_test), constants.ML_SEQ_TSV_PATH["validation"])

  if FLAGS.task in ["ml_tags", "all"]:
    # Load and decode the genome scores
    genome_scores = pd.read_csv(paths["genome_scores"])
    tags_dict = collections.defaultdict(list)

    logging.info("Filtering tags with relevance < .8")
    filtered = list(filter(lambda x: x[2] > .8, tqdm.tqdm(genome_scores.values)))

    logging.info("Building tag lists")
    for movie_id, tag_id, _ in tqdm.tqdm(filtered):
      tags_dict[movie_decoder[movie_id]].append(tag_decoder[tag_id])
    logging.info("Generated tag lists for %d movies" % len(tags_dict))

    # Add the genre data
    for movie, _ in tags_dict.items():
      tags_dict[movie].extend(genre_decoder[movie].split("|"))

    def format_tags(ex, p=.33):
      """Format tag data, sampling smaller sequences from the tag associations.

      Each subsequent tag in the sequence has a probability of p of being added
      to the current example or starting a new example.

      Args:
        ex: a (movie, tag_list) pair

      Returns:
        a list of strings containing formatted tag examples
      """
      movie, tags = ex
      tags = list(set([x.lower() for x in tags]))
      result = []
      for i in range(2):
        random.shuffle(tags)
        sublist = []
        for tag in tags:
          sublist.append(tag)
          if random.random() <= p:
            result.append(", ".join(sublist) + "\t" + movie)
            sublist = []
        if sublist != []:
          result.append(", ".join(sublist) + "\t" + movie)
      return result

    # save the association of movie -> full tags list for use in probes
    full_tags = [f"{movie}\t{', '.join(tags)}"
                 for movie, tags in tags_dict.items()]

    full_tags_train, full_tags_test = train_test_split(full_tags,
                                             test_size=FLAGS.tags_test_size,
                                             random_state=FLAGS.random_seed,
                                             shuffle=True)
    write_tsv(full_tags_train, constants.ML_TAGS_TSV_PATH["full_train"])
    write_tsv(full_tags_test, constants.ML_TAGS_TSV_PATH["full_validation"])

    # generate shorter training examples
    tags_formatted = list(flat_map(format_tags, tqdm.tqdm(tags_dict.items())))

    tags_train, tags_test = train_test_split(tags_formatted,
                                             test_size=FLAGS.tags_test_size,
                                             random_state=FLAGS.random_seed,
                                             shuffle=True)

    write_tsv(tags_train, constants.ML_TAGS_TSV_PATH["train"])
    write_tsv(tags_test, constants.ML_TAGS_TSV_PATH["validation"])

def flat_map(func, arr):
  """Maps a funciton then flattens using functools.reduce."""
  return functools.reduce(lambda a, b: a + b, map(func, arr))

def mask_multple(ex):
  """Returns a list of multiple masked example created from one example."""
  result = []
  for _ in range(FLAGS.sample_rate):
    result.append(mask_text(ex))
  return result

def mask_text(ex):
  """Masks text based on t5 strategy.

  Uses the strategy described in the t5 paper section 3.1.4
  (https://arxiv.org/abs/1910.10683) 15% of tokens are replaced with sentinel
  mask tokens
  """
  movie, tags = ex
  tokens = np.append(movie, tags.split(", "))
  indecies = np.random.choice(len(tokens), int(np.ceil(len(tokens)*FLAGS.mask_rate)))
  sentinel_tokens = ["<extra_id_%d>" % x for x in range(len(indecies) + 1)]
  targets = []

  for idx, st in zip(indecies, sentinel_tokens):
    targets.extend((st, tokens[idx]))
    tokens[idx] = st
  targets.append(sentinel_tokens[-1]) # Add final mask token

  return "\t".join([", ".join(tokens), ", ".join(targets)])

def write_tsv(arr, filepath):
  """Writes an array to a tsv."""
  with tf.io.gfile.GFile(filepath, 'w') as f:
    for line in arr:
      f.write(line + "\n")

def parse_user_seq(line):
  """Parses a sequence of ids from a line of the ml_user_sequences dataset."""
  sequence_string = line.strip('()').split(',', 1)[1]
  return json.loads(sequence_string)

def flip_titles(title_string):
  """Flips movie titles of form "Title, The (2020)" to "The Title (2020)".

  Args:
    title_string: a string representing a group of titles. For example:
      "@ Usual Suspects, The (1995) @ Matrix, The (1999) @ Rock, The (1996) @"
  Returns:
    formatted string, for example:
      "@ The Usual Suspects (1995) @ The Matrix (1999) @  The Rock (1996) @"
      """
  prefixes = set(["the", "a"])
  flipped_titles = []
  for title in title_string.split("@"):
    title = title.strip()
    fragments = title.split("(", 1)
    name = fragments[0]
    year = None if len(fragments) < 2 else "(" + fragments[1]
    name = name.split()
    # if it matcehs the form "...word word, prefix"
    if len(name) > 1 and name[-1].lower() in prefixes and name[-2][-1] == ",":
      name.insert(0, name.pop())
      name[-1] = name[-1][:-1]
    if year:
      name.append(year)
    flipped_titles.append(" ".join(name))
  return " @ ".join(flipped_titles)

if __name__ == "__main__":
  app.run(main)
