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
"""Scripts For Building Probe 3 (Movie-to-Review Recommendations)."""


import collections
import json
import os
import random

from absl import app
from absl import flags
from absl import logging
from data import build_movielens
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
import t5
from t5.data.vocabularies import SentencePieceVocabulary
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from trainer import constants

FLAGS = flags.FLAGS

delattr(FLAGS, "random_seed")
flags.DEFINE_integer("random_seed", 1, "seed for random movie selection. Choose"
                     + "-1 for a randomly picked seed")
flags.DEFINE_integer("probe_min_pop", 30, "minimum poularity to be in probe")
flags.DEFINE_integer("popular_min_pop", 138, "minimum popularity to be"
                     + " considered a popular movie")
flags.DEFINE_integer("review_snippet_length", 4, "number of sentences to be the"
                     + " max length of a review snippet in the probes")

nltk.download("punkt")
spv = SentencePieceVocabulary(t5.data.DEFAULT_SPM_PATH)


def build_review_examples(review_df, movie_ids):
  r"""transforms the dataset of reviews into trainable review examples.
  
  Uses at most five reviews from each movie in order to balance the dataset.

  Args:
    review_df: a pandas dataframe containing the review data separated into
      columns--"query" (containing a review), and "relevant_doc" (containing the
      associated movie)
    movie_ids: a dictionary containing the movie to id mapping and popularity
      information (generated in data.build_probe_1_data)

  Returns:
    a list of strings of the format:
      "Review for @ <MOVIE> @: <REVIEW START>\t<NEXT LINE IN REVIEW>"
  """
  review_map = {}

  for review, movie in tqdm(zip(review_df["query"], review_df["relevant_doc"])):
    # filter out movies which only occur once to omit spelling/input errors
    if movie in movie_ids["all_movies"] and movie_ids["popularity"][movie] > 1:
      if movie not in review_map:
        review_map[movie] = [review]
      else:
        review_map[movie].append(review)

  examples = []
  for movie, reviews in tqdm(review_map.items()):
    random.shuffle(reviews)
    for review in reviews[:5]:
      # unmask the movie title
      review = review.replace("[ITEM_NAME]", movie)
      review_start = ""
      encoded_length = 0

      # create an example for each line of the review
      for line in nltk.tokenize.sent_tokenize(review):
        examples.append(f"Review for @ {movie} @: {review_start}\t{line}")

        # ensure the input isn't too long
        encoded_length += len(spv.encode(line))
        if encoded_length < constants.INPUT_LENGTH:
          if review_start:
            review_start += " "
          review_start += line

  return examples


def main(_):
  """generate probe 3 data from reviews dataset."""
  logging.info("generating probe_3.tsv")
  # set random seed for picking random movies
  if FLAGS.random_seed != -1:
    random.seed(FLAGS.random_seed)

  with tf.io.gfile.GFile(constants.MATRIX_PATHS["movie_ids"], "r") as f:
    movie_ids = json.load(f)

  # define "popular" set as movie which appear in over FLAGS.popular_min_pop
  # user sequences

  popular_movies = [x for x in movie_ids["all_movies"]
                    if movie_ids["popularity"][x] >= FLAGS.popular_min_pop]

  logging.info("popular movies: filtered %d movies where popularity > %d",
               len(popular_movies), FLAGS.popular_min_pop)

  # define "filtered" set as movie which appear in over FLAGS.probe_min_pop
  # user sequences

  filtered_movies = [x for x in movie_ids["all_movies"]
                     if movie_ids["popularity"][x] >= FLAGS.probe_min_pop]

  logging.info("filtered movies: filtered %d movies where popularity > %d",
               len(filtered_movies), FLAGS.probe_min_pop)

  reviews_paths = tf.io.gfile.glob(os.path.join(constants.PROBE_DIR,
                                                "probe_3_reviews",
                                                "*"))
  review_df = pd.concat(pd.read_csv(f) for f in reviews_paths)

  def filter_sentences(review):
    """returns the first n sentences of a review.

    Args:
      review: a string with the review text

    Returns:
      a string representing the first n sentences of the review. n is specified
      the "review_snippet_length" flag
    """
    review = review.strip()
    sentences = nltk.tokenize.sent_tokenize(review)
    if len(sentences) > FLAGS.review_snippet_length:
      sentences = sentences[:FLAGS.review_snippet_length]
    return "".join(sentences)

  # flip titles to match training format and drop unecessary data
  review_df = review_df.copy()
  review_df["relevant_doc"] = review_df["relevant_doc"
                                        ].map(build_movielens.flip_titles)
  review_df.drop(review_df.columns.difference(["query", "relevant_doc"]),
                 1, inplace=True)
  train_df, probe_df = train_test_split(review_df, test_size=0.2,
                                        random_state=1)

  train_examples = build_review_examples(train_df, movie_ids)

  with tf.io.gfile.GFile(constants.ML_REVIEWS_TSV_PATH["train"], "w") as f:
    for line in train_examples:
      f.write(f"{line}\n")

  validation_examples = build_review_examples(probe_df, movie_ids)

  with tf.io.gfile.GFile(constants.ML_REVIEWS_TSV_PATH["validation"], "w") as f:
    for line in validation_examples:
      f.write(f"{line}\n")

  probe_df["query"] = probe_df["query"].map(filter_sentences)
  review_map = collections.defaultdict(list)

  for review, movie in zip(probe_df["query"], probe_df["relevant_doc"]):
    if movie in filtered_movies:
      review_map[movie].append(review)

  popular_movies = [x for x in popular_movies if x in review_map]

  probes = []
  for movie in review_map.keys():
    for review in review_map[movie]:
      for _ in range(4):
        random_popular = random.choice(popular_movies)
        prompt1 = f"[User] What is your opinion on @ {movie} @?"
        prompt2 = f"[User] What is your opinion on @ {random_popular} @?"
        probes.append(f"{prompt1}\t{review}")
        probes.append(f"{prompt2}\t{review}")

  logging.info("%d pairs generated", len(probes))

  with tf.io.gfile.GFile(constants.PROBE_3_TSV_PATH["validation"], "w") as f:
    for line in probes:
      f.write(f"{line}\n")
if __name__ == "__main__":
  app.run(main)
