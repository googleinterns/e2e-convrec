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


import tensorflow.compat.v1 as tf
from tqdm import tqdm
from collections import defaultdict
import sklearn
import numpy as np
import json
import random
from absl import app
from absl import flags
from absl import logging
import tqdm
import pandas as pd
import os
import nltk
from trainer import constants
from data import build_movielens
from sklearn.model_selection import train_test_split

nltk.download('punkt')

def build_review_examples(review_df, movie_ids):
  review_map = {}

  for review, movie in zip(review_df['query'], review_df['relevant_doc']):
    if movie in movie_ids["all_movies"] and movie_ids["popularity"][movie] > 1:
      if movie not in review_map:
        review_map[movie] = [review]
      else:
        review_map[movie].append(review)
  examples = []

  for movie, reviews in review_map.items():
    for review in reviews[:5]:
      review_start = ""
      for line in nltk.tokenize.sent_tokenize(review):
        examples.append(f"Review for @ {movie} @: {review_start}\t{line}")
        if review_start != "":
          review_start += " "
        review_start += line
  
  return examples
def main(_):
  """generate probe 3 data from reviews dataset."""
  logging.info("generating probe_3.tsv")
  random.seed(42)
  with tf.io.gfile.GFile(os.path.join(constants.MATRIX_PATHS["movie_ids"]), "r") as f:
    movie_ids = json.load(f)

  # define "popular" set as moview which appear in over 500 user sequences
  popular_movies = list(sorted(movie_ids["all_movies"],
                                key=lambda x: movie_ids["popularity"][x],
                                reverse=True))

  popular_movies = [x for x in popular_movies
                    if movie_ids["popularity"][x] >= 500]

  # filter out movies which appear in under 10 user sequences
  filtered_movies = list(sorted(movie_ids["all_movies"],
                                key=lambda x: movie_ids["popularity"][x],
                                reverse=True))

  filtered_movies = [x for x in filtered_movies
                      if movie_ids["popularity"][x] >= 10]
  reviews_paths = tf.io.gfile.glob(os.path.join(constants.PROBE_DIR,
                                                "probe_3_reviews",
                                                "*"))
  
  review_df = pd.concat(pd.read_csv(f) for f in reviews_paths)


  def filter_sentences(review):
    review = review.strip()
    sentences = nltk.tokenize.sent_tokenize(review)
    if len(sentences) > 4:
      sentences = sentences[:4]
    return "".join(sentences)

  filtered = review_df.copy()
  
  filtered['relevant_doc'] = filtered['relevant_doc'].map(build_movielens.flip_titles)
  filtered.drop(filtered.columns.difference(['query', 'relevant_doc']), 1, inplace=True)
  train_df, probe_df = train_test_split(filtered, test_size=0.2, random_state=1)
  
  train_examples = build_review_examples(train_df, movie_ids)

  with tf.io.gfile.GFile(constants.ML_REVIEWS_TSV_PATH["train"], "w") as f:
    for line in train_examples:
      f.write(f"{line}\n")

  validation_examples = build_review_examples(probe_df, movie_ids)

  with tf.io.gfile.GFile(constants.ML_REVIEWS_TSV_PATH["validation"], "w") as f:
    for line in validation_examples:
      f.write(f"{line}\n")
  
  probe_df['query'] = probe_df['query'].map(filter_sentences)
  review_map = {}

  for review, movie in zip(probe_df['query'], probe_df['relevant_doc']):
    if movie not in review_map:
      review_map[movie] = [review]
    else:
      review_map[movie].append(review)
  # filtered_movies = [x for x in review_map.keys() if len(review_map[x]) >= 10]
  popular_movies = [x for x in popular_movies if x in review_map]
  print(len(popular_movies))
  probes = []
  for movie in review_map.keys():
    for review in review_map[movie]:
      random_popular = random.choice(popular_movies)
      # random_review = random.choice(review_map[random_popular])
      probes.append(f"[User] What is your opinion on @ {movie} @?\t{review}")
      probes.append(f"[User] What is your opinion on @ {random_popular} @?\t{review}")
  
  logging.info("%d pairs generated", len(probes))

  with tf.io.gfile.GFile(constants.PROBE_3_TSV_PATH["validation"], "w") as f:
    for line in probes:
      f.write(f"{line}\n")
if __name__ == "__main__":
  app.run(main)
