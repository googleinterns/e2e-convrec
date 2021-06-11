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
"""Scripts For Building Probe 1 (Movie-to-Movie Recommendations)."""

import collections
import json
import random

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from trainer import constants

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode", "auto", ["ids", "probes", "all", "auto"],
                  "auto to build whatever's missing, ids to rebuild ids, "
                  + "cooccurrence and MI, all to do all")


def create_mutual_info(co_matrix, movie_ids):
  """Build mutual info matrix from cooccurrence matrix."""

  popularities = []

  for x in range(len(movie_ids["all_movies"])):
    popularities.append(movie_ids["popularity"][movie_ids["id_to_movie"][x]])

  popularities = np.array(popularities)

  # PMI is calculated as log(P(X, Y) / (P(X) * P(Y)))
  pxy = co_matrix / movie_ids["num_sequences"]
  px = popularities / movie_ids["num_sequences"]
  py = (popularities / movie_ids["num_sequences"]).T
  mutual_info = np.log(pxy / (px @ py))
  return mutual_info


def create_cooccurrence(sequences, movie_ids):
  """Build cooccurrence matrix from list of sequences."""
  co_matrix = np.zeros((len(movie_ids["all_movies"]),
                        len(movie_ids["all_movies"])))
  print("building cooccurrence matrix")
  for seq in tqdm(sequences):
    for movie1 in seq:
      for movie2 in seq:
        id1 = movie_ids["movie_to_id"][movie1]
        id2 = movie_ids["movie_to_id"][movie2]
        co_matrix[id1][id2] += 1
  return co_matrix


def main(_):
  """Generate probe 1 data from movielens sequences."""

  if not (tf.io.gfile.exists(constants.MATRIX_PATHS["movie_ids"]) or
          FLAGS.mode in ["all", "ids"]):

    logging.info("generating movie_id_info.json")

    def parse_sequence(sequence_str):
      sequence_str = sequence_str.replace("\n", "")
      sequence_str = sequence_str.replace("\t", "")
      return [x.strip() for x in sequence_str.split("@") if x.strip()]

    with tf.io.gfile.GFile(constants.ML_SEQ_TSV_PATH["train"], "r") as f:
      sequence_list = list(f)
      sequences_data = []
      for sequence_str in tqdm(sequence_list):
        sequences_data.append(parse_sequence(sequence_str))

    with tf.io.gfile.GFile(constants.ML_SEQ_TSV_PATH["validation"], "r") as f:
      sequence_list = list(f)
      for sequence_str in tqdm(sequence_list):
        sequences_data.append(parse_sequence(sequence_str))

    movie_set = set()
    popularity = collections.defaultdict(int)

    # record each movie's popularity (# of sequences containing it)
    for seq in sequences_data:
      for movie in seq:
        movie_set.add(movie)
        popularity[movie] += 1

    num_sequences = len(sequences_data)
    movie_set = sorted(movie_set)
    vocab_size = len(movie_set)
    movie_to_id = dict(zip(movie_set, list(range(vocab_size))))
    id_to_movie = dict(zip(list(range(vocab_size)), movie_set))

    movie_ids = {
        "all_movies": movie_set,
        "movie_count": vocab_size,
        "movie_to_id": movie_to_id,
        "id_to_movie": id_to_movie,
        "popularity": popularity,
        "num_sequences": num_sequences
    }
    with tf.io.gfile.GFile(constants.MATRIX_PATHS["movie_ids"], "w") as f:
      json.dump(movie_ids, f)

    logging.info("generating co_matrix.npy and mi_matrix.npy")

    co_matrix = create_cooccurrence(sequences_data, movie_ids)
    mi_matrix = create_mutual_info(co_matrix, movie_ids)

    with tf.io.gfile.GFile(constants.MATRIX_PATHS["co_matrix"], "w") as f:
      np.save(f, co_matrix)

    with tf.io.gfile.GFile(constants.MATRIX_PATHS["mi_matrix"], "w") as f:
      np.save(f, mi_matrix)

  if (not tf.io.gfile.exists(constants.PROBE_1_TSV_PATH["validation"]) or
      FLAGS.mode in ["all", "probes"]):
    logging.info("generating probe_1.tsv")
    random.seed(42)

    with tf.io.gfile.GFile(constants.MATRIX_PATHS["co_matrix"], "rb") as f:
      co_matrix = np.load(f)

    with tf.io.gfile.GFile(constants.MATRIX_PATHS["mi_matrix"], "rb") as f:
      mi_matrix = np.load(f)

    with tf.io.gfile.GFile(constants.MATRIX_PATHS["movie_ids"], "r") as f:
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

    def get_related_movies(movie, k=5):
      movie_id = movie_ids["movie_to_id"][movie]
      row = mi_matrix[movie_id]
      related_ids = np.argsort(row)[::-1]
      return [movie_ids["id_to_movie"][str(x)] for x in related_ids[:k + 1]][1:]

    probes = []
    for movie in filtered_movies:
      related_list = get_related_movies(movie, k=10)
      random_list = random.sample(popular_movies, k=10)

      for related, rand in zip(related_list, random_list):
        prompt = f"[User] Can you recommend me a movie like @ {movie} @"
        probes.append(f"{prompt}\tSure, have you seen @ {related} @?")
        probes.append(f"{prompt}\t\tSure, have you seen @ {rand} @?")

    for i in range(0, 20):
      if i % 2 == 0:
        logging.info("good/random pair:")
      logging.info(probes[i])

    logging.info("%d pairs generated", len(probes))
    with tf.io.gfile.GFile(constants.PROBE_1_TSV_PATH["validation"], "w") as f:
      for line in probes:
        f.write(f"{line}\n")


if __name__ == "__main__":
  app.run(main)
