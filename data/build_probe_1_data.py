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
                  + "cooccurrence and MI, all to do all, probes to generate "
                  + "just the probe data without rebuilding the movie id or "
                  + "pmi matricies")
flags.DEFINE_integer("random_seed", 1, "seed for random movie selection. Choose"
                     + "-1 for a randomly picked seed")
flags.DEFINE_integer("probe_min_pop", 30, "minimum popularity to be in probe")
flags.DEFINE_integer("popular_min_pop", 138, "minimum popularity to be"
                     + " considered a popular movie")
flags.DEFINE_enum("format", "normal", ["normal", "sequences"],
                  "specify the probe format: normal for pairs in dialogue, "
                  + "sequences for movie only probes for sequences task")


def create_pmi(co_matrix, movie_ids):
  """Build pointwise mutual info matrix from cooccurrence matrix.

  Args:
    co_matrix: a cooccurence matrix off all the movies in the movielens
      sequences dataset
    movie_ids: a dictionary containing the movie to id mapping generated in this
      script

  Returns:
    a matrix pmi_matrix where:
    pmi_matrix[i][j] = pointwise_mutual_info(movie_i, movie_j)
  """

  popularities = []

  for x in range(len(movie_ids["all_movies"])):
    popularities.append(movie_ids["popularity"][movie_ids["id_to_movie"][x]])

  popularities = np.array(popularities)
  popularities[popularities < 1] = 1
  # PMI^2 is calculated as log(P(X, Y)^2 / (P(X) * P(Y)))
  pxy = co_matrix / movie_ids["num_sequences"]
  pxy[pxy == 0] = 1e-12
  px = (popularities / movie_ids["num_sequences"]).reshape((-1, 1))
  py = (popularities / movie_ids["num_sequences"]).reshape((1, -1))
  pmi = np.log((pxy**2) / np.matmul(px, py))
  return pmi


def create_cooccurrence(sequences, movie_ids):
  """Build cooccurrence matrix from list of sequences.

  Args:
    sequences: a list of lists of strings containing the 10 movies in each
      sequences in the movielens sequences dataset
    movie_ids: a dictionary containing the movie to id mapping generated in this
      script

  Returns:
    a matrix co_matrix where:
    co_matrix[i][j] = number of sequences containing both movie_i and movie_j
  """
  co_matrix = np.zeros((len(movie_ids["all_movies"]),
                        len(movie_ids["all_movies"])))

  for seq in tqdm(sequences):
    for movie1 in seq:
      for movie2 in seq:
        id1 = movie_ids["movie_to_id"][movie1]
        id2 = movie_ids["movie_to_id"][movie2]
        co_matrix[id1][id2] += 1
  return co_matrix


def create_movie_ids(sequences_data):
  """Build cooccurrence matrix from list of sequences.

  Args:
    sequences_data: a list of lists of strings containing the 10 movies in each
      sequences in the movielens sequences dataset

  Returns:
    a dictionary movie_ids wwhich keeps track of a movie to id mapping for each
    movie as well as the movie popularity and number of sequences information
  """
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
  return movie_ids


def get_related_movies(movie, movie_ids, pmi_matrix, filtered_set, k=5):
  """Get the k closest related movies as sorted by pmi.

  The results are filtered so that the related movies are above
  FLAGS.probe_min_pop popularity.

  Args:
    movie: a string representing the title of the query movie
    movie_ids: dictionary containing the movie-id mappings
    pmi_matrix: matrix containing the pmi values
    filtered_set: set of movies to filter with
    k: an int representing the number of related movies to retrieve

  Returns:
    a list of strings: the titles of the k most related movies
  """
  movie_id = movie_ids["movie_to_id"][movie]
  row = pmi_matrix[movie_id]
  related_ids = list(np.argsort(row)[::-1])
  # convert to strings and ignore the 1st most related movie (itself)
  related_ids.remove(movie_id)
  movie_titles = [movie_ids["id_to_movie"][str(x)] for x in related_ids]

  # filter out movies with popularity < FLAGS.probe_min_pop
  movie_titles = [x for x in movie_titles if x in filtered_set]
  return movie_titles[:k]


def main(_):
  """Generate probe 1 data from movielens sequences."""
  if (not tf.io.gfile.exists(constants.MATRIX_PATHS["movie_ids"]) or
      FLAGS.mode in ["all", "ids"]):

    logging.info("generating movie_id_info.json")

    def parse_sequence(sequence_str):
      sequence_str = sequence_str.replace("\n", "")
      sequence_str = sequence_str.replace("\t", "")
      return [x.strip() for x in sequence_str.split("@") if x.strip()]

    with tf.io.gfile.GFile(constants.ML_SEQ_TSV_PATH["full_train"], "r") as f:
      sequence_list = list(f)
      sequences_data = []
      for sequence_str in tqdm(sequence_list):
        sequences_data.append(parse_sequence(sequence_str))

    with tf.io.gfile.GFile(constants.ML_SEQ_TSV_PATH["full_validation"],
                           "r") as f:
      sequence_list = list(f)
      for sequence_str in tqdm(sequence_list):
        sequences_data.append(parse_sequence(sequence_str))

    movie_ids = create_movie_ids(sequences_data)

    with tf.io.gfile.GFile(constants.MATRIX_PATHS["movie_ids"], "w") as f:
      json.dump(movie_ids, f)

    logging.info("generating co_matrix.npy and pmi_matrix.npy")

    co_matrix = create_cooccurrence(sequences_data, movie_ids)
    pmi_matrix = create_pmi(co_matrix, movie_ids)

    logging.info("writing_matricies")

    with tf.io.gfile.GFile(constants.MATRIX_PATHS["co_matrix"], "w") as f:
      np.save(f, co_matrix)

    with tf.io.gfile.GFile(constants.MATRIX_PATHS["pmi_matrix"], "w") as f:
      np.save(f, pmi_matrix)

  if (not tf.io.gfile.exists(constants.PROBE_1_TSV_PATH["validation"]) or
      FLAGS.mode in ["all", "probes"]):
    logging.info("generating probe_1.tsv")

    # set random seed for picking random movies
    if FLAGS.random_seed != -1:
      random.seed(FLAGS.random_seed)

    with tf.io.gfile.GFile(constants.MATRIX_PATHS["co_matrix"], "rb") as f:
      co_matrix = np.load(f)

    with tf.io.gfile.GFile(constants.MATRIX_PATHS["pmi_matrix"], "rb") as f:
      pmi_matrix = np.load(f)

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

    filtered_set = set(filtered_movies)
    probes = []

    for movie in tqdm(filtered_movies):
      related_list = get_related_movies(movie, movie_ids, pmi_matrix,
                                        filtered_set, k=10)
      random_list = random.sample(popular_movies, k=10)

      for related, rand in zip(related_list, random_list):
        if FLAGS.format == "sequences":
          probes.append(f"@ {movie} @\t{related}")
          probes.append(f"@ {movie} @\t{rand}")
          path, extension = constants.PROBE_1_TSV_PATH["validation"].split(".")
          probe_1_path = path + "_sequences" + "." + extension
        else:
          prompt = f"[User] Can you recommend me a movie like @ {movie} @"
          probes.append(f"{prompt}\tSure, have you seen @ {related} @?")
          probes.append(f"{prompt}\tSure, have you seen @ {rand} @?")
          probe_1_path = constants.PROBE_1_TSV_PATH["validation"]

    logging.info("%d pairs generated", len(probes))
    with tf.io.gfile.GFile(probe_1_path, "w") as f:
      for line in probes:
        f.write(f"{line}\n")


if __name__ == "__main__":
  app.run(main)
