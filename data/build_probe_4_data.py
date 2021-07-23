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
"""Scripts For Building Probe 4 (Movie/Tag-to-Movie Recommendations)."""

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
flags.DEFINE_integer("random_seed", 1, "seed for random movie selection. Choose"
                     + "-1 for a randomly picked seed")
flags.DEFINE_integer("probe_min_pop", 30, "minimum poularity to be in probe")
flags.DEFINE_integer("popular_min_pop", 138, "minimum popularity to be"
                     + " considered a popular movie")


def main(_):
  """generate probe 4 data from movielens tags."""
  logging.info("generating probe_4.tsv")
  # set random seed for picking random movies
  if FLAGS.random_seed != -1:
    random.seed(FLAGS.random_seed)

  with tf.io.gfile.GFile(constants.MATRIX_PATHS["pmi_matrix"], "rb") as f:
    pmi_matrix = np.load(f)

  with tf.io.gfile.GFile(constants.MATRIX_PATHS["movie_ids"], "r") as f:
    movie_ids = json.load(f)

  # define "popular" set as movie which appear in over FLAGS.popular_min_pop
  # user sequences

  popular_movies = [x.lower() for x in movie_ids["all_movies"]
                    if movie_ids["popularity"][x] >= FLAGS.popular_min_pop]

  logging.info("popular movies: filtered %d movies where popularity > %d",
               len(popular_movies), FLAGS.popular_min_pop)

  # define "filtered" set as movie which appear in over FLAGS.probe_min_pop
  # user sequences

  filtered_movies = [x.lower() for x in movie_ids["all_movies"]
                     if movie_ids["popularity"][x] >= FLAGS.probe_min_pop]

  logging.info("filtered movies: filtered %d movies where popularity > %d",
               len(filtered_movies), FLAGS.probe_min_pop)

  # lowercase movie titles to match ml_tags data
  movie_ids["all_movies"] = [x.lower() for x in movie_ids["all_movies"]]
  ids = list(range(len(movie_ids["all_movies"])))
  movie_ids["movie_to_id"] = dict(zip(movie_ids["all_movies"], ids))
  movie_ids["id_to_movie"] = dict(zip(ids, movie_ids["all_movies"]))

  def intersection(list1, list2):
    return [x for x in list1 if x in list2]

  filtered_set = set(filtered_movies)
  def get_related_movies(movie, k=5):
    """Get the k closest related movies as sorted by pmi.

    Args:
      movie: a string representing the title of the query movie
      k: an int representing the number of related movies to retrieve

    Returns:
      a list of strings: the titles of the k most related movies
    """
    movie_id = movie_ids["movie_to_id"][movie]
    row = pmi_matrix[movie_id]
    related_ids = list(np.argsort(row)[::-1])
    # convert to strings and ignore the 1st most related movie (itself)
    related_ids.remove(movie_id)
    movie_titles = [movie_ids["id_to_movie"][x] for x in related_ids]

    # filter out movies with popularity < 10
    movie_titles = [x for x in movie_titles if x in filtered_set]
    return movie_titles[:k]

  probes = []
  tag_data = {}

  with tf.io.gfile.GFile(constants.ML_TAGS_V1_TSV_PATH["train"], "r") as f:
    for line in tqdm(f):
      movie, tags = line.replace("\n", "").split("\t")
      movie = movie.strip().lower()
      tags = list(set(tags.lower().split(", ")))
      if movie in filtered_set:
        tag_data[movie] = tags

  # filter out discrepencies between sequences and tag data
  popular_movies = [x for x in popular_movies if x in tag_data]
  for movie, tags in tag_data.items():
    related_movies = get_related_movies(movie, k=10)
    for related in related_movies:
      if related in tag_data:
        common_tags = intersection(tags, tag_data[related])
        if len(common_tags) > 5:
          common_tags = random.sample(common_tags, k=5)
        for tag in intersection(tags, tag_data[related]):
          # Find the list of popular movies not associated with the current tag
          popular_filtered = [x for x in popular_movies
                              if tag not in tag_data[x]]
          rand = random.choice(popular_filtered)
          inp = f"[User] Can you recommend me a {tag} movie like @ {movie} @?"
          probes.append(f"{inp}?\tSure, have you seen @ {related} @?")
          probes.append(f"{inp}\tSure, have you seen @ {rand} @?")

  logging.info("%d pairs generated", len(probes))
  with tf.io.gfile.GFile(constants.PROBE_4_TSV_PATH["validation"], "w") as f:
    for line in probes:
      f.write(f"{line}\n")
if __name__ == "__main__":
  app.run(main)
