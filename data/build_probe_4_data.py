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


def main(_):
  """generate probe 4 data from movielens tags."""
  logging.info("generating probe_1.tsv")
  random.seed(42)

  with tf.io.gfile.GFile("gs://e2e_central/data/probes/co_matrix.npy", 'rb') as f:
    co_matrix = np.load(f)

  with tf.io.gfile.GFile("gs://e2e_central/data/probes/mi_matrix.npy", 'rb') as f:
    mi_matrix = np.load(f)

  with tf.io.gfile.GFile("gs://e2e_central/data/probes/movie_id_info.json", 'r') as f:
    movie_ids = json.load(f)
  
  

  # define "popular" set as moview which appear in over 500 user sequences
  popular_movies = [x.lower() for x in movie_ids["all_movies"] if movie_ids["popularity"][x] >= 500]
  
  # filter out movies which appear in under 10 user sequences
  filtered_movies = [x.lower() for x in movie_ids["all_movies"] if movie_ids["popularity"][x] >= 10]

  movie_ids["all_movies"] = [x.lower() for x in movie_ids["all_movies"]]
  movie_ids["movie_to_id"] = dict(zip(movie_ids["all_movies"], list(range(len(movie_ids["all_movies"])))))
  movie_ids["id_to_movie"] = dict(zip(list(range(len(movie_ids["all_movies"]))), movie_ids["all_movies"]))
  def intersection(list1, list2):
    return [x for x in list1 if x in list2]
  
  def get_related_movies(movie, k=5):
    movie_id = movie_ids["movie_to_id"][movie]
    row = mi_matrix[movie_id]
    related_ids = np.argsort(row)[::-1]
    return [movie_ids["id_to_movie"][x] for x in related_ids[:k + 1]][1:]

  probes = []
  tag_data = {}

  with tf.io.gfile.GFile("gs://e2e_central/data/ml-tags-train.tsv", 'r') as f:
    for line in tqdm.tqdm(f):
      movie, tags = line.replace("\n", "").split("\t")
      movie = movie.strip().lower()
      tags = list(set(tags.lower().split(", ")))
      if movie in filtered_movies:
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
          popular_filtered = [x for x in popular_movies if tag not in tag_data[x]]
          probes.append(f"[User] Can you recommend me a {tag} movie like @ {movie} @?\tSure, have you seen @ {related} @?")
          probes.append(f"[User] Can you recommend me a {tag} movie like @ {movie} @?\tSure, have you seen @ {random.choice(popular_filtered)} @?")

  logging.info(f"{len(probes)} pairs generated")
  with tf.io.gfile.GFile(constants.PROBE_4_TSV_PATH["validation"], 'w') as f:
    for line in probes:
      f.write(f"{line}\n")
if __name__ == "__main__":
  app.run(main)
