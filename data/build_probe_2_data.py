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
"""Scripts For Building Probe 2 (Tag-to-Movie Recommendations)."""


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
from trainer import constants

FLAGS = flags.FLAGS
flags.DEFINE_integer("random_seed", 1, "seed for random movie selection. Choose" 
                     + "-1 for a randomly picked seed")
flags.DEFINE_integer("probe_min_pop", 10, "minimum poularity to be in probe")
flags.DEFINE_integer("popular_min_pop", 500, "minimum popularity to be" 
                     + " considered a popular movie")

            
def main(_):
  """generate probe 2 data from movielens tags."""
  logging.info("generating probe_1.tsv")

  if FLAGS.random_seed != -1:
      random.seed(FLAGS.random_seed)

  with tf.io.gfile.GFile(constants.MATRIX_PATHS["movie_ids"], 'r') as f:
    movie_ids = json.load(f)
  
  # define "popular" set as movie which appear in over FLAGS.popular_min_pop 
  # user sequences

  popular_movies = [x for x in movie_ids["all_movies"]
                    if movie_ids["popularity"][x] >= FLAGS.popular_min_pop]

  # define "filtered" set as movie which appear in over FLAGS.probe_min_pop 
  # user sequences

  filtered_movies = [x for x in movie_ids["all_movies"]
                      if movie_ids["popularity"][x] >= FLAGS.probe_min_pop]

  probes = []
  tag_data = {}

  with tf.io.gfile.GFile(constants.ML_TAGS_TSV_PATH["train"], 'r') as f:
    for line in tqdm.tqdm(f):
      movie, tags = line.replace("\n", "").split("\t")
      movie = movie.strip().lower()
      tags = list(set(tags.lower().split(", ")))
      if movie in filtered_movies:
        tag_data[movie] = tags

  # filter out discrepencies between sequences and tag data
  popular_movies = [x for x in popular_movies if x in tag_data]
  print(len(popular_movies))
  print(len(tag_data))
  for movie, tags in tag_data.items():
    for tag in tags:

      # Find the list of popular movies not associated with the current tag
      popular_filtered = [x for x in popular_movies if tag not in tag_data[x]]
      pop_movie = random.choice(popular_filtered)

      prompt = f"[User] Can you recommend me a {tag} movie?"
      probes.append(f"{prompt}\tSure, have you seen @ {movie} @?")
      probes.append(f"{prompt}\tSure, have you seen @ {pop_movie} @?")

  logging.info(f"{len(probes)} pairs generated")
  with tf.io.gfile.GFile(constants.PROBE_2_TSV_PATH["validation"], 'w') as f:
    for line in probes:
      f.write(f"{line}\n")
if __name__ == "__main__":
  app.run(main)
