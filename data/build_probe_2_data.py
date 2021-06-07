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
  """generate probe 1 data from movielens sequences"""
  logging.info("generating probe_1.tsv")
  random.seed(42)

  with tf.io.gfile.GFile("gs://e2e_central/data/probes/co_matrix.npy", 'rb') as f:
    co_matrix = np.load(f)

  with tf.io.gfile.GFile("gs://e2e_central/data/probes/mi_matrix.npy", 'rb') as f:
    mi_matrix = np.load(f)

  with tf.io.gfile.GFile("gs://e2e_central/data/probes/movie_id_info.json", 'r') as f:
    movie_ids = json.load(f)
  
  # define "popular" set as moview which appear in over 500 user sequences
  popular_movies = list(sorted(movie_ids["all_movies"], 
                                key=lambda x: movie_ids["popularity"][x], 
                                reverse=True))

  popular_movies = [x for x in popular_movies if movie_ids["popularity"][x] >= 500]

  # filter out movies which appear in under 10 user sequences
  filtered_movies = list(sorted(movie_ids["all_movies"],
                                key=lambda x: movie_ids["popularity"][x],
                                reverse=True))
    
  filtered_movies = [x for x in filtered_movies if movie_ids["popularity"][x] >= 10]

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
    for tag in tags:

      # Find the list of popular movies not associated witht he current tag
      popular_filtered = [x for x in popular_movies if tag not in tag_data[x]]
      probes.append(f"[User] Can you recommend me a {tag} movie?\tSure, have you seen @ {movie} @?")
      probes.append(f"[User] Can you recommend me a {tag} movie?\tSure, have you seen @ {random.choice(popular_filtered)} @?")

  logging.info(f"{len(probes)} pairs generated")
  with tf.io.gfile.GFile("gs://e2e_central/data/probes/probe_2.tsv", 'w') as f:
    for line in probes:
      f.write(f"{line}\n")
if __name__ == "__main__":
  app.run(main)
