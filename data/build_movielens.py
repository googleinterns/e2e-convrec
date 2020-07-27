import os
import glob
from absl import app, logging, flags
import numpy as np
import json
import pandas as pd
from collections import defaultdict
import tqdm
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from functools import reduce

flags.DEFINE_string("movielens_dir", "./data/movielens", "path to the movielens folder")
flags.DEFINE_enum("dataset", "both", ["tags", "sequences", "both"], "which dataset to generate from movielens: tags, movie sequences, or both")
flags.DEFINE_string("output_dir", "gs://e2e_central/data", "path to write datasets")
flags.DEFINE_enum("task", "all", ["ml_tags", "ml_sequences", "all"], "task to build datasets for: ml_tags, ml_sequences, or all")
flags.DEFINE_bool("mask", False, "boolean create masked version of ml_tags dataset")
flags.DEFINE_integer("sample_rate", 3, "sampling rate for creating multiple masked examples from one ml_tags example")
FLAGS = flags.FLAGS

def main(_):
  
  # Define filepaths
  paths = {
    "sequences": sorted(glob.glob(os.path.join("./output/movielens", "*.csv*"))),
    "movies": glob.glob(os.path.join(FLAGS.movielens_dir, "ml-25m/movies.csv"))[0],
    "tags": glob.glob(os.path.join(FLAGS.movielens_dir, "ml-25m/tags.csv"))[0],
    "genome_tags": glob.glob(os.path.join(FLAGS.movielens_dir, "ml-25m/genome-tags.csv"))[0],
    "genome_scores": glob.glob(os.path.join(FLAGS.movielens_dir, "ml-25m/genome-scores.csv"))[0]
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
    user_seqs_strings = np.concatenate([np.loadtxt(f, dtype=str, delimiter = "\n", unpack=False) for f in paths["sequences"]])
    user_seqs_ids = list(map(parse_user_seq, user_seqs_strings))
    user_seqs = list(map(lambda x: [movie_decoder[movieId] for movieId in x], tqdm.tqdm(user_seqs_ids)))

    def format_sequence(arr):
      inputs = "@ %s @" % " @ ".join(arr[:-1])
      targets = arr[-1]
      return "\t".join((inputs, targets))


    seqs_formatted = list(map(format_sequence, user_seqs))
    seqs_train, seqs_test = train_test_split(seqs_formatted, test_size=.2, random_state=None, shuffle=False)

    # writ tsvs to bucket
    logging.info("Writing TSVs")
    write_tsv(tqdm.tqdm(seqs_train), os.path.join(FLAGS.output_dir, "ml-sequences-train.tsv"))
    write_tsv(tqdm.tqdm(seqs_test), os.path.join(FLAGS.output_dir, "ml-sequences-validation.tsv"))

  if FLAGS.task in ["ml_tags", "all"]:
    # Load and decode the genome scores
    genome_scores = pd.read_csv(paths["genome_scores"])

    tags_dict = defaultdict(list)

    logging.info("Filtering tags with relevance < .8")
    filtered = list(filter(lambda x: x[2] > .8, tqdm.tqdm(genome_scores.values)))

    logging.info("Building tag lists")
    for movieId, tagId, _ in tqdm.tqdm(filtered):
      tags_dict[movie_decoder[movieId]].append(tag_decoder[tagId])
    
    logging.info("Generated tag lists for %d movies" % len(tags_dict))

    # Add the genre data
    for movie, _ in tags_dict.items():
      tags_dict[movie].extend(genre_decoder[movie].split("|"))

    tags_formatted = ["\t".join((movie, ", ".join(tags))) for movie, tags in list(tags_dict.items())]

    if FLAGS.mask:
      tags_formatted = list(flat_map(mask_multple, tqdm.tqdm(tags_formatted)))
    tags_train, tags_test = train_test_split(tags_formatted, test_size=.2, random_state=1)

    # Write ml_tags TSVs
    modifier = ""
    if FLAGS.mask:
      modifier = "-masked-" + str(FLAGS.sample_rate)
    write_tsv(tags_train, os.path.join(FLAGS.output_dir, "ml-tags-train%s.tsv" % modifier))
    write_tsv(tags_test, os.path.join(FLAGS.output_dir, "ml-tags-validation%s.tsv" % modifier))

def flat_map(func, arr):
  return reduce(lambda a, b: a + b, map(func, arr))

def mask_multple(ex):
  result = []
  for i in range(FLAGS.sample_rate):
    result.append(mask_text(ex))
  return result

def mask_text(ex):
  movie, tags = ex.split("\t")
  tokens = np.append(movie, tags.split(", "))
  indecies = np.random.choice(len(tokens), int(np.ceil(len(tokens)*.15)))
  sentinel_tokens = ["<extra_id_%d>" % x for x in range(len(indecies) + 1)]

  targets = []

  for idx, st in zip(indecies, sentinel_tokens):
    targets.extend((st, tokens[idx]))
    tokens[idx] = st
  targets.append(sentinel_tokens[-1]) # Add final mask token

  return "\t".join([", ".join(tokens), ", ".join(targets)])

def write_tsv(arr, filepath):
  with tf.io.gfile.GFile(filepath, 'w') as f:
    for line in arr:
      f.write(line + "\n")

def parse_user_seq(line):
  sequence_string = line.strip('()').split(',', 1)[1]
  return json.loads(sequence_string)

def flip_titles(title_string):
    """flips movie titles of form "Title, The (2020)" to "The Title (2020)".
    
    Args:
        title_string: a string representing a group of movie titles. For example:
            "Usual Suspects, The (1995) @ Matrix, The (1999) @ Rock, The (1996) @ Saving Private Ryan (1998)
    Returns:
        formatted string, for example:
            "The Usual Suspects (1995) @ The Matrix (1999) @  The Rock (1996) @ Saving Private Ryan (1998)"""
    prefixes = set(["the", "a"])
    flipped_titles = []
    for title in title_string.split("@"):
        title = title.strip()
        fragments = title.split("(", 1)
        name = fragments[0]
        year = None if len(fragments) < 2 else "(" + fragments[1]
        name = name.split()
        if len(name) > 1 and name[-1].lower() in prefixes and name[-2][-1] == ",":
            name.insert(0, name.pop())
            name[-1] = name[-1][:-1]
        if year:
            name.append(year)
        flipped_titles.append(" ".join(name))
    return " @ ".join(flipped_titles)

if __name__ == "__main__":
  app.run(main)