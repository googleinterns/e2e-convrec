import os
import glob
from absl import app, logging, flags
import numpy as np
import json
import pandas as pd
from collections import defaultdict
import tqdm
import matplotlib.pyplot as plt

flags.DEFINE_string("movielens_dir", "./data/movielens", "path to the movielens folder")
flags.DEFINE_enum("dataset", "both", ["tags", "sequences", "both"], "which dataset to generate from movielens: tags, movie sequences, or both")
FLAGS = flags.FLAGS

def main(_):
  
  # Define filepaths
  paths = {
    "sequences": glob.glob(os.path.join(FLAGS.movielens_dir, "ml_user_sequences/*")),
    "movies": glob.glob(os.path.join(FLAGS.movielens_dir, "ml-latest/movies.csv"))[0],
    "tags": glob.glob(os.path.join(FLAGS.movielens_dir, "ml-latest/tags.csv"))[0],
    "genome_tags": glob.glob(os.path.join(FLAGS.movielens_dir, "ml-latest/genome-tags.csv"))[0],
    "genome_scores": glob.glob(os.path.join(FLAGS.movielens_dir, "ml-latest/genome-scores.csv"))[0]
  }
  
  # Load the movie and tag id files
  genome_tags = pd.read_csv(paths["genome_tags"])
  tag_decoder = dict(zip(genome_tags.tagId, genome_tags.tag))
  movies = pd.read_csv(paths["movies"])
  movies["title"] = movies["title"].apply(flip_titles)
  movie_decoder = dict(zip(movies.movieId, movies.title))
  genre_decoder = dict(zip(movies.title, movies.genres))
  

  # Load the User Sequences
  user_seqs_strings = np.concatenate([np.loadtxt(f, dtype=str, delimiter = "\n", unpack=False) for f in paths["sequences"]])
  user_seqs_ids = list(map(parse_user_seq, user_seqs_strings))
  # user_seqs = list(map(lambda x: [movie_decoder[movieId] for movieId in x], tqdm.tqdm(user_seqs_ids)))
  
  # Load and decode the genome scores
  genome_scores = pd.read_csv(paths["genome_scores"])

  tags_dict = defaultdict(list)

  logging.info("Filtering tags with relevance < .8")
  filtered = list(filter(lambda x: x[2] > .8, tqdm.tqdm(genome_scores.values)))

  logging.info("Building tag lists")
  for movieId, tagId, _ in tqdm.tqdm(filtered):
    tags_dict[movie_decoder[movieId]].append(tag_decoder[tagId])
  
  logging.info("Generated tag lists for %d movies" % len(tags_dict))
  print(list(tags_dict.items())[-5:])

  for movie, tags in tags_dict.items():
    tags_dict[movie].extend(genre_decoder[movie].split("|"))
  
  print(list(tags_dict.items())[-5:])

  lengths = [len(x) for x in list(tags_dict.values())]
  print(lengths[-5:])
  plt.hist(lengths, bins=80)
  start, end = plt.xlim()
  plt.xticks(np.arange(0, 80, 5))
  plt.savefig("lengths.png")
  
  for movie, tags in list(tags_dict.items())[:10]:
    if len(tags) > 5:
      print(movie, ", ".join(tags))
    tags_dict[movie].extend(genre_decoder[movie].split("|"))

  # print(listed)
  # dicted = dict(listed)
  # print(dicted)
  # relisted = list(dicted.values())
  # print(relisted)
  # test = list(dict(map(rearrange, tqdm.tqdm(genome_scores.values))).values())
  # print(test[0:2])
def rearrange(line):
  return (int(line[0]), [int(line[1]), line[2]])
#   for i, row in tqdm.tqdm(genome_scores.iterrows()):
#     scores_dict[row["movieId"]].append([row["tagId"], row["relevance"]])
  
#   print(scores_dict.values[:10])

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