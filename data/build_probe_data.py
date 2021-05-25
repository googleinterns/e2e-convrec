import tensorflow.compat.v1 as tf
from tqdm import tqdm
from collections import defaultdict
import sklearn
import numpy as np
import json

with tf.io.gfile.GFile("gs://e2e_central/data/ml-sequences-train.tsv", 'r') as f:
  sequence_list = list(f)
  data = []
  for sequence_str in tqdm(sequence_list):
    data.append([x.strip() for x in sequence_str.replace("\n", "").replace('\t', "").split("@") if x.strip() != ""])

with tf.io.gfile.GFile("gs://e2e_central/data/ml-sequences-test.tsv", 'r') as f:
  sequence_list = list(f)
  for sequence_str in tqdm(sequence_list):
    data.append([x.strip() for x in sequence_str.replace("\n", "").replace('\t', "").split("@") if x.strip() != ""])

# def write_json(filepath, dictionary):
#   with tf.io.gfile.GFile(filepath, 'w') as f:
#     json.dump(dictionary, filepath)

# def write_json(filepath, dictionary):
#   with tf.io.gfile.GFile(filepath, 'w') as f:
#     json.dump(dictionary, filepath)




movie_set = set()
popularity = defaultdict(int)

for seq in data:
  for movie in seq:
    movie_set.add(movie)
    popularity[movie] += 1

# for seq in data:
#   if len(set(seq)) != 10:
#     print(seq)


num_sequences = len(data)

popular_movies = list(sorted(movie_set, key=lambda x: popularity[x], reverse=True))
movie_set = sorted(movie_set)
vocab_size = len(movie_set)
embed = dict(zip(movie_set, list(range(vocab_size))))
unembed = dict(zip(list(range(vocab_size)), movie_set))

movie_ids = {
  "all_movies": movie_set,
  "movie_count": vocab_size,
  "movie_to_id": embed,
  "id_to_movie": unembed,
  "popularity": popularity
}
with tf.io.gfile.GFile("gs://e2e_central/data/probes/movie_id_info.json", 'w') as f:
  json.dump(movie_ids, f)



def create_cooccurrence(sequences):

  co_matrix = np.zeros((vocab_size, vocab_size))
  print("building cooccurrence matrix")
  for seq in tqdm(sequences):
    for movie1 in seq:
      for movie2 in seq:
        co_matrix[embed[movie1]][embed[movie2]] += 1
  return co_matrix

def get_mutual_info(co_matrix):
  total = np.sum(co_matrix)
  popularities = np.array([popularity[unembed[x]] for x in range(vocab_size)])
  pxy = co_matrix / num_sequences
  px = popularities / num_sequences
  py = (popularities / num_sequences).T

  mutual_info = np.log(pxy / (px @ py))
  return mutual_info

co_matrix = create_cooccurrence(data)
mi = get_mutual_info(co_matrix)

with tf.io.gfile.GFile("gs://e2e_central/data/probes/co_matrix.npy", 'w') as f:
  np.save(f, co_matrix)

with tf.io.gfile.GFile("gs://e2e_central/data/probes/mi_matrix.npy", 'w') as f:
  np.save(f, mi)




def get_related_movies(movie="The Lord of the Rings: The Return of the King (2003)"):
  movie_number = embed[movie]
  row = co_matrix[movie_number, :]
  return [unembed[x] for x in np.argsort(row)][::-1]

print(get_related_movies()[:10])

print("popular: ", popular_movies[:10])
print("unpopular: ", popular_movies[-10:])


def display_10(matrix):

  pop = popular_movies[:10]
  pop_ids = [embed[x] for x in pop]


  print(pop)

  print(matrix[pop_ids, :][:, pop_ids])

display_10(co_matrix)
display_10(mi)



