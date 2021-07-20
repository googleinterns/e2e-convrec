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

"""Unit Tests for Probe Scripts."""
import unittest

from data import build_probe_1_data
import numpy as np


class TestBuildProbes(unittest.TestCase):

  def test_movie_ids(self):
    """Tests for data.build_probe_1_data.create_movie_ids."""
    sequences = [
        ["a", "b", "c"],
        ["b", "c", "d"],
        ["c"]
    ]

    unique_elements = ["a", "b", "c", "d"]
    ids = [0, 1, 2, 3]
    popularities = [1, 2, 3, 1]
    movie_ids = build_probe_1_data.create_movie_ids(sequences)
    self.assertEqual(movie_ids["all_movies"],
                     unique_elements,
                     "all_movies not correctly initialized")
    self.assertEqual(movie_ids["movie_to_id"],
                     dict(zip(unique_elements, ids)),
                     "incorrect movie to id mapping")
    self.assertEqual(movie_ids["id_to_movie"],
                     dict(zip(ids, unique_elements)),
                     "incorrect id to movie mapping")
    self.assertEqual(movie_ids["popularity"],
                     dict(zip(unique_elements, popularities)),
                     "incorrect item popularity mapping")
    self.assertEqual(movie_ids["num_sequences"],
                     len(sequences),
                     "incorrect number of sequences")

  def test_cooccurence(self):
    """Tests for data.build_probe_1_data.create_cooccurrence."""
    sequences = [
        ["a", "b", "c"],
        ["b", "c", "d"],
        ["c"]
    ]
    movie_ids = build_probe_1_data.create_movie_ids(sequences)
    co = build_probe_1_data.create_cooccurrence(sequences, movie_ids)
    expected_co = np.array([
        [1, 1, 1, 0],
        [1, 2, 2, 1],
        [1, 2, 3, 1],
        [0, 1, 1, 1]
    ])
    self.assertTrue(np.array_equal(co, expected_co), "incorrect co_matrix calc")

  def test_pmi(self):
    """Tests for data.build_probe_1_data.create_pmi."""
    def calc_pmi2(co_ab, pop_a, pop_b, num_seq):
      """basic pmi2 calculation."""
      p_ab = co_ab / num_seq
      p_a = pop_a / num_seq
      p_b = pop_b / num_seq
      return np.log(p_ab**2 / (p_a*p_b))

    sequences = [
        ["a", "b", "c"],
        ["b", "c", "d"],
        ["c"]
    ]

    movie_ids = build_probe_1_data.create_movie_ids(sequences)
    co = build_probe_1_data.create_cooccurrence(sequences, movie_ids)
    pops = [movie_ids["popularity"][x] for x in movie_ids["all_movies"]]
    num_seqs = movie_ids["num_sequences"]
    pmi2 = build_probe_1_data.create_pmi(co, movie_ids)
    expected_pmi2 = np.zeros_like(pmi2)

    for i in range(len(movie_ids["all_movies"])):
      for j in range(len(movie_ids["all_movies"])):
        expected_pmi2[i][j] = calc_pmi2(co[i][j], pops[i], pops[j], num_seqs)

    self.assertTrue(np.allclose(np.exp(pmi2), np.exp(expected_pmi2)),
                    "pmi2 calculation incorrect")

if __name__ == "__main__":
  unittest.main()
