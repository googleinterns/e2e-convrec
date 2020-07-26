from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from google.protobuf import text_format

import dialog_pb2
import ml_user_watch_sequence


TEST_DATA_PATH = "test/test-data/"

class GenerateMovieLensUserWatchTest(unittest.TestCase):

  def test_generate_user_sequences(self):
    ratings_file_path = os.path.join(TEST_DATA_PATH, "ratings.csv")
    movies_dict_path = os.path.join(TEST_DATA_PATH, "movies.csv")
    num_ratings_per_user = 3
    liked_threshold = 4.0
    with TestPipeline() as root:
      pipeline = ml_user_watch_sequence.create_pipeline(
          input_ratings_path=ratings_file_path,
          movies_dict_path=movies_dict_path,
          liked_threshold=liked_threshold,
          num_ratings_per_user=num_ratings_per_user,
          substitute_movie_id_with_title=False)
      output_seqs, output_dialogs = pipeline(root)
      assert_that(output_seqs, equal_to([(1, [100, 105, 108])]))

  def test_generate_user_dialogs_with_movie_title(self):
    ratings_file_path = os.path.join(TEST_DATA_PATH, "ratings.csv")
    movies_dict_path = os.path.join(TEST_DATA_PATH, "movies.csv")
    num_ratings_per_user = 3
    liked_threshold = 4.0
    expected_dialog_proto = """
    source: "conversation_1"
events {
  speaker: "USER"
  utterance: "Toy Story (1995) @ Jumanji (1995)"
  time_ms: 1
}
events {
  speaker: "ASSISTANT"
  utterance: "Grumpier Old Men (1995)"
  time_ms: 2
}
"""
    expected_dialog = dialog_pb2.Dialog()
    text_format.Parse(expected_dialog_proto, expected_dialog)
    with TestPipeline() as root:
      pipeline = ml_user_watch_sequence.create_pipeline(
          input_ratings_path=ratings_file_path,
          movies_dict_path=movies_dict_path,
          liked_threshold=liked_threshold,
          num_ratings_per_user=num_ratings_per_user,
          substitute_movie_id_with_title=True)
      output_seqs, output_dialogs = pipeline(root)
      assert_that(output_dialogs, equal_to([expected_dialog]))

  def test_generate_user_dialogs_with_movie_id(self):
    ratings_file_path = os.path.join(TEST_DATA_PATH, "ratings.csv")
    movies_dict_path = os.path.join(TEST_DATA_PATH, "movies.csv")
    num_ratings_per_user = 3
    liked_threshold = 4.0
    expected_dialog_proto = """
    source: "conversation_1"
events {
  speaker: "USER"
  utterance: "@100 @105"
  time_ms: 1
}
events {
  speaker: "ASSISTANT"
  utterance: "@108"
  time_ms: 2
}
"""
    expected_dialog = dialog_pb2.Dialog()
    text_format.Parse(expected_dialog_proto, expected_dialog)
    with TestPipeline() as root:
      pipeline = ml_user_watch_sequence.create_pipeline(
          input_ratings_path=ratings_file_path,
          movies_dict_path=movies_dict_path,
          liked_threshold=liked_threshold,
          num_ratings_per_user=num_ratings_per_user,
          substitute_movie_id_with_title=False)
      output_seqs, output_dialogs = pipeline(root)
      assert_that(output_dialogs, equal_to([expected_dialog]))