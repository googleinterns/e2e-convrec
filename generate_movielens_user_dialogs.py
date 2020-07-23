""" Utility script to generate sequences of liked movies for each user in the movielens dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import apache_beam as beam
from absl import app
from absl import flags
from absl import logging
from apache_beam.options.pipeline_options import PipelineOptions

import ml_user_watch_sequence
import dialog_pb2

flags.DEFINE_string("ratings_file_path", "data/movielens/ratings.csv",
                    "Path to the MovieLens ratings CSV file.")
flags.DEFINE_string("movies_dict_path", "data/movielens/movies.csv",
                    "Path to the MovieLens movies CSV file.")
flags.DEFINE_string(
    "output_seq_path", "output/movielens/user_watch_seq.csv",
    "Path to the output CSV files with the list of liked movies for each user.")
flags.DEFINE_string(
    "output_dialog_path",
    "output/movielens/user_dialogs.tfrecord",
    "Path to the output TFRecord with dialogs generated from the user's sequence of liked movies."
)
flags.DEFINE_integer(
    "num_ratings_per_user", 10,
    "Number of liked movies to include in the output for each user.")
flags.DEFINE_float("liked_threshold", 4.0,
                   "The minimum threshold of a positive rating.")
flags.DEFINE_integer("num_shards", 4, "Number of shards in output file.")
FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 7:
    raise app.UsageError("Too many command-line arguments.")
  ratings_file_path = FLAGS.ratings_file_path
  movies_dict_path = FLAGS.movies_dict_path
  output_seq_path = FLAGS.output_seq_path
  output_dialog_path = FLAGS.output_dialog_path
  num_ratings_per_user = FLAGS.num_ratings_per_user
  liked_threshold = FLAGS.liked_threshold
  num_shards = FLAGS.num_shards
  logging.info("Creating pipeline...")
  with beam.Pipeline() as root:
    pipeline = ml_user_watch_sequence.create_pipeline(
        input_ratings_path=ratings_file_path,
        movies_dict_path=movies_dict_path,
        liked_threshold=liked_threshold,
        num_ratings_per_user=num_ratings_per_user,
        substitute_movie_id_with_title=True)
    output_sequences, output_dialogs = pipeline(root)
    (output_sequences | "WriteSequences" >> beam.io.WriteToText(
      output_seq_path, num_shards=num_shards))
    (output_dialogs | "WriteResults" >> beam.io.WriteToTFRecord(
        output_dialog_path,
        num_shards=num_shards,
        coder=beam.coders.ProtoCoder(dialog_pb2.Dialog)))


if __name__ == "__main__":
  app.run(main)
